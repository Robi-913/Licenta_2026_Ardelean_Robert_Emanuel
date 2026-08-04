import argparse
import os
import random
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageFilter
from transformers import AutoProcessor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Biblioteca pytorch_grad_cam - implementari standard de CAM pt PyTorch
from pytorch_grad_cam import EigenCAM, GradCAM, GradCAMPlusPlus, LayerCAM
from pytorch_grad_cam.utils.image import show_cam_on_image  # suprapune heatmap pe imagine

from src.datasets.oct5k_medsiglip import OCT5kDataset
from src.model.medsiglip import MedSigLIPMultiTask
from src.utils.seed import set_seed


CHECKPOINT = "experiments/medsiglip_v15/ckpts/final_with_probe.pth"  # modelul complet (backbone + probe head)
MODEL_PATH = "models/medsiglip-448"  # procesorul SigLIP (tokenizer + image processor)
OUTPUT_DIR = "experiments/figures/gradcam"  # directorul unde se salveaza heatmap-urile
CLASSES = ["AMD", "DME", "DRUSEN", "NORMAL"]  # cele 4 clase de boala
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Dictionar cu metodele CAM disponibile
# "rollout" e un string special - nu foloseste biblioteca pytorch_grad_cam
CAM_METHODS = {
    "gradcam": GradCAM,
    "gradcam++": GradCAMPlusPlus,
    "eigencam": EigenCAM,
    "layercam": LayerCAM,
    "rollout": "rollout",
}


class GradCAMWrapper(torch.nn.Module):
    """
    Wrapper care simplifica modelul MedSigLIP pentru biblioteca GradCAM.

    Problema: MedSigLIP e multi-task - primeste imagine + 2 texte si returneaza
    7 output-uri (img_emb, emb_a, emb_b, fused_emb, logit_scale, sev_pred, cls_logits).
    Biblioteca GradCAM se asteapta la un model simplu: input -> output unic.

    Solutia: wrapper-ul expune DOAR calea imagine -> clasificare.
    GradCAM calculeaza gradientii de la cls_logits inapoi prin backbone,
    deci are nevoie de o cale clara fara ramificatii.
    """

    def __init__(self, model: MedSigLIPMultiTask):
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # encode_image: imagine -> embedding vizual (trece prin ViT + LoRA + projection)
        image_pooled = self.model.encode_image(pixel_values)
        # classification_head: embedding -> logits pt 4 clase
        return self.model.classification_head(image_pooled)


def load_model(n_classes: int = 4) -> tuple:
    """
    Incarca modelul MedSigLIP din checkpoint si creeaza wrapper-ul GradCAM.

    Remapping-ul de chei e necesar pt ca in versiuni diferite ale codului
    s-au folosit nume diferite pt aceleasi module:
      - sev_head -> severity_head
      - cls_head -> classification_head
      - attn_a2b -> attn_a_to_b (cross-attention directia A->B)
    """
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt)  # suporta ambele formate: {"model": ...} sau direct state_dict

    # Remapam cheile vechi la cele noi - altfel load_state_dict ar da KeyError
    remapped = {
        k.replace("sev_head.", "severity_head.")
        .replace("cls_head.", "classification_head.")
        .replace("fusion.attn_a2b.", "fusion.attn_a_to_b.")
        .replace("fusion.attn_b2a.", "fusion.attn_b_to_a."): v
        for k, v in state.items()
    }

    # Detectam automat dimensiunea hidden a classification head-ului
    cls_hidden = 256  # default
    for key in ["classification_head.1.weight", "cls_head.1.weight"]:
        if key in remapped:
            cls_hidden = remapped[key].shape[0]  # nr de neuroni pe primul strat
            break

    model = MedSigLIPMultiTask(MODEL_PATH, n_classes=n_classes, cls_hidden=cls_hidden).to(DEVICE)
    # strict=False pt ca pot exista chei extra in checkpoint (ex: logit_scale)
    # care nu se potrivesc perfect - le ignora fara eroare
    model.load_state_dict(remapped, strict=False)
    model.eval()  # modul inferenta - dezactiveaza dropout si batch norm

    wrapper = GradCAMWrapper(model).to(DEVICE)
    return model, wrapper


def get_attention_rollout(model: MedSigLIPMultiTask, pixel_values: torch.Tensor) -> np.ndarray:
    """
    Calculeaza Attention Rollout - metoda de explicabilitate FARA gradieniti.

    Ideea: extrage matricele de atentie din TOATE straturile ViT-ului si le
    inmulteste cumulativ. Rezultatul arata cat de mult "curge" informatia
    de la fiecare patch al imaginii catre output.

    """
    # Accesam modelul HuggingFace din interiorul wrapper-ului LoRA
    base_hf_model = model.backbone.base_model.model

    with torch.no_grad():
        # output_attentions=True forteaza ViT-ul sa returneze matricele de atentie
        # pe langa embedding-urile normale
        # attentions = lista de tensori, unul per strat transformer
        # Fiecare: (batch=1, n_heads=16, seq_len=1024, seq_len=1024)
        outputs = base_hf_model.vision_model(pixel_values, output_attentions=True)
        attentions = outputs.attentions  # lista cu ~27 matrice (un strat = un element)

    seq_len = attentions[0].shape[2]  # 1024 patch-uri (32x32 grid pt imagine 448x448)

    # Pornim cu matricea identitate - fiecare token "se uita" doar la el insusi
    result = torch.eye(seq_len).to(pixel_values.device)

    for attention in attentions:
        # attention[0] = (n_heads, seq_len, seq_len) - luam primul (si singurul) element din batch
        # .mean(axis=0) = mediem peste toate attention heads => (seq_len, seq_len)
        # Fiecare head "se uita" la lucruri diferite; media le combina
        attention_heads_fused = attention[0].mean(axis=0)

        # Adaugam identitate - simuleaza conexiunea reziduala din transformer
        # In transformer: output = attention(x) + x (skip connection)
        # Fara asta, informatia originala a tokenului s-ar pierde dupa multe straturi
        attention_heads_fused += torch.eye(seq_len).to(pixel_values.device)

        # Renormalizam - fiecare rand trebuie sa sumeze la 1 (distributie de probabilitate)
        # Dupa adunarea identitatii, suma depaseste 1, trebuie corectata
        attention_heads_fused = attention_heads_fused / attention_heads_fused.sum(dim=-1, keepdim=True)

        # Inmultire matriciala cumulativa: combinam atentia din stratul curent
        # cu toate cele anterioare. Dupa N straturi, result[i][j] = cat de mult
        # "curge" informatie de la tokenul j la tokenul i prin TOATE straturile
        result = torch.matmul(attention_heads_fused, result)

    # Mediem pe prima dimensiune (toate tokenii) => vector 1D cu scorul per patch
    mask = result.mean(dim=0)

    # Reshape din vector 1D in grid 2D: sqrt(1024) = 32, deci 32x32
    grid_size = int(mask.shape[0] ** 0.5)
    mask = mask.reshape(grid_size, grid_size).cpu().numpy()

    # Min-max normalizare in [0, 1] pt vizualizare
    # 1e-8 evita impartirea la 0 daca toate valorile sunt egale
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
    return mask


def reshape_transform(tensor: torch.Tensor, height: int = 32, width: int = 32) -> torch.Tensor:
    """
    Transforma output-ul ViT din secventa 1D in grid 2D pt biblioteca GradCAM.

    Problema: ViT produce tensori de forma (batch, n_patches, embed_dim) - ex: (1, 1024, 1152)
    GradCAM se asteapta la forma CNN-like: (batch, channels, H, W) - ex: (1, 1152, 32, 32)

    Aceasta functie face conversia:
    - Detecteaza si elimina CLS token-ul daca exista (n_patches = h*w + 1)
    - Reshape: (batch, 1024, 1152) -> (batch, 32, 32, 1152)
    - Permute: (batch, 32, 32, 1152) -> (batch, 1152, 32, 32) - channels first
    """
    n_patches = tensor.shape[1]
    h = w = int(n_patches ** 0.5)  # sqrt(1024) = 32

    # Unele ViT-uri au CLS token ca primul element - il eliminam
    # Detectie: daca n_patches = 32*32 + 1 = 1025, atunci are CLS
    if n_patches == h * w + 1:
        tensor = tensor[:, 1:, :]  # eliminam primul token (CLS)
        n_patches = tensor.shape[1]
        h = w = int(n_patches ** 0.5)

    # Reshape in grid 2D si mutam channels pe pozitia 1 (format PyTorch)
    return tensor[:, :h * w, :].reshape(tensor.size(0), h, w, tensor.size(2)).permute(0, 3, 1, 2)


def auto_crop(img: Image.Image, threshold: int = 35) -> Image.Image:

    arr = np.array(img.convert("L"))  # grayscale
    mask = arr > threshold  # True = pixel cu continut, False = fundal negru

    # Verificam pe fiecare rand si coloana daca exista macar un pixel cu continut
    rows = mask.any(axis=1)  # vector boolean pt fiecare rand
    cols = mask.any(axis=0)  # vector boolean pt fiecare coloana

    if not (rows.any() and cols.any()):
        return img  # imaginea e complet neagra - returnam originalul

    # Gasim primul si ultimul rand/coloana cu continut
    y1 = max(0, int(rows.argmax()) - 5)  # primul rand cu continut - 5px padding
    y2 = min(arr.shape[0], int(len(rows) - rows[::-1].argmax()) + 5)  # ultimul rand + 5px
    x1 = max(0, int(cols.argmax()) - 5)
    x2 = min(arr.shape[1], int(len(cols) - cols[::-1].argmax()) + 5)

    # Safety check: nu crop-am daca rezultatul ar fi prea mic
    if (x2 - x1) > 50 and (y2 - y1) > 50:
        return img.crop((x1, y1, x2, y2))
    return img


def preprocess_image(path: str) -> Image.Image:
    pil = Image.open(path).convert("RGB")
    pil = pil.filter(ImageFilter.GaussianBlur(radius=0.5))
    pil = auto_crop(pil)
    pil = pil.resize((448, 448), Image.LANCZOS)
    return pil


def process_image(path, model, processor, cam_obj, method="eigencam"):

    pil = preprocess_image(path)
    rgb_resized = np.array(pil)  # (448, 448, 3)
    rgb_float = np.float32(rgb_resized) / 255.0  # (448, 448, 3) float [0,1] - necesar pt overlay

    # Procesam imaginea cu procesorul SigLIP (normalizare specifica modelului)
    input_tensor = processor(images=pil, return_tensors="pt")["pixel_values"].to(DEVICE)


    if method == "rollout":
        # Rollout returneaza un grid mic (32x32) - il redimensionam la 448x448
        grayscale_cam = get_attention_rollout(model, input_tensor)
        grayscale_cam = cv2.resize(grayscale_cam, (448, 448), interpolation=cv2.INTER_CUBIC)
    else:
        # GradCAM/EigenCAM/etc - biblioteca returneaza deja 448x448 (face resize intern)
        # targets=None => ia automat clasa cu probabilitatea cea mai mare
        # [0] => primul (si singurul) element din batch
        grayscale_cam = cam_obj(input_tensor=input_tensor, targets=None)[0]

    # Eliminam valorile negative (artefacte rare din GradCAM)
    grayscale_cam = np.maximum(grayscale_cam, 0)


    if method == "rollout":
        # Rollout produce heatmap-uri mai difuze (informatia curge prin tot graful)
        # Blur 31x31 - mai mic decat la GradCAM pt ca rollout e deja smooth
        grayscale_cam = cv2.GaussianBlur(grayscale_cam, (31, 31), 0)
        hi = grayscale_cam.max()
        if hi > 1e-8:
            grayscale_cam /= hi  # normalizare in [0, 1]

        # Ridicare la patrat - accentueaza diferentele:
        #   valori mari (0.8) -> 0.64 (raman vizibile)
        #   valori mici (0.3) -> 0.09 (dispar)
        # heatmap-ul devine mai focused, fara threshold dur
        grayscale_cam = grayscale_cam ** 2

        hi = grayscale_cam.max()
        if hi > 1e-8:
            grayscale_cam /= hi  # re-normalizare dupa squaring
    else:

        grayscale_cam = cv2.GaussianBlur(grayscale_cam, (71, 71), 0)
        hi = grayscale_cam.max()
        if hi > 1e-8:
            grayscale_cam /= hi  # normalizare in [0, 1]

        # Threshold dur la 0.35 - tot ce e sub 35% din maxim e considerat fundal
        grayscale_cam[grayscale_cam < 0.35] = 0

        # Re-normalizare dupa threshold - maximul ramane 1.0
        hi = grayscale_cam.max()
        if hi > 1e-8:
            grayscale_cam /= hi

    # Suprapunem heatmap-ul pe imgainea originala
    overlay = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)


    with torch.no_grad():  # fara gradieniti - doar inferenta
        # Extragem embedding-ul vizual (acelasi cu cel folosit la retrieval)
        image_pooled = model.encode_image(input_tensor)

        # Clasificare: embedding -> logits -> softmax -> probabilitati
        cls_logits = model.classification_head(image_pooled)
        probs = torch.softmax(cls_logits, dim=1)[0]  # [0] = scoatem din batch
        pred_class = CLASSES[probs.argmax().item()]  # clasa cu prob maxima
        confidence = probs.max().item() * 100  # probabilitatea in %

        # Severitate: embedding -> scalar [0, 1] -> procente
        # clamp(0, 1) = siguranta ca nu iese din interval
        severity_pct = model.severity_head(image_pooled).clamp(0, 1).item() * 100

    return rgb_resized, grayscale_cam, overlay, pred_class, confidence, severity_pct


def get_test_images(processor: AutoProcessor, samples_per_class: int = 2) -> list[dict]:

    ds = OCT5kDataset(
        split_csv="data/oct5k/splits_v3/test.csv",
        split_json="data/OCT5k/medgemma_prompts_split_v2_27b.json",
        severity_json="data/oct5k/severity_scores_v2.json",
        processor=processor,
        mode="eval",
    )

    # Dictionar de liste goale, cate una per clasa
    class_samples = {cls: [] for cls in ds.classes}

    # Amestecam indicii cu seed fix pt reproductibilitate
    indices = list(range(len(ds)))
    random.seed(42)
    random.shuffle(indices)

    for idx in indices:
        row = ds.df.iloc[idx]
        disease = row["disease"]

        # Daca am adunat destule pt aceasta clasa, sarim
        if len(class_samples[disease]) >= samples_per_class:
            continue

        # Verificam ca fisierul exista pe disk
        disk = ds._locate(row["image_path"])
        if disk:
            class_samples[disease].append({
                "path": disk,
                "disease": disease,
                "label": ds.lbl_map[disease],
            })

        # Daca am adunat destule pt TOATE clasele, ne oprim
        if all(len(v) >= samples_per_class for v in class_samples.values()):
            break

    # Returnam in ordine: mai intai AMD, apoi DME, DRUSEN, NORMAL
    return [sample for cls in CLASSES for sample in class_samples.get(cls, [])]


def plot_triplet(axes_row, rgb, heatmap, overlay, title_orig, title_cam, title_overlay):

    axes_row[0].imshow(rgb)
    axes_row[0].set_title(title_orig, fontsize=12)
    axes_row[0].axis("off")  # ascunde axele x/y - nu sunt relevante pt imagini

    axes_row[1].imshow(heatmap, cmap="jet")  # jet = rosu-galben-verde-albastru
    axes_row[1].set_title(title_cam, fontsize=12)
    axes_row[1].axis("off")

    axes_row[2].imshow(overlay)
    axes_row[2].set_title(title_overlay, fontsize=12)
    axes_row[2].axis("off")


def save_individual(rgb, heatmap, overlay, disease, pred, conf, sev, method, idx):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    plot_triplet(
        ax, rgb, heatmap, overlay,
        f"Original ({disease})",
        method.upper(),
        f"{pred} ({conf:.0f}%) Sev: {sev:.0f}%",
    )
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/{method}_{disease}_{idx}.png", dpi=150)
    plt.close(fig)  # inchide figura pt a elibera memorie


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-path", type=str, default=None, help="Cale imagine custom")
    parser.add_argument("--method", type=str, default="rollout", choices=list(CAM_METHODS.keys()))
    parser.add_argument("--samples", type=int, default=2, help="Imagini per clasa (2-4)")
    args = parser.parse_args()

    set_seed()

    print(f"  EXPLAINABILITY: {args.method.upper()}")

    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    model, cam_wrapper = load_model(n_classes=len(CLASSES))
    print(f"  Model incarcat pe {DEVICE}")


    cam_obj = None
    if args.method != "rollout":
        # Accesam modelul HuggingFace din interiorul wrapper-ului LoRA
        base_hf_model = model.backbone.base_model.model

        # Target layer = penultimul strat transformer din ViT (layer_norm1)
        target_layers = [base_hf_model.vision_model.encoder.layers[-2].layer_norm1]

        cam_obj = CAM_METHODS[args.method](
            model=cam_wrapper,
            target_layers=target_layers,
            reshape_transform=reshape_transform,
        )


    if args.image_path:
        rgb, heatmap, overlay, pred, conf, sev = process_image(
            args.image_path, model, processor, cam_obj, args.method
        )
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        plot_triplet(ax, rgb, heatmap, overlay, "Original", args.method.upper(),
                     f"Pred: {pred} ({conf:.0f}%) | Sev: {sev:.0f}%")
        fig.tight_layout()
        out_path = f"{OUTPUT_DIR}/{args.method}_custom.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {out_path}")


    else:
        images = get_test_images(processor, samples_per_class=args.samples)
        n_images = len(images)
        print(f"  Generam {n_images} imagini ({args.samples} per clasa)\n")

        fig, axes = plt.subplots(n_images, 3, figsize=(15, 4.5 * n_images))

        for i, info in enumerate(images):
            rgb, heatmap, overlay, pred, conf, sev = process_image(
                info["path"], model, processor, cam_obj, args.method
            )
            disease = info["disease"]  # ground truth (eticheta reala)
            correct = "good" if pred == disease else "bad"

            # Adaugam la grid-ul mare
            plot_triplet(
                axes[i], rgb, heatmap, overlay,
                f"Original ({disease})",
                args.method.upper(),
                f"{correct} {pred} ({conf:.0f}%) | Sev: {sev:.0f}%",
            )

            save_individual(rgb, heatmap, overlay, disease, pred, conf, sev, args.method, i)
            print(f"  {disease}: pred={pred} ({conf:.0f}%) sev={sev:.0f}% {correct}")

        fig.suptitle(f"MedSigLIP v14 - {args.method.upper()} Explainability", fontsize=16, y=1.01)
        fig.tight_layout()
        grid_path = f"{OUTPUT_DIR}/{args.method}_grid.png"
        fig.savefig(grid_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\n  Grid salvat: {grid_path}")


if __name__ == "__main__":
    main()