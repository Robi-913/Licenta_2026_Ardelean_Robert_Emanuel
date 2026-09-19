# MedSigLIP-OCT: Domain Adaptation of Vision-Language Models for Retinal OCT

Code for the paper **"Domain Adaptation of Vision-Language Models for Retinal OCT Analysis via Multi-Task Latent Alignment and Cross-Attention Fusion"** (ICCP 2026), and for the bachelor's thesis it comes from (Technical University of Cluj-Napoca).

The project adapts [MedSigLIP](https://huggingface.co/google/medsiglip-448) to retinal OCT with LoRA. Long clinical reports (180-256 tokens) do not fit the 64-token limit of the SigLIP text encoder, so each report is split into two sub-prompts (structural and pathological). Both are encoded separately and merged by a Cross-Attention Fusion module. The model is trained on several tasks at once: image-text retrieval, disease classification, and severity regression. Nine biomarker heads are trained separately on top of the frozen backbone.

> **Research prototype.** The generated reports and the severity score are not clinically validated. Do not use this code or the model for patient diagnosis.

- **Model weights:** https://huggingface.co/robi913/medsiglip-retinal-oct-lora
- **Base model (gated, accept its terms first):** https://huggingface.co/google/medsiglip-448

## Results (OCT5k test set, 748 images)

| | Accuracy | F1 macro | Avg. R@1 | Severity MAE |
| :--- | :---: | :---: | :---: | :---: |
| ResNet18 (from scratch) | 66.8% | 0.724 | - | - |
| MedSigLIP zero-shot | 25.8% | 0.107 | 41.9% | - |
| MIRAGE backbone | 43.2% | - | 48.0% | 39.1 |
| **MedSigLIP v15 (this work)** | **83.8%** | **0.837** | **84.8%** | **23.3** |

Recall@K is measured at the diagnosis level: a query is a hit if one of the top-K results has the same diagnosis. It is not exact image-report matching. Severity MAE is in points on the 0-100 severity index. All results come from a single training run. See the paper for details and limitations.

## Repository layout

```
scripts/                      dataset metadata, CSV files and patient-level splits
src/
  datasets/                   OCT5k datasets (CNN, MedSigLIP)
  model/medsiglip.py          MedSigLIPBase, CrossAttentionFusion, MedSigLIPMultiTask, BiomarkerHeadsV5
  model/cnn_resnet18.py       ResNet18 baseline
  losses/siglip_loss.py       SigLIP sigmoid loss
  pipelines/
    yolo/                     YOLO bounding boxes (silver labels)
    medgemma/                 report generation and severity scoring
    gemini/                   splitting reports into prompt_a / prompt_b
    medsiglip/                training, classification probing, biomarker heads, MIRAGE variant
    qwen/                     alternative Qwen-based scripts (not used for the reported results)
  training/                   ResNet18 baseline and earlier SigLIP experiments
  evaluation/                 test-set evaluation, zero-shot, t-SNE, YOLO vs MedSigLIP
  retrieval/                  retrieval analysis and demo
  explainability/             EigenCAM
  uncertainty/                Monte Carlo Dropout
  demo/gradio_app.py          Gradio demo
experiments/                  metrics, JSON results and figures produced by the runs
wandb/                        logs of the training runs
```

## Setup

```bash
git clone https://github.com/Robi-913/Licenta_2026_Ardelean_Robert_Emanuel
cd Licenta_2026_Ardelean_Robert_Emanuel
pip install -r requirements.txt
huggingface-cli login      # needed for google/medsiglip-448 and medgemma
```

`requirements.txt` is a full environment export from a Windows machine. If a package fails to install on your system, install the main ones (`torch`, `transformers`, `peft`, `accelerate`, `bitsandbytes`, `scikit-learn`, `pandas`, `matplotlib`, `seaborn`, `grad-cam`, `gradio`, `wandb`, `tqdm`) and add the rest as needed.

**Data.** The OCT5k dataset is not included (see Arikan et al., *Scientific Data* 12, 267, 2025). Download it and place it under `data/OCT5k/`. Generated reports, severity scores and the patient-level splits (`data/oct5k/splits_v3`) are created by the scripts below.

**Paths and settings.** Every script has a `Config` class at the top with its paths and hyperparameters. Run the scripts from the repository root and edit `Config` to match your folders.

## Reproducing the pipeline

Run from the repository root, for example `python src/pipelines/medsiglip/train_medsiglip.py`.

1. **Metadata and splits:** `scripts/build_metadata.py`, `scripts/create_csv_from_folders.py`, `scripts/make_splits_biomk.py`
2. **YOLO silver labels:** `src/pipelines/yolo/generate_bbox.py`
3. **Reports and severity:**
   - `src/pipelines/medgemma/generate_prompts_medgemma.py` generates the 180-256 token reports (MedGemma 27B, 4-bit).
   - `src/pipelines/gemini/split_prompts_gemini.py` splits them into `prompt_a` and `prompt_b`. It needs a Gemini API key, which you set in the `Config` class. Do not commit it.
   - `src/pipelines/medgemma/severity_medgemma.py` computes the severity index.
4. **Training:**
   - `src/pipelines/medsiglip/train_medsiglip.py` runs the multi-task LoRA training.
   - `src/pipelines/medsiglip/linear_probing_cls.py` trains the classification head on the frozen embeddings (the reported accuracy is measured after this step).
   - `src/pipelines/medsiglip/train_biomarker.py` trains the 9 biomarker heads.
5. **Evaluation:** `src/evaluation/evaluate.py`, `src/retrieval/retrieval_analysis.py`, `src/evaluation/zero_shot.py`, `src/evaluation/tsne_viz.py`, `src/uncertainty/mc_dropout.py`, `src/explainability/gradcam.py`
6. **Baselines:** `src/training/train_cnn.py` (ResNet18), `src/pipelines/medsiglip/train_medsiglip_mirage.py` (MIRAGE backbone)

## Using the trained model

The checkpoints are on Hugging Face and load into `MedSigLIPMultiTask` and `BiomarkerHeadsV5` from `src/model/medsiglip.py`. A complete loading example, with classification, severity, image-text similarity and biomarker detection, is in the [model card](https://huggingface.co/robi913/medsiglip-retinal-oct-lora).

To try the model interactively, edit the checkpoint path at the top of `src/demo/gradio_app.py` and run `python src/demo/gradio_app.py`.

## Severity index

The severity score is built from a base value per disease, a weight per biomarker instance, the lesion area and a log-scaled lesion count. It was made with medical input, but it is not a clinically validated scale. Lesion counts come from physician annotations where available, otherwise from YOLO detections verified and corrected by MedGemma. The code is in `src/pipelines/medgemma/severity_medgemma.py`.

## License and terms

- **Code:** TODO: choose a license (for example MIT or Apache-2.0) and add a `LICENSE` file.
- **Models:** MedSigLIP and MedGemma are covered by the [Health AI Developer Foundations terms](https://developers.google.com/health-ai-developer-foundations/terms). The released checkpoints contain the MedSigLIP base weights, so the same terms apply to them.
- **Data:** follow the license and terms of OCT5k.

## Citation

If you use this work, please cite the paper (full reference to be added after publication):

```
R.-E. Ardelean and A. Marginean,
"Domain Adaptation of Vision-Language Models for Retinal OCT Analysis
via Multi-Task Latent Alignment and Cross-Attention Fusion,"
ICCP 2026.
```

## Acknowledgements

This work was supported in part by the project "Romanian Hub for Artificial Intelligence - HRIA", Smart Growth, Digitization and Financial Instruments Program, MySMIS no. xxxxx. Thanks to the creators of OCT5k and to Google for MedSigLIP and MedGemma.
