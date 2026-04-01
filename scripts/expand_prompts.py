import json
import random
from pathlib import Path

random.seed(42)


# ---------- templates ----------

TEMPLATES = {
    "scene": [
        "an OCT scan of a human retina showing {cond}",
        "a retinal OCT image displaying {cond}",
        "a medical OCT scan illustrating {cond}",
        "cross-sectional retinal OCT revealing {cond}",
        "an optical coherence tomography scan showing {cond}",
        "a high-resolution OCT image of a retina with {cond}",
    ],
    "formal": [
        "optical coherence tomography demonstrating {cond}",
        "OCT scan revealing {cond}",
        "retinal imaging showing {cond}",
        "diagnostic OCT with {cond}",
        "OCT examination displaying {cond}",
        "spectral-domain OCT demonstrating {cond}",
    ],
    "descriptive": [
        "OCT image depicting signs of {cond}",
        "retinal scan with visible {cond}",
        "OCT displaying characteristic features of {cond}",
        "imaging study showing {cond}",
        "OCT scan presenting {cond}",
        "retinal cross-section with evidence of {cond}",
    ],
    "short": [
        "OCT with {cond}",
        "retinal OCT showing {cond}",
        "{cond} visible in OCT",
        "{cond} on optical coherence tomography",
        "OCT indicating {cond}",
        "{cond} detected on OCT",
    ],
    "clinical": [
        "patient OCT scan revealing {cond}",
        "OCT findings consistent with {cond}",
        "retinal imaging confirming {cond}",
        "diagnostic imaging of {cond}",
        "OCT evidence of {cond}",
        "clinical OCT scan showing signs of {cond}",
    ],
    "anatomy": [
        "OCT showing {cond} affecting the retinal layers",
        "cross-sectional OCT with {cond} involving the macula",
        "OCT revealing {cond} at the level of the retinal pigment epithelium",
        "retinal OCT with {cond} visible across the foveal region",
        "OCT scan showing {cond} disrupting the photoreceptor layer",
        "macular OCT demonstrating {cond} in the outer retina",
    ],
    "morpho": [
        "OCT image with structural changes consistent with {cond}",
        "retinal OCT showing tissue alterations due to {cond}",
        "OCT revealing morphological features of {cond}",
        "OCT with retinal architecture disrupted by {cond}",
        "cross-sectional OCT showing layered disruption from {cond}",
        "OCT demonstrating structural damage caused by {cond}",
    ],
    "diag": [
        "OCT scan suggestive of {cond}",
        "retinal OCT with features indicative of {cond}",
        "OCT image compatible with a diagnosis of {cond}",
        "OCT findings pointing toward {cond}",
        "retinal scan with hallmarks of {cond}",
        "OCT consistent with clinical presentation of {cond}",
    ],
    "device": [
        "high-resolution spectral-domain OCT showing {cond}",
        "swept-source OCT image displaying {cond}",
        "enhanced-depth OCT revealing {cond}",
        "widefield OCT scan with {cond}",
        "3D OCT volume rendering showing {cond}",
        "high-definition OCT B-scan with {cond}",
    ],
    "edu": [
        "textbook OCT example of {cond}",
        "teaching case OCT demonstrating {cond}",
        "classic OCT presentation of {cond}",
        "representative OCT scan of {cond}",
        "illustrative OCT image showing {cond}",
        "typical OCT appearance of {cond}",
    ],
    "progress": [
        "OCT scan documenting {cond}",
        "OCT showing progression of {cond}",
        "serial OCT demonstrating {cond}",
        "follow-up OCT revealing {cond}",
        "OCT monitoring {cond} over time",
    ],
    "findings": [
        "OCT with findings of {cond}",
        "retinal OCT demonstrating findings associated with {cond}",
        "OCT scan with imaging findings of {cond}",
        "OCT revealing key findings of {cond}",
        "imaging findings on OCT consistent with {cond}",
    ],
    "assess": [
        "OCT assessment showing {cond}",
        "retinal OCT grading reveals {cond}",
        "quantitative OCT analysis demonstrating {cond}",
        "OCT thickness map indicating {cond}",
        "OCT-based evaluation of {cond}",
    ],
    "screen": [
        "screening OCT positive for {cond}",
        "referral OCT scan showing {cond}",
        "routine OCT detecting {cond}",
        "incidental finding of {cond} on OCT",
        "OCT screening revealing {cond}",
    ],
    "scan_type": [
        "OCT B-scan showing {cond}",
        "horizontal OCT section revealing {cond}",
        "vertical OCT cross-section with {cond}",
        "foveal OCT cut demonstrating {cond}",
        "perifoveal OCT scan showing {cond}",
    ],
"severity": [
        "an OCT scan indicating a {cond} case",
        "retinal imaging showing signs of {cond} progression",
        "OCT displaying a {cond} state of the macula",
        "high-resolution OCT revealing a {cond} clinical picture",
    ],
    "layered": [
        "disruption of the outer retinal layers consistent with {cond}",
        "OCT cross-section showing {cond} localized beneath the fovea",
        "intraretinal alterations on OCT diagnostic for {cond}",
        "subretinal architecture changes indicating {cond}",
    ],
}


# ---------- condition variations per class ----------

CONDITIONS = {
    "AMD": [
        "age-related macular degeneration", "macular degeneration",
        "AMD pathology", "AMD-related retinal degeneration",
        "degenerative macular disease", "senile macular degeneration",
        "age-related maculopathy", "atrophic macular changes",
        "AMD-associated retinal thinning", "progressive macular degeneration",
        "geographic atrophy from AMD", "macular atrophy due to aging",
        "age-related retinal pigment changes", "AMD with RPE degeneration",
        "maculopathy associated with aging",
    ],
    "DME": [
        "diabetic macular edema", "diabetic retinal edema",
        "macular swelling due to diabetes", "diabetic macular thickening",
        "DME pathology", "cystoid macular edema from diabetes",
        "diabetic intraretinal fluid", "diabetic foveal edema",
        "macular edema secondary to diabetes mellitus",
        "diabetes-related retinal fluid accumulation",
        "center-involving diabetic macular edema", "non-center-involving DME",
        "chronic diabetic macular edema", "tractional diabetic macular edema",
        "diabetic macular cysts",
    ],
    "DRUSEN": [
        "drusen deposits", "retinal drusen accumulation",
        "macular drusen deposits", "sub-retinal pigment epithelium deposits",
        "drusen pathology", "lipid-rich subretinal deposits",
        "basal laminar drusen", "extracellular debris under the RPE",
        "yellow-white retinal deposits",
        "accumulation of waste material beneath the retina",
        "cuticular drusen", "calcified drusen", "reticular pseudodrusen",
        "subretinal drusenoid deposits", "drusen with pigmentary changes",
    ],
    "NORMAL": [
        "normal retinal anatomy", "healthy retina", "no retinal pathology",
        "normal macular structure", "healthy retinal tissue",
        "intact retinal layers", "normal foveal contour",
        "unremarkable retinal architecture", "physiologically normal macula",
        "well-preserved retinal structure", "no evidence of macular disease",
        "healthy foveal depression", "normal inner and outer retinal layers",
        "intact photoreceptor ellipsoid zone", "normal retinal thickness profile",
    ],
}


# ---------- modifiers ----------

MODIFIERS = {
    "AMD":    ["", "early ", "intermediate ", "advanced ", "dry ", "wet ", "severe ", "late-stage ", "bilateral ", "unilateral ", "progressive ", "stable "],
    "DME":    ["", "mild ", "moderate ", "severe ", "diffuse ", "focal ", "cystoid ", "refractory ", "center-involving ", "non-center-involving ", "chronic ", "treatment-resistant "],
    "DRUSEN": ["", "soft ", "hard ", "confluent ", "small ", "large ", "scattered ", "dense ", "calcified ", "numerous ", "bilateral ", "macular "],
    "NORMAL": ["", "completely ", "perfectly ", "textbook ", "entirely ", "demonstrably ", "clearly "],
}


# ---------- negative / comparative templates ----------

NEG_TEMPLATES = [
    "OCT scan showing {cond}, not {neg}",
    "retinal OCT with {cond} and no signs of {neg}",
    "OCT image of {cond} without evidence of {neg}",
    "OCT demonstrating {cond}, ruling out {neg}",
    "{cond} on OCT, with absence of {neg}",
    "OCT confirming {cond} and excluding {neg}",
    "retinal imaging positive for {cond}, negative for {neg}",
    "OCT scan consistent with {cond} but not {neg}",
]

COMP_TEMPLATES = [
    "OCT showing {cond}, which differs from {other} by {feat}",
    "retinal OCT consistent with {cond} rather than {other}",
    "OCT distinguishing {cond} from {other}",
    "OCT image favoring {cond} over {other} based on retinal features",
    "{cond} on OCT, differentiated from {other}",
    "OCT findings supporting {cond} instead of {other} due to {feat}",
    "retinal scan showing {cond}, distinguished from {other} by the presence of {feat}",
    "OCT evidence of {cond} with {feat}, unlike {other}",
]

ALL_CLASSES = list(CONDITIONS.keys())

DIFF_FEATURES = {
    # --- AMD (Age-related Macular Degeneration) ---
    ("AMD", "DRUSEN"): [
        "geographic atrophy", "RPE disruption", "outer retinal thinning",
        "photoreceptor loss", "subretinal tubulations", "fibrotic scars",
        "choroidal neovascularization", "subretinal hyperreflective material"
    ],
    ("AMD", "NORMAL"): [
        "RPE irregularity", "retinal thinning", "pigmentary changes",
        "loss of outer retinal bands", "macular atrophy", "drastic RPE elevation"
    ],
    ("AMD", "DME"): [
        "dry macular degeneration", "retinal pigment epithelium atrophy",
        "lack of intraretinal cysts", "outer retinal layers collapse", "no diffuse swelling"
    ],

    ("DRUSEN", "AMD"): [
        "isolated deposits without atrophy", "preserved RPE", "no geographic atrophy",
        "intact outer retina", "nodular sub-RPE deposits", "confluent drusen without scarring"
    ],
    ("DRUSEN", "NORMAL"): [
        "subretinal deposits", "RPE undulations", "hyperreflective bumps",
        "irregular RPE profile", "focal RPE elevations", "Bruch's membrane thickening"
    ],
    ("DRUSEN", "DME"): [
        "sub-RPE material", "lack of retinal edema", "no fluid pockets",
        "dry retinal texture", "no cystoid spaces", "localized RPE humps"
    ],

    ("DME", "NORMAL"): [
        "intraretinal fluid", "cystoid macular edema", "increased retinal thickness",
        "hyporeflective spaces", "neurosensory detachment", "diffuse retinal swelling"
    ],
    ("DME", "AMD"): [
        "intraretinal cystic spaces", "spongiform retinal thickening",
        "no sub-RPE deposits", "fluid in inner nuclear layer", "massive macular edema"
    ],
    ("DME", "DRUSEN"): [
        "diffuse intraretinal fluid", "large cystoid spaces", "lack of focal drusen",
        "increased foveal thickness", "presence of hard exudates"
    ],

    ("NORMAL", "DRUSEN"): [
        "smooth RPE", "no deposits", "uniform retinal layers",
        "clean sub-RPE space", "flat Bruch's membrane", "continuous ELM line"
    ],
    ("NORMAL", "AMD"): [
        "preserved macular thickness", "intact photoreceptor layer",
        "no degeneration", "normal RPE", "continuous RPE-choriocapillaris complex"
    ],
    ("NORMAL", "DME"): [
        "no macular thickening", "no intraretinal fluid",
        "normal retinal thickness", "no cystoid changes", "foveal pit depression preserved"
    ],
}

# ---------- generators ----------

def gen_positive(cls_name, count=100):
    seen = set()
    pool = []

    variations = CONDITIONS[cls_name]
    mods = MODIFIERS[cls_name]

    for group in TEMPLATES.values():
        for tpl in group:
            for var in variations:
                for mod in mods:
                    cond = f"{mod}{var}".strip()
                    prompt = tpl.format(cond=cond)
                    if prompt not in seen:
                        seen.add(prompt)
                        pool.append(prompt)

    random.shuffle(pool)
    print(f"    Pool: {len(pool)} candidates -> selecting {count}")
    return pool[:count]


def gen_negative(cls_name, count=25):
    results = []
    seen = set()
    positives = CONDITIONS[cls_name]
    others = [c for c in ALL_CLASSES if c != cls_name]

    attempts = count * 8
    for _ in range(attempts):
        tpl = random.choice(NEG_TEMPLATES)
        pos = random.choice(positives)
        neg_cls = random.choice(others)
        neg_term = random.choice(CONDITIONS[neg_cls])

        prompt = tpl.format(cond=pos, neg=neg_term)
        if prompt not in seen:
            seen.add(prompt)
            results.append(prompt)
        if len(results) >= count:
            break

    return results[:count]


def gen_comparative(cls_name, count=25):
    results = []
    seen = set()
    positives = CONDITIONS[cls_name]

    pairs = {k: v for k, v in DIFF_FEATURES.items() if k[0] == cls_name}
    if not pairs:
        return results

    attempts = count * 8
    for _ in range(attempts):
        pair = random.choice(list(pairs.keys()))
        feats = pairs[pair]

        tpl = random.choice(COMP_TEMPLATES)
        pos = random.choice(positives)
        other = random.choice(CONDITIONS[pair[1]])
        feat = random.choice(feats)

        prompt = tpl.format(cond=pos, other=other, feat=feat)
        if prompt not in seen:
            seen.add(prompt)
            results.append(prompt)
        if len(results) >= count:
            break

    return results[:count]


# ---------- main generation ----------

def generate_all(target=150):
    n_pos = int(target * 0.67)
    n_neg = int(target * 0.165)
    n_comp = target - n_pos - n_neg

    output = {}

    for cls_name in ALL_CLASSES:
        print(f"\n{'=' * 60}\n  {cls_name}\n{'=' * 60}")

        pos = gen_positive(cls_name, n_pos)
        neg = gen_negative(cls_name, n_neg)
        comp = gen_comparative(cls_name, n_comp)

        gap = target - (len(pos) + len(neg) + len(comp))
        if gap > 0:
            extra = gen_positive(cls_name, n_pos + gap + 50)
            already = set(pos)
            filler = [p for p in extra if p not in already]
            pos.extend(filler[:gap])

        combined = pos + neg + comp
        seen = set()
        unique = []
        for p in combined:
            if p not in seen:
                seen.add(p)
                unique.append(p)
        combined = unique[:target]

        output[cls_name] = {
            "positive": pos,
            "negative": neg,
            "comparative": comp,
            "all": combined,
        }

        print(f"  Pos: {len(pos)} | Neg: {len(neg)} | Comp: {len(comp)} | Total: {len(combined)}")

    return output


# ---------- saving ----------

def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved -> {path}")


def save_structured(prompts, path):
    save_json(prompts, path)


def save_flat_positive(prompts, path):
    flat = {cls: d["positive"] for cls, d in prompts.items()}
    save_json(flat, path)


def save_flat_all(prompts, path):
    flat = {cls: d["all"] for cls, d in prompts.items()}
    save_json(flat, path)


# ---------- stats ----------

def show_stats(prompts):
    print(f"\n{'=' * 60}\n  STATISTICS\n{'=' * 60}")

    total = 0
    for cls, d in prompts.items():
        n = len(d["all"])
        total += n
        print(
            f"  {cls:8s} -> {n:3d}  "
            f"(pos:{len(d['positive'])}, neg:{len(d['negative'])}, comp:{len(d['comparative'])})"
        )

    print(f"  {'TOTAL':8s} -> {total}")

    n_templates = sum(len(v) for v in TEMPLATES.values())
    print(f"  Templates: {n_templates}")

    pool = sum(
        n_templates * len(CONDITIONS[c]) * len(MODIFIERS[c])
        for c in ALL_CLASSES
    )
    print(f"  Candidate pool: {pool:,} unique combinations")


# ---------- entry ----------

if __name__ == "__main__":
    TARGET = 3000

    out_dir = Path("data/old")
    out_dir.mkdir(exist_ok=True)

    prompts = generate_all(target=TARGET)

    save_structured(prompts, out_dir / "prompts_expanded_structured.json")
    save_flat_positive(prompts, out_dir / "prompts_expanded.json")
    save_flat_all(prompts, out_dir / "prompts_expanded_all.json")

    show_stats(prompts)