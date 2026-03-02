import json
from pathlib import Path

prompts = {
    "AMD": [
        "an OCT scan showing age-related macular degeneration",
        "retinal OCT with AMD pathology",
        "macular degeneration visible in optical coherence tomography",
        "OCT image depicting AMD lesions"
    ],
    "CNV": [
        "an OCT scan showing choroidal neovascularization",
        "retinal OCT with CNV pathology",
        "choroidal neovascularization in optical coherence tomography",
        "OCT image with neovascular membrane"
    ],
    "CSR": [
        "an OCT scan showing central serous retinopathy",
        "retinal OCT with CSR pathology",
        "central serous retinopathy in optical coherence tomography",
        "OCT image depicting serous detachment"
    ],
    "DME": [
        "an OCT scan showing diabetic macular edema",
        "retinal OCT with DME pathology",
        "diabetic macular edema in optical coherence tomography",
        "OCT image with macular swelling from diabetes"
    ],
    "DR": [
        "an OCT scan showing diabetic retinopathy",
        "retinal OCT with diabetic retinopathy pathology",
        "diabetic retinopathy visible in optical coherence tomography",
        "OCT image depicting diabetic retinal changes"
    ],
    "DRUSEN": [
        "an OCT scan showing drusen deposits",
        "retinal OCT with drusen pathology",
        "drusen visible in optical coherence tomography",
        "OCT image with subretinal deposits"
    ],
    "MH": [
        "an OCT scan showing macular hole",
        "retinal OCT with macular hole pathology",
        "macular hole visible in optical coherence tomography",
        "OCT image depicting full-thickness macular defect"
    ],
    "NORMAL": [
        "a normal OCT scan of healthy retina",
        "healthy retinal OCT without pathology",
        "normal optical coherence tomography scan",
        "OCT image showing healthy retinal layers"
    ]
}

output_path = Path("data/prompts.json")
with open(output_path, "w") as f:
    json.dump(prompts, f, indent=2)

print(f"prompts.json creat cu {len(prompts)} clase")
for label, prompt_list in prompts.items():
    print(f"  {label}: {len(prompt_list)} prompturi")