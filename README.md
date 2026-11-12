# MedSAM3-PDAC

Fully automatic segmentation of **pancreatic ductal adenocarcinoma (PDAC)** on CT with Meta's **SAM3**,
using LoRA weights fine-tuned from **[MedSAM3](https://github.com/Joey-S-Liu/MedSAM3)**.
You give it a CT scan; it returns a 3D tumour mask. No clicks or boxes are needed: every axial slice is
prompted with the text `pancreatic tumor`.

This repository contains only the fine-tuned LoRA layers (74 MB) and the code to run them. SAM3's
base weights come from Meta via Hugging Face, and MedSAM3's code is downloaded from its own repository.

## Training

Trained on 77 MSD Task07 patients with confirmed PDAC (Memorial Sloan Kettering Cancer Center). 

## Test Results
Tested on 20 held-out **[MSD](http://medicaldecathlon.com)** patients and zero-shot on 83 **[CPTAC-PDA](https://www.cancerimagingarchive.net/collection/cptac-pda/)** patients from other hospitals in the US and Poland.
Evaluated on mean per-patient 3D Dice, 95% bootstrap intervals and a confidence cut-off of 0.5, no post-processing.

| Model | Same hospital (MSD, n = 20) | New hospitals (CPTAC, n = 83) | Tumours found (CPTAC) | False spots per scan (CPTAC) |
|---|---|---|---|---|
| MedSAM3-PDAC (this repo), text only | 0.433 [0.323-0.546] | 0.368 [0.325-0.410] | 81/83 | 6.06 |
| nnU-Net 3d_fullres, same training data | 0.548 [0.438-0.650] | 0.391 [0.333-0.447] | 72/83 | 0.25 |

The model finds almost every tumour but also marks several false spots per scan, mostly in bowel and
stomach. **Review every prediction.** Dice also rises with tumour size; small tumours (< 5 mL) are
segmented much less accurately.

## Quick start

You need an **NVIDIA GPU with at least 16 GB of memory**, and access to
[facebook/sam3](https://huggingface.co/facebook/sam3): open that page while logged in to Hugging Face,
request access, and wait for approval. Then create a **Read** token under
Hugging Face → Settings → Access Tokens. Keep this token stored somewhere safely for future use.

Open a terminal (on Windows, see below) and paste these three commands:

**1. Install** (about 10 minutes; downloads PyTorch, MedSAM3's code and the dependencies)

```bash
git clone https://github.com/JacobLiu124/PDAC-MedSAM3.git #cloning repository
cd PDAC-MedSAM3
bash install.sh
```

**2. Log in to Hugging Face** (paste your Read token when asked, and answer `n` to the git question)

```bash
.venv/bin/hf auth login 
```

**3. Run** on one scan, or on a folder of scans

```bash
.venv/bin/python predict.py --input scan.nii.gz --out predictions/
```

The first run also downloads the fine-tuned weights (74 MB) from this repository's release `v0.1`,
checks them, and keeps them in `~/.cache/medsam3_pdac/`. Because the repository is private, this uses the
GitHub sign-in from step 1. If the download asks for access, set a GitHub token first
(`export GITHUB_TOKEN=...`), or download `medsam3_pdac_lora_epoch5.pt` from the release page yourself
and add `--weights medsam3_pdac_lora_epoch5.pt`.

### On Windows

Use WSL, Windows' built-in Linux, which can use your NVIDIA GPU. Once, in PowerShell as administrator:

```powershell
wsl --install -d Ubuntu-24.04
```

Restart, open **Ubuntu** from the Start menu, and run the three commands above there.
Your Windows files are under `/mnt/c/`, for example
`--input /mnt/c/Users/<you>/Desktop/scan.nii.gz`. A current NVIDIA driver installed in Windows is all
the GPU needs; do not install a separate driver inside Ubuntu.

### Tested on

Linux (Ubuntu, CUDA 12.8) with an NVIDIA A100: a fresh install reproduced the study's predictions
voxel for voxel. Windows via WSL and smaller GPUs are expected to work but have not been tested.
On a GPU with less memory, add `--batch 2`.

**Input:** CT scans in Hounsfield units as `.nii` or `.nii.gz`, one file per scan. Any orientation and
slice thickness; contrast-enhanced, portal-venous phase works best (that is what the model was trained on).

**Output:** `predictions/<name>_tumour.nii.gz` per scan (1 = tumour, 0 = background), on the same grid as
the input, plus `slice_scores.json` with the highest confidence on every slice.

| Option | Default | Meaning |
|---|---|---|
| `--cutoff` | 0.5 | Confidence cut-off. Lower finds more but adds false spots; 0.5 was best on both test sets. |
| `--weights` | `auto` | `auto` fine-tuned weights, `medsam3` (MedSAM3-v1 as released) or `none` (plain SAM3) |
| `--batch` | 8 | Slices per GPU batch. Lower it if you run out of GPU memory. |
| `--prompt` | `pancreatic tumor` | Text prompt. The model was trained on this phrase. **Changing NOT reccomended** |

## How the model was trained

- **Start:** MedSAM3-v1 LoRA weights on the frozen SAM3 base model; only the LoRA layers (rank 16,
  alpha 32, on all attention and MLP layers) were trained.
- **Data:** every slice showing the pancreas plus 10% of other slices from 77 patients (2,762 slices),
  abdominal window normalising (level 40, width 400 HU), standard radiology orientation, changed to grayscale in three channels (RGB).
- **Negative queries:** each slice was also asked for the concept it does not contain, with an empty
  answer, so the model learns to say "no tumour here".
- **Optimisation:** AdamW, learning rate 5e-5 (constant), weight decay 0.01, batch size 4, 5 epochs.
  The final (epoch 5) weights are released.

Prediction uses exactly the same slice preprocessing (`medsam3_pdac/ct_slices.py`).

## Limitations

- Research use only. Not a medical device and not validated for clinical use.
- Trained on a single hospital's scans (MSKCC) and 77 patients.
- Slices are segmented independently, without 3D context, which causes most of the false spots.
- Keeping only the largest connected region does **not** reliably fix the false spots: in our tests it
  sometimes kept a false spot and deleted the real tumour.

## Licences and credits

- **SAM3** (Meta): weights and code under Meta's SAM licence; download them yourself from Hugging Face.
- **MedSAM3** (Liu et al., 2025, [arXiv:2511.19046](https://arxiv.org/abs/2511.19046)): its code is not
  copied here; `install.sh` clones it. These weights were initialised from MedSAM3-v1. At the time of
  writing the MedSAM3 repository did not state a licence.
- **Training data:** Medical Segmentation Decathlon Task07 (Antonelli et al., 2022), with PDAC diagnoses
  from PANORAMA. **Test data:** CPTAC-PDA via the NCI Imaging Data Commons.
- No licence has been chosen for this repository's own code yet, so all rights are reserved.
