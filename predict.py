"""Segment pancreatic ductal adenocarcinoma (PDAC) on CT with SAM3 + the MedSAM3-PDAC LoRA weights.

Fully automatic: every axial slice gets the text prompt "pancreatic tumor", detections above the
confidence cut-off are merged into the slice mask, and the slices are stacked back into a 3D mask
on the input scan's own grid (same size, spacing and orientation as the input).

  python predict.py --input scan.nii.gz --out predictions/
  python predict.py --input folder_of_scans/ --out predictions/ --cutoff 0.5

Input: CT volumes in Hounsfield units, as .nii or .nii.gz (one file per scan).
Output: <name>_tumour.nii.gz per scan (1 = tumour, 0 = background), plus slice_scores.json.
Needs an NVIDIA GPU (>= 16 GB), access to facebook/sam3 on Hugging Face, and MedSAM3's code (install.sh).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from medsam3_pdac.ct_slices import load_volume, slice_rgb  # noqa: E402
from medsam3_pdac.weights import get_weights  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--input", required=True, type=Path, help="a CT .nii/.nii.gz file, or a folder of them")
ap.add_argument("--out", required=True, type=Path, help="output folder")
ap.add_argument("--weights", default="auto",
                help="'auto' (download this repo's release, default), a local .pt file, "
                     "'medsam3' (MedSAM3-v1 as released, no PDAC training) or 'none' (plain SAM3)")
ap.add_argument("--cutoff", type=float, default=0.5, help="confidence cut-off (0.5 was best in our tests)")
ap.add_argument("--prompt", default="pancreatic tumor", help="text prompt (the model was trained on this one)")
ap.add_argument("--batch", type=int, default=8, help="slices per GPU batch; lower it if you run out of memory")
ap.add_argument("--nms-iou", type=float, default=0.5)
ap.add_argument("--medsam3-dir", type=Path, default=Path(os.environ.get("MEDSAM3_DIR", HERE / "MedSAM3")))
a = ap.parse_args()

a.input, a.out = a.input.resolve(), a.out.resolve()
if not a.input.exists():
    raise SystemExit(f"input not found: {a.input}")
cases = sorted(p for p in ([a.input] if a.input.is_file() else a.input.iterdir())
               if p.name.endswith((".nii.gz", ".nii")))
if not cases:
    raise SystemExit(f"no .nii/.nii.gz files found in {a.input}")
if not (a.medsam3_dir / "sam3").is_dir():
    raise SystemExit(f"MedSAM3 code not found at {a.medsam3_dir}; run install.sh or pass --medsam3-dir")
weights = {"auto": None, "medsam3": "medsam3", "none": "none"}.get(a.weights, a.weights)
if weights is None:
    weights = str(get_weights())
a.out.mkdir(parents=True, exist_ok=True)

os.chdir(a.medsam3_dir)            # SAM3's code loads some assets by relative path
sys.path.insert(0, str(a.medsam3_dir))
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image as PILImage  # noqa: E402
from torchvision.ops import nms  # noqa: E402
from sam3.model_builder import build_sam3_image_model  # noqa: E402
from sam3.train.data.sam3_image_dataset import Datapoint, Image as SAMImage, FindQueryLoaded, InferenceMetadata  # noqa: E402
from sam3.train.data.collator import collate_fn_api  # noqa: E402
from sam3.model.utils.misc import copy_data_to_device  # noqa: E402
from sam3.train.transforms.basic_for_api import ComposeAPI, RandomResizeAPI, ToTensorAPI, NormalizeAPI  # noqa: E402
from lora_layers import LoRAConfig, apply_lora_to_model  # noqa: E402

if not torch.cuda.is_available():
    raise SystemExit("SAM3 needs an NVIDIA GPU (CUDA); none was found.")
device = torch.device("cuda")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
model = build_sam3_image_model(device="cuda", compile=False, load_from_HF=True,
                               bpe_path="sam3/assets/bpe_simple_vocab_16e6.txt.gz", eval_mode=True)
if weights != "none":
    lc = yaml.safe_load(open(HERE / "configs" / "lora.yaml"))["lora"]
    model = apply_lora_to_model(model, LoRAConfig(
        rank=lc["rank"], alpha=lc["alpha"], dropout=0.0, target_modules=lc["target_modules"],
        **{k: v for k, v in lc.items() if k.startswith("apply_to_")}))
    if weights == "medsam3":
        from huggingface_hub import hf_hub_download
        weights = hf_hub_download("lal-Joey/MedSAM3_v1", "best_lora_weights.pt")
    sd = torch.load(weights, map_location="cpu")
    own = model.state_dict()
    bad = [k for k in sd if k not in own or own[k].shape != sd[k].shape]
    if bad:
        raise SystemExit(f"{len(bad)} LoRA tensors don't fit the model, e.g. {bad[:3]}")
    model.load_state_dict(sd, strict=False)
    print(f"loaded {len(sd)} LoRA tensors from {weights}")
model.to(device).eval()
transform = ComposeAPI(transforms=[
    RandomResizeAPI(sizes=1008, max_size=1008, square=True, consistent_transform=False),
    ToTensorAPI(), NormalizeAPI(mean=[0.5] * 3, std=[0.5] * 3)])


def datapoint(rgb):
    h, w = rgb.shape[:2]
    q = FindQueryLoaded(query_text=a.prompt, image_id=0, object_ids_output=[], is_exhaustive=True,
                        query_processing_order=0,
                        inference_metadata=InferenceMetadata(coco_image_id=0, original_image_id=0,
                                                             original_category_id=1, original_size=[w, h],
                                                             object_id=0, frame_index=0))
    return transform(Datapoint(find_queries=[q], images=[SAMImage(data=PILImage.fromarray(rgb), objects=[], size=[h, w])]))


@torch.no_grad()
def predict_slices(rgbs):
    """Per slice: (h, w) bool tumour mask at the cut-off, and the slice's top confidence."""
    h, w = rgbs[0].shape[:2]
    batch = collate_fn_api([datapoint(r) for r in rgbs], dict_key="input")["input"]
    out = model(copy_data_to_device(batch, device, non_blocking=True))[-1]
    logits, boxes, masks = out["pred_logits"], out["pred_boxes"], out["pred_masks"]
    results = []
    for i in range(len(rgbs)):
        scores = logits[i].sigmoid().max(-1)[0]
        keep = scores > a.cutoff
        m = np.zeros((h, w), bool)
        if keep.any():
            cx, cy, bw, bh = boxes[i, keep].unbind(-1)
            xyxy = torch.stack([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h], -1)
            k2 = nms(xyxy.float(), scores[keep].float(), a.nms_iou)
            small = masks[i, keep][k2].sigmoid() > 0.5
            big = F.interpolate(small.unsqueeze(0).float(), size=(h, w), mode="bilinear", align_corners=False)[0] > 0.5
            m = big.any(0).cpu().numpy()
        results.append((m, float(scores.max())))
    return results


log_path = a.out / "slice_scores.json"
log = json.loads(log_path.read_text()) if log_path.exists() else {}
t0 = time.time()
for n, path in enumerate(cases, 1):
    name = path.name.removesuffix(".gz").removesuffix(".nii")
    orig, img, vol = load_volume(path)
    mask = np.zeros(vol.shape, np.uint8)
    top = []
    for i in range(0, len(vol), a.batch):
        zs = list(range(i, min(i + a.batch, len(vol))))
        for z, (m, s) in zip(zs, predict_slices([slice_rgb(vol, z) for z in zs])):
            mask[z] = m
            top.append(s)
    out = sitk.GetImageFromArray(mask)
    out.CopyInformation(img)                                    # standard-view grid
    out = sitk.Resample(out, orig, sitk.Transform(), sitk.sitkNearestNeighbor, 0)   # back to the input's grid
    sitk.WriteImage(out, str(a.out / f"{name}_tumour.nii.gz"))
    log[name] = top
    log_path.write_text(json.dumps(log))
    ml = mask.sum() * np.prod(img.GetSpacing()) / 1000
    print(f"[{n}/{len(cases)}] {name}: {len(vol)} slices, tumour {ml:.1f} mL, top score {max(top):.2f} "
          f"({(time.time() - t0) / n:.0f} s/scan)", flush=True)
print("done:", a.out)
