"""How a CT volume becomes 2D pictures for SAM3. Identical to the preprocessing used in training,
so the model sees new scans drawn exactly like its training slices.

- Orientation: every volume is turned to the standard radiology view (LPS: patient's right on the
  image's left, front at the top), whatever hospital or file format it came from.
- Intensity: abdominal window, level 40 HU / width 400 HU, mapped to 0-255.
- Channels: the slice is repeated in all three colour channels.
"""
import numpy as np
import SimpleITK as sitk

LEVEL, WIDTH = 40, 400


def to_standard_view(img: sitk.Image) -> sitk.Image:
    return sitk.DICOMOrient(img, "LPS")


def window(hu: np.ndarray) -> np.ndarray:
    lo = LEVEL - WIDTH / 2
    return (np.clip((hu.astype(np.float32) - lo) / WIDTH, 0, 1) * 255).astype(np.uint8)


def slice_rgb(vol_u8: np.ndarray, z: int) -> np.ndarray:
    """vol_u8: windowed volume as (z, y, x). Returns (y, x, 3) uint8."""
    s = vol_u8[z]
    return np.stack([s, s, s], -1)


def load_volume(path):
    """Read a CT (values in HU), return (original image, standard-view image, windowed uint8 array (z, y, x))."""
    orig = sitk.ReadImage(str(path))
    img = to_standard_view(orig)
    return orig, img, window(sitk.GetArrayFromImage(img))
