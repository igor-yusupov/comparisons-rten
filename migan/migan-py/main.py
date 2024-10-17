import os

import numpy as np
import onnxruntime
from PIL import Image

WEIGHT_DIR = "../weights"
IMG_PATH = "../data/img.png"
MASK_PATH = "../data/mask.png"


def read_mask(mask_path, invert=True):
    mask = Image.open(mask_path)
    mask = np.array(mask)
    if len(mask.shape) == 3:
        if mask.shape[2] == 4:
            _r, _g, _b, _a = np.rollaxis(mask, axis=-1)
            mask = np.dstack([_a, _a, _a])
        elif mask.shape[2] == 2:
            _l, _a = np.rollaxis(mask, axis=-1)
            mask = np.dstack([_a, _a, _a])
        elif mask.shape[2] == 3:
            _r, _g, _b = np.rollaxis(mask, axis=-1)
            mask = np.dstack([_r, _r, _r])
    else:
        mask = np.dstack([mask, mask, mask])
    if invert:
        mask = 255 - mask
    mask[mask < 255] = 0

    return Image.fromarray(mask).convert("L")


def get_inputs(img_path: str, mask_path: str):
    img = Image.open(img_path).convert("RGB")
    mask = read_mask(mask_path)
    img = np.expand_dims(img, 0).transpose(0, 3, 1, 2)
    mask = np.expand_dims(mask, (0, 1))
    return img, mask


def main() -> None:
    model_path = os.path.join(WEIGHT_DIR, "migan.onnx")
    model = onnxruntime.InferenceSession(
        model_path, providers=["CPUExecutionProvider"]
    )

    img, mask = get_inputs(IMG_PATH, MASK_PATH)
    output = model.run(None, {'image': img, 'mask':  mask})

    result_image = output[0][0].transpose(1, 2, 0)
    result_image = Image.fromarray(result_image)
    result_image.save("../data/result.png")


if __name__ == "__main__":
    main()
