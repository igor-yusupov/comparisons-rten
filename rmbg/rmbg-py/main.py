import argparse
import os
import time
import typing as t
from enum import Enum

import cv2
import numpy as np
import onnxruntime
from PIL import Image

IMAGE_PATH = "../data/image.jpeg"
WEIGHT_DIR = "../weights"


class WeightsType(Enum):
    Default = "fp32"
    Fp16 = "fp16"
    Quant = "int8"


def get_image() -> np.ndarray:
    img = cv2.imread(IMAGE_PATH)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, (1024, 1024))
    img = img / 255.0

    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([1.0, 1.0, 1.0])
    image_data = (img - mean) / std

    image_data = np.transpose(image_data, (2, 0, 1))
    return np.expand_dims((image_data.astype(np.float32)), axis=0)


def postprocess_image(result: np.ndarray, orig_size: t.Tuple[int, int]) -> np.ndarray:
    result = np.transpose(result, (1, 2, 0))
    result = cv2.resize(result, orig_size)
    ma = np.max(result)
    mi = np.min(result)
    result = (result - mi) / (ma - mi)
    im_array = (result * 255).astype(np.uint8)
    im_array = np.squeeze(im_array)
    return im_array


def main(weights_type: WeightsType) -> None:
    image = get_image()

    match weights_type:
        case WeightsType.Default:
            model_path = os.path.join(WEIGHT_DIR, "model.onnx")
        case WeightsType.Fp16:
            model_path = os.path.join(WEIGHT_DIR, "model_fp16.onnx")
        case WeightsType.Quant:
            model_path = os.path.join(WEIGHT_DIR, "model_quantized.onnx")
        case _:
            model_path = os.path.join(WEIGHT_DIR, "model.onnx")

    model = onnxruntime.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    start = time.time()
    output = model.run(None, {"input": image})
    print(f"Inference time: {(time.time() - start) * 1000} ms")

    orig_image = Image.open(IMAGE_PATH)
    result_image = postprocess_image(output[0][0], orig_image.size)

    pil_im = Image.fromarray(result_image)
    no_bg_image = Image.new("RGBA", pil_im.size, (0, 0, 0, 0))
    no_bg_image.paste(orig_image, mask=pil_im)
    no_bg_image.save("../data/example_image_no_bg.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights_type",
        default="fp32",
        type=WeightsType,
        help="Type of weights",
    )
    args = parser.parse_args()
    main(args.weights_type)
