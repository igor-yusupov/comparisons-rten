import argparse
import os
import time
from enum import Enum

import cv2
import numpy as np
import onnxruntime

IMAGE_PATH = "../data/image.jpeg"
WEIGHT_DIR = "../weights"


class ModelType(Enum):
    ResNet18 = "resnet18"
    MobileNet = "mobilenet"
    EfficientNet = "efficientnet"


def get_image() -> np.ndarray:
    img = cv2.imread(IMAGE_PATH)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, (224, 224))
    img = img / 255.0

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_data = (img - mean) / std

    image_data = np.transpose(image_data, (2, 0, 1))
    return np.expand_dims((image_data.astype(np.float32)), axis=0)


def main(model_type: ModelType, is_quant: bool) -> None:
    model_path = (
        os.path.join(WEIGHT_DIR, f"{model_type.value}_quant.onnx")
        if is_quant
        else os.path.join(WEIGHT_DIR, f"{model_type.value}.onnx")
    )
    model = onnxruntime.InferenceSession(
        model_path, providers=["CPUExecutionProvider"]
    )

    image = get_image()

    start = time.time()
    output = model.run(None, {"input": image})
    print(f"Inference time: {time.time() - start}")
    label = np.argmax(output[0])

    print(f"Got label: {label}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quant", action="store_true", help="Use quant models"
    )
    parser.add_argument(
        "--model_type",
        default="resnet18",
        type=ModelType,
        help="Version of whisper model",
    )

    args = parser.parse_args()
    main(args.model_type, args.quant)
