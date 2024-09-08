import argparse
import os

import torch
import torchvision.models as models

from utils import ModelType

WEIGHTS_DIR = "./weights"


def main(model_type: ModelType) -> None:
    match model_type:
        case ModelType.ResNet18:
            model = models.resnet18(pretrained=True)
        case ModelType.MobileNet:
            model = models.mobilenet_v2(pretrained=True)
        case ModelType.EfficientNet:
            model = models.efficientnet_v2_s(pretrained=True)

    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)
    dynamic_axes = {
        "input": {0: "batch_size"},
    }

    torch.onnx.export(
        model,
        dummy_input,
        os.path.join(WEIGHTS_DIR, f"{model_type.value}.onnx"),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        default="resnet18",
        type=ModelType,
        help="Version of whisper model",
    )

    args = parser.parse_args()
    main(args.model_type)
