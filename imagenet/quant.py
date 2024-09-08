import argparse
import os

from onnxruntime.quantization import QuantType, quantize_dynamic

from utils import ModelType

WEIGHTS_DIR = "weights"


def _add_postfix_to_filename(file_path, postfix="_quant"):
    dir_name, base_name = os.path.split(file_path)
    file_name, file_extension = os.path.splitext(base_name)
    new_file_name = f"{file_name}{postfix}{file_extension}"
    new_file_path = os.path.join(dir_name, new_file_name)
    return new_file_path


def quant_model(fp32_path: str):
    int32_path = _add_postfix_to_filename(fp32_path)

    quantize_dynamic(fp32_path, int32_path, weight_type=QuantType.QUInt8)


def main(model_type: ModelType) -> None:
    weights_path = os.path.join(WEIGHTS_DIR, f"{model_type.value}.onnx")

    quant_model(weights_path)


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
