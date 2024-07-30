import argparse
import os

from onnxruntime.quantization import QuantType, quantize_dynamic

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


def main(args: argparse.Namespace) -> None:
    model_type = args.model_type
    weights_enc_path = ""
    weights_dec_path = ""

    if model_type == "tiny.en" or model_type == "tiny":
        weights_enc_path = os.path.join(WEIGHTS_DIR, "tiny_encoder.onnx")
        weights_dec_path = os.path.join(WEIGHTS_DIR, "tiny_decoder.onnx")
    elif model_type == "base.en" or model_type == "base":
        weights_enc_path = os.path.join(WEIGHTS_DIR, "base_encoder.onnx")
        weights_dec_path = os.path.join(WEIGHTS_DIR, "base_decoder.onnx")
    elif model_type == "small.en" or model_type == "small":
        weights_enc_path = os.path.join(WEIGHTS_DIR, "small_encoder.onnx")
        weights_dec_path = os.path.join(WEIGHTS_DIR, "small_decoder.onnx")
    elif model_type == "medium.en" or model_type == "medium":
        weights_enc_path = os.path.join(WEIGHTS_DIR, "medium_encoder.onnx")
        weights_dec_path = os.path.join(WEIGHTS_DIR, "medium_decoder.onnx")
    else:
        raise ValueError(f"There is no model type: {model_type}")
    quant_model(weights_enc_path)
    quant_model(weights_dec_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type", default="base", type=str, help="Use quant models"
    )
    args = parser.parse_args()
    main(args)
