from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic


def quant_encoder():
    fp32_path = Path("weights/base_encoder.onnx")
    int32_path = Path("weights/encoder_quant.onnx")

    quantize_dynamic(fp32_path, int32_path, weight_type=QuantType.QUInt8)


def quant_decoder():
    fp32_path = Path("weights/base_decoder.onnx")
    int32_path = Path("weights/decoder_quant.onnx")
    quantize_dynamic(fp32_path, int32_path, weight_type=QuantType.QUInt8)


def main():
    quant_encoder()
    quant_decoder()


if __name__ == "__main__":
    main()
