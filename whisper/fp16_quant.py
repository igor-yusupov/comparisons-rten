import onnx
from onnxconverter_common import float16


def quant_encoder():
    fp32_path = "weights/base_encoder.onnx"
    fp16_path = "weights/encoder_fp16.onnx"

    model = onnx.load(fp32_path)
    model_fp16 = float16.convert_float_to_float16(model)
    onnx.save(model_fp16, fp16_path)


def quant_decoder():
    fp32_path = "weights/base_decoder.onnx"
    fp16_path = "weights/decoder_fp16.onnx"

    model = onnx.load(fp32_path)
    model_fp16 = float16.convert_float_to_float16(model)
    onnx.save(model_fp16, fp16_path)


def main():
    quant_encoder()
    quant_decoder()


if __name__ == "__main__":
    main()
