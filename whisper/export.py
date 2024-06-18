# Main idea from repo: https://github.com/axinc-ai/whisper-export/tree/onnx-export

import argparse
import os
from enum import Enum

import torch
import torch.nn as nn

import whisper
from src.decoder import TextDecoder
from src.encoder import AudioEncoder


WEIGHTS_DIR = "weights"
ENCODER_NAME = "encoder"
DECODER_NAME = "decoder"


class ModelType(Enum):
    TINY = "tiny"
    BASE = "base"
    SMALL = "small"
    MEDIUM = "medium"


class EncoderModel(nn.Module):
    def __init__(self, encoder) -> None:
        super().__init__()
        self.encoder = encoder

    def forward(self, x: torch.Tensor):
        return self.encoder(x)


class DecoderModel(nn.Module):
    def __init__(self, decoder) -> None:
        super().__init__()
        self.decoder = decoder

    def forward(
        self,
        tokens,
        audio_features,
        kv_cache,
        offset,
    ):
        return self.decoder(
            tokens,
            audio_features,
            kv_cache,
            offset,
        )


def export_encoder(model, model_name):
    model = EncoderModel(model).eval()

    x = torch.zeros((1, 80, 3000), dtype=torch.float32)
    input_names = ["mel"]
    dynamic_axes = {"mel": {0: "batch_size"}}

    torch.onnx.export(
        model,
        x,
        model_name,
        verbose=False,
        opset_version=14,
        input_names=input_names,
        # output_names=output_names,
        dynamic_axes=dynamic_axes,
    )


def export_decoder(model, model_name):
    model = DecoderModel(model).eval()

    (
        tokens,
        audio_features,
        kv_cache,
        offset,
    ) = (
        torch.zeros((1, 1), dtype=torch.int32),
        torch.rand((1, 1500, 512), dtype=torch.float32),
        torch.zeros((12, 1, 451, 512), dtype=torch.float32),
        torch.tensor(0),
    )
    input_names = [
        "tokens",
        "audio_features",
        "kv_cache",
        "offset",
    ]
    output_names = [
        "logits",
        "output_kv_cache",
    ]

    dynamic_axes = {
        "tokens": {0: "batch_size", 1: "token_len"},
        "audio_features": {0: "batch_size"},
        "kv_cache": {1: "batch_size"},
        "logits": {0: "batch_size", 1: "token_len"},
        "output_kv_cache": {1: "batch_size"},
    }

    torch.onnx.export(
        model,
        (
            tokens,
            audio_features,
            kv_cache,
            offset,
        ),
        model_name,
        verbose=False,
        opset_version=14,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        default="base",
        type=ModelType,
        help="Version of whisper model",
    )

    args = parser.parse_args()
    model = whisper.load_model(args.model_type.value)
    model = model.eval()

    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    torch.save(
        model.encoder.state_dict(),
        os.path.join(
            WEIGHTS_DIR, f"{args.model_type.value}_{ENCODER_NAME}.pt"
        ),
    )
    torch.save(
        model.decoder.state_dict(),
        os.path.join(
            WEIGHTS_DIR, f"{args.model_type.value}_{DECODER_NAME}.pt"
        ),
    )

    dims = model.dims

    encoder = AudioEncoder(
        n_mels=dims.n_mels,
        n_ctx=dims.n_audio_ctx,
        n_state=dims.n_audio_state,
        n_head=dims.n_audio_head,
        n_layer=dims.n_audio_layer,
    )
    encoder.load_state_dict(
        torch.load(
            os.path.join(
                WEIGHTS_DIR, f"{args.model_type.value}_{ENCODER_NAME}.pt"
            )
        )
    )
    encoder = encoder.eval()

    decoder = TextDecoder(
        n_vocab=dims.n_vocab,
        n_ctx=dims.n_text_ctx,
        n_state=dims.n_text_state,
        n_head=dims.n_text_head,
        n_layer=dims.n_text_layer,
    )
    decoder.load_state_dict(
        torch.load(
            os.path.join(
                WEIGHTS_DIR, f"{args.model_type.value}_{DECODER_NAME}.pt"
            )
        )
    )
    decoder = decoder.eval()

    export_encoder(
        encoder,
        os.path.join(
            WEIGHTS_DIR, f"{args.model_type.value}_{ENCODER_NAME}.onnx"
        ),
    )
    export_decoder(
        decoder,
        os.path.join(
            WEIGHTS_DIR, f"{args.model_type.value}_{DECODER_NAME}.onnx"
        ),
    )


if __name__ == "__main__":
    main()
