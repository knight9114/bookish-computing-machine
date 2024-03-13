from argparse import ArgumentParser, Namespace
from pathlib import Path
import torch
from transformers import PreTrainedTokenizerFast
from safetensors import safe_open
from gpt import Gpt


def main(argv: tuple[str] | None = None):
    args = parse_cli_arguments(argv)

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=args.tokenizer,
        bos_token="[SOS]",
        eos_token="[EOS]",
        pad_token="[PAD]",
        unk_token="[UNK]",
    )

    ckpt = {}
    with safe_open(args.ckpt, framework="pt", device="cpu") as fp:
        for key in fp.keys():
            ckpt[key] = fp.get_tensor(key)

    net = Gpt(
        vocab_size=tokenizer.vocab_size,
        n_layers=args.n_layers,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
        max_seq_len=args.max_length,
        bias=args.use_bias,
    ).train(False)

    # NOTE: `strict=True` raises assertion for buffers (timesteps and masks)
    #       which are already present in the model initialization
    net.load_state_dict(ckpt, strict=False)

    match (args.mode, args.qtype):
        case ("legacy", "none"):
            torch.onnx.export(
                model=net,
                args=torch.LongTensor(
                    [[tokenizer.bos_token_id, tokenizer.eos_token_id]]
                ),
                f=args.output,
                input_names=["prefixes"],
                output_names=["logits"],
                dynamic_axes={
                    "prefixes": {0: "batch_axis", 1: "prefix_length"},
                    "logits": {0: "batch_axis", 1: "prefix_length"},
                },
            )

        case ("dynamo", "none"):
            ortnet = torch.onnx.dynamo_export(
                net,
                torch.LongTensor([[tokenizer.bos_token_id, tokenizer.eos_token_id]]),
                export_options=torch.onnx.ExportOptions(dynamic_shapes=True),
            )
            ortnet.save(args.output)

        case ("dynamo", "ptdq"):
            qi8net = torch.ao.quantization.quantize_dynamic(
                model=net,
                qconfig_spec={torch.nn.Linear},
                dtype=torch.qint8,
            )
            ortnet = torch.onnx.dynamo_export(
                qi8net,
                torch.LongTensor([[tokenizer.bos_token_id, tokenizer.eos_token_id]]),
                export_options=torch.onnx.ExportOptions(dynamic_shapes=True),
            )
            ortnet.save(args.output)

        case _:
            raise NotImplementedError


def parse_cli_arguments(argv: tuple[str] | None = None) -> Namespace:
    parser = ArgumentParser()

    data = parser.add_argument_group("Dataset Hyperparameters")
    data.add_argument("--name", default="roneneldan/TinyStories")
    data.add_argument("--tokenizer", required=True, type=str)

    model = parser.add_argument_group("Model Hyperparameters")
    model.add_argument("--ckpt", type=Path, required=True)
    model.add_argument("--n-layers", type=int, default=6)
    model.add_argument("--d-model", type=int, default=192)
    model.add_argument("--n-heads", type=int, default=6)
    model.add_argument("--d-ff", type=int, default=768)
    model.add_argument("--dropout", type=float, default=0.0)
    model.add_argument("--max-length", type=int, default=100)
    model.add_argument("--use-bias", action="store_true")

    convert = parser.add_argument_group("Conversion Hyperparameters")
    convert.add_argument("--output", type=Path, required=True)
    convert.add_argument("--mode", default="legacy", choices=("legacy", "dynamo"))
    convert.add_argument("--qtype", default="none", choices=("none", "ptdq", "ptsq"))

    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
