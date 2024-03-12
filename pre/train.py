from argparse import ArgumentParser, Namespace
import tqdm
import torch
from torch import nn, optim
from transformers import PreTrainedTokenizerFast
from safetensors.torch import save_file
from gpt import Gpt
from dataset import TextDataset


def main(argv: tuple[str] | None = None):
    args = parse_cli_arguments(argv)

    ctx = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=args.tokenizer,
        bos_token="[SOS]",
        eos_token="[EOS]",
        pad_token="[PAD]",
        unk_token="[UNK]",
    )
    ds = TextDataset(
        tokenizer=tokenizer,
        name=args.name,
        max_token_length=args.max_length,
        min_token_length=args.min_length,
    )
    net = Gpt(
        vocab_size=tokenizer.vocab_size,
        n_layers=args.n_layers,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
        max_seq_len=args.max_length,
        bias=args.use_bias,
    ).to(ctx)
    optimizer = optim.AdamW(net.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    min_validation_loss = float("inf")
    for epoch in tqdm.trange(args.n_epochs):
        net.train(True)
        for i, batch in enumerate(
            tqdm.tqdm(ds.train_dataloader(args.batch_size), leave=None)
        ):
            srcs, tgts = batch["srcs"].to(ctx), batch["tgts"].to(ctx)

            optimizer.zero_grad()
            logits = net(srcs)
            loss = criterion(logits.view(-1, logits.size(-1)), tgts.view(-1))
            loss.backward()
            optimizer.step()

        net.train(False)
        val_loss = 0.0
        with torch.no_grad():
            for i, batch in enumerate(
                tqdm.tqdm(ds.validation_dataloader(args.batch_size), leave=None)
            ):
                srcs, tgts = batch["srcs"].to(ctx), batch["tgts"].to(ctx)
                logits = net(srcs)
                loss = criterion(logits.view(-1, logits.size(-1)), tgts.view(-1))
                val_loss += (loss.item() - val_loss) / (i + 1)

            if val_loss < min_validation_loss:
                min_validation_loss = val_loss
                save_file(
                    tensors={n: p for n, p in net.named_parameters()},
                    filename="model.safetensors",
                )


def parse_cli_arguments(argv: tuple[str] | None = None) -> Namespace:
    parser = ArgumentParser()

    data = parser.add_argument_group("Dataset Hyperparameters")
    data.add_argument("--name", default="roneneldan/TinyStories")
    data.add_argument("--tokenizer", required=True, type=str)
    data.add_argument("--max-length", type=int, default=100)
    data.add_argument("--min-length", type=int, default=5)

    model = parser.add_argument_group("Model Hyperparameters")
    model.add_argument("--n-layers", type=int, default=6)
    model.add_argument("--d-model", type=int, default=192)
    model.add_argument("--n-heads", type=int, default=6)
    model.add_argument("--d-ff", type=int, default=768)
    model.add_argument("--dropout", type=float, default=0.0)
    model.add_argument("--use-bias", action="store_true")

    train = parser.add_argument_group("Training Hyperparameters")
    train.add_argument("--n-epochs", type=int, default=20)
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--learning-rate", type=float, default=3e-4)

    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
