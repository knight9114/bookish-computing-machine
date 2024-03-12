from datasets import load_dataset
from transformers import PreTrainedTokenizerFast
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader


class TextDataset(Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        name: str = "roneneldan/TinyStories",
        max_token_length: int = 256,
        min_token_length: int = 1,
        **kwargs,
    ):
        super().__init__()

        ds = load_dataset(name, **kwargs)
        ds = ds.map(lambda ex: tokenizer(ex["text"]), batched=True)
        self.ds = ds.filter(
            lambda ex: len(ex["input_ids"]) >= min_token_length
            and len(ex["input_ids"]) <= max_token_length
        )
        self.tokenizer = tokenizer

    def train_dataloader(self, batch_size: int) -> DataLoader:
        return DataLoader(
            dataset=self.ds["train"],
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=8,
            pin_memory=True,
        )

    def validation_dataloader(self, batch_size: int) -> DataLoader:
        return DataLoader(
            dataset=self.ds["validation"],
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=8,
        )

    def collate_fn(self, batch: list[dict]) -> dict[str, Tensor]:
        seqs = pad_sequence(
            sequences=[torch.LongTensor(ex["input_ids"]) for ex in batch],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        return {"srcs": seqs[:, :-1], "tgts": seqs[:, 1:]}
