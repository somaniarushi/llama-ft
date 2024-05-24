import torch
from torch.utils.data import DataLoader
from pathlib import Path
from typing import List
from typing import NamedTuple
import pandas as pd
from llama.llama.tokenizer import Tokenizer


class ParquetDictDataLoader(DataLoader):
    """
    Given a parquet path, loads the data into a pandas dataframe and
    returns a dict representation of the row when indexed.
    """

    def __init__(self, parquet_path: str) -> None:
        assert Path(parquet_path).exists(), f"File {parquet_path} does not exist"
        self.dataframe = pd.read_parquet(parquet_path)

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> dict:
        return self.dataframe.iloc[idx].to_dict()


class IntelOrcaTokens(NamedTuple):
    tokens: torch.Tensor
    loss_mask: torch.Tensor


class IntelOrcaTokensDataLoader(ParquetDictDataLoader):
    def __init__(self, tokenizer: Tokenizer, parquet_path: str) -> None:
        super().__init__(parquet_path)
        self.tokenizer = tokenizer

    def row_to_tokens(self, row: dict, get_chosen: bool = True) -> dict:
        # Assert that the keys are 'question', 'chosen' and 'rejected'
        assert all(key in row for key in ["question", "chosen", "rejected"])
        question_tokens: List[int] = self.tokenizer.encode(
            row["question"], bos=True, eos=False
        )
        question_mask = [0] * len(question_tokens)  # No loss on the questions

        chosen_tokens: List[int] = self.tokenizer.encode(
            row["chosen"], bos=False, eos=True
        )
        chosen_mask = [1] * len(chosen_tokens)

        rejected_tokens: List[int] = self.tokenizer.encode(
            row["rejected"], bos=False, eos=True
        )
        rejected_mask = [1] * len(rejected_tokens)

        # If chosen, then return question + chosen
        if get_chosen:
            tokens = question_tokens + chosen_tokens
            mask = question_mask + chosen_mask
            return IntelOrcaTokens(
                tokens=torch.tensor(tokens), loss_mask=torch.tensor(mask)
            )
        else:
            tokens = question_tokens + rejected_tokens
            mask = question_mask + rejected_mask
            return IntelOrcaTokens(
                tokens=torch.tensor(tokens), loss_mask=torch.tensor(mask)
            )
