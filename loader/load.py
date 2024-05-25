import torch
from torch.utils.data import DataLoader
from pathlib import Path
from typing import List
import pandas as pd
from llama.llama.tokenizer import Tokenizer


class ParquetDictDataLoader(DataLoader):
    """
    Given a parquet path, loads the data into a pandas dataframe and
    returns a dict representation of the row when indexed.
    """

    def __init__(self, parquet_path: str, *args, **kwargs) -> None:
        super().__init__(self, *args, **kwargs)
        assert Path(parquet_path).exists(), f"File {parquet_path} does not exist"
        self.dataframe = pd.read_parquet(parquet_path)

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> dict:
        return self.dataframe.iloc[idx].to_dict()


class IntelOrcaTokensDataLoader(ParquetDictDataLoader):
    def __init__(
        self, tokenizer: Tokenizer, parquet_path: str, *args, **kwargs
    ) -> None:
        super().__init__(parquet_path, *args, **kwargs)
        self.tokenizer = tokenizer

    def row_to_tokens(self, row: dict, get_chosen: bool = True) -> torch.Tensor:
        # Assert that the keys are 'question', 'chosen' and 'rejected'
        assert all(key in row for key in ["question", "chosen", "rejected"])

        question_tokens: List[int] = self.tokenizer.encode(
            row["question"], bos=True, eos=False
        )
        # If chosen, then return question + chosen
        if get_chosen:
            chosen_tokens: List[int] = self.tokenizer.encode(
                row["chosen"], bos=False, eos=True
            )
            tokens = question_tokens + chosen_tokens
            return torch.tensor(tokens)
        else:
            rejected_tokens: List[int] = self.tokenizer.encode(
                row["rejected"], bos=False, eos=True
            )
            tokens = question_tokens + rejected_tokens
            return torch.tensor(tokens)

    def __getitem__(self, idx: int) -> torch.Tensor:
        data = super().__getitem__(idx)
        return self.row_to_tokens(data)
