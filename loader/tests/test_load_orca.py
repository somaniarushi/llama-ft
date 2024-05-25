from loader.load import IntelOrcaTokensDataLoader
from llama.llama.tokenizer import Tokenizer

TOKENIZER_PATH = "data/llama3_8b/tokenizer.model"
DATA_PATH = "data/intel_orca/train.parquet"
TEST_PATH = "data/intel_orca/test.parquet"


class TestIntelOrcaDataLoader:
    def test_load_data_dict(self) -> None:
        tokenizer = Tokenizer(TOKENIZER_PATH)
        loader = IntelOrcaTokensDataLoader(tokenizer=tokenizer, parquet_path=DATA_PATH)
        data = loader[0]
        assert len(loader) == 11573, f"Expected 11573 rows, got {len(loader)}"
        assert data is not None, "Data is None for index 0"
        assert all(
            key in data for key in ["question", "chosen", "rejected"]
        ), f"Data keys are not as expected: {data.keys()}"

    def test_load_val_data_dict(self) -> None:
        tokenizer = Tokenizer(TOKENIZER_PATH)
        loader = IntelOrcaTokensDataLoader(tokenizer=tokenizer, parquet_path=TEST_PATH)
        data = loader[0]
        assert len(loader) == 1286, f"Expected 1286 rows, got {len(loader)}"
        assert data is not None, "Data is None for index 0"
        assert all(
            key in data for key in ["question", "chosen", "rejected"]
        ), f"Data keys are not as expected: {data.keys()}"

    def test_load_tokens(self) -> None:
        tokenizer = Tokenizer(TOKENIZER_PATH)
        loader = IntelOrcaTokensDataLoader(tokenizer=tokenizer, parquet_path=DATA_PATH)
        data = loader[0]
        tokens = loader.row_to_tokens(data)
        assert tokens is not None, "Tokens are None for index 0"

        # Assert that the first token is bos, and the last token is eos
        assert tokens[0] == tokenizer.bos_id, "First token is not bos"
        assert tokens[-1] == tokenizer.eos_id, "Last token is not eos"

        # When detokenized, the tokens should be equal to the original question + chosen
        detokenized = tokenizer.decode(tokens.tokens.tolist()[1:-1])  # drop bos and eos
        assert (
            detokenized == data["question"] + data["chosen"]
        ), "Detokenized tokens do not match question + chosen"
