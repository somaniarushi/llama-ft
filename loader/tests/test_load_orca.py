from loader.load import IntelOrcaTokensDataLoader, IntelOrcaTokens
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
        assert data is not None, f"Data is None for index 0"
        assert all(key in data for key in ['question', 'chosen', 'rejected']), f"Data keys are not as expected: {data.keys()}"

    def test_load_val_data_dict(self) -> None:
        tokenizer = Tokenizer(TOKENIZER_PATH)
        loader = IntelOrcaTokensDataLoader(tokenizer=tokenizer, parquet_path=TEST_PATH)
        data = loader[0]
        assert len(loader) == 1286, f"Expected 1286 rows, got {len(loader)}"
        assert data is not None, f"Data is None for index 0"
        assert all(key in data for key in ['question', 'chosen', 'rejected']), f"Data keys are not as expected: {data.keys()}"
    
    def test_load_tokens(self) -> None:
        tokenizer = Tokenizer(TOKENIZER_PATH)
        loader = IntelOrcaTokensDataLoader(tokenizer=tokenizer, parquet_path=DATA_PATH)
        data = loader[0]
        tokens: IntelOrcaTokens = loader.row_to_tokens(data)
        assert tokens is not None, f"Tokens are None for index 0"
        assert len(tokens.tokens) == len(tokens.loss_mask), f"Tokens and loss mask are not the same length"
        
        # Assert that the first token is bos, and the last token is eos
        assert tokens.tokens[0] == tokenizer.bos_id, f"First token is not bos"
        assert tokens.tokens[-1] == tokenizer.eos_id, f"Last token is not eos"
        
        # get number of tokens occupied by question
        num_question_tokens = len(tokenizer.encode(data['question'], bos=True, eos=False))
        assert tokens.loss_mask[:num_question_tokens].sum() == 0, f"Loss mask is not zero for question tokens"
        assert all(tokens.loss_mask[num_question_tokens:]), f"Loss mask is not one for chosen and rejected tokens"
        
        
        # When detokenized, the tokens should be equal to the original question + chosen
        detokenized = tokenizer.decode(tokens.tokens.tolist()[1:-1]) # drop bos and eos
        assert detokenized == data['question'] + data['chosen'], f"Detokenized tokens do not match question + chosen"