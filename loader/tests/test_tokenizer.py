from llama.llama.tokenizer import Tokenizer

TEST_TOKENIZER_PATH = "data/llama3_8b/tokenizer.model"

class TestTokenizer:
    def test_basic_load(self) -> None:
        tokenizer = Tokenizer(TEST_TOKENIZER_PATH)
        assert tokenizer is not None
        assert hasattr(tokenizer, 'encode')
        assert hasattr(tokenizer, 'decode')
    
    def test_encode(self) -> None:
        tokenizer = Tokenizer(TEST_TOKENIZER_PATH)
        encoded = tokenizer.encode("Hello, world!", bos=True, eos=True)
        assert len(encoded) > 0 and all(isinstance(elem, int) for elem in encoded)
        assert encoded == [128000, 9906, 11, 1917, 0, 128001]
        assert encoded[0] == tokenizer.bos_id
        assert encoded[-1] == tokenizer.eos_id
    
    def test_decode(self) -> None:
        tokenizer = Tokenizer(TEST_TOKENIZER_PATH)
        decoded = tokenizer.decode([9906, 11, 1917, 0])
        assert decoded == "Hello, world!"
    
    def test_roundtrip(self) -> None:
        tokenizer = Tokenizer(TEST_TOKENIZER_PATH)
        original = "Hello, world!"
        assert tokenizer.decode(tokenizer.encode(original, bos=False, eos=False)) == original