import torch
from typing import Tuple
from llama.llama.tokenizer import Tokenizer
from llama.llama.generation import Llama
from llama.llama.model import Transformer
from loader.load import IntelOrcaTokensDataLoader, IntelOrcaTokens


TOKENIZER_PATH = "data/llama3_8b/tokenizer.model"
TRAIN_PATH = "data/intel_orca/train.parquet"
TEST_PATH = "data/intel_orca/test.parquet"

MODEL_DIR = "data/llama3_8b"
MODEL_PATH = "data/llama3_8b/consolidated.00.pth"
MODEL_CONFIG_PATH = "data/llama3_8b/params.json"

def get_tokenizer_and_model() -> Tuple[Transformer, Tokenizer]:
    llama = Llama.build(
        ckpt_dir=MODEL_DIR,
        tokenizer_path=TOKENIZER_PATH,
        max_seq_len=2048, 
        max_batch_size=32,
    )
    return llama.model, llama.tokenizer

if __name__ == "__main__":
    model, tokenizer = get_tokenizer_and_model()
    train_loader = IntelOrcaTokensDataLoader(tokenizer=tokenizer, parquet_path=TRAIN_PATH)
    val_loader = IntelOrcaTokensDataLoader(tokenizer=tokenizer, parquet_path=TEST_PATH)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    loss = torch.nn.CrossEntropyLoss()
    
    data = train_loader[0]
    token_data: IntelOrcaTokens = train_loader.row_to_tokens(data)
    tokens = token_data.tokens.reshape(1, -1) # (bsz, seq_len)
    logits = model.forward(tokens, start_pos=0)
    
    loss = loss(logits, tokens)
    loss.backward()
    
    # Update the model
    optimizer.step()