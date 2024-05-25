import torch
from typing import Tuple
from llama.llama.tokenizer import Tokenizer
from llama.llama.generation import Llama
from llama.llama.model import Transformer
from loader.load import IntelOrcaTokensDataLoader
import lightning as L

TOKENIZER_PATH = "data/llama3_8b/tokenizer.model"
TRAIN_PATH = "data/intel_orca/train.parquet"
TEST_PATH = "data/intel_orca/test.parquet"

MODEL_DIR = "data/llama3_8b"
MODEL_PATH = "data/llama3_8b/consolidated.00.pth"
MODEL_CONFIG_PATH = "data/llama3_8b/params.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_BATCH_SIZE = 32
MAX_SEQ_LEN = 2048

VOCAB_SIZE = 128256


def get_tokenizer_and_model() -> Tuple[Transformer, Tokenizer]:
    llama = Llama.build(
        ckpt_dir=MODEL_DIR,
        tokenizer_path=TOKENIZER_PATH,
        max_seq_len=MAX_SEQ_LEN,
        max_batch_size=MAX_BATCH_SIZE,
    )
    return llama.model, llama.tokenizer


class LlamaTrainer(L.Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        tokens, loss_mask = batch
        logits = self.model(tokens, start_pos=0)
        loss = self.loss(logits, tokens, loss_mask)
        return loss

    def validation_step(self, batch, batch_idx):
        tokens, loss_mask = batch
        logits = self.model(tokens, start_pos=0)
        loss = self.loss(logits, tokens, loss_mask)
        return loss

    def loss(self, logits, tokens, loss_mask):
        loss = torch.nn.functional.cross_entropy(logits, tokens, reduction="none")
        loss = loss * loss_mask
        return loss.mean()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5)
        return optimizer


if __name__ == "__main__":
    model, tokenizer = get_tokenizer_and_model()
    train_loader = IntelOrcaTokensDataLoader(
        tokenizer=tokenizer, parquet_path=TRAIN_PATH, batch_size=MAX_BATCH_SIZE
    )
    val_loader = IntelOrcaTokensDataLoader(
        tokenizer=tokenizer, parquet_path=TEST_PATH, batch_size=MAX_BATCH_SIZE
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    loss = torch.nn.CrossEntropyLoss()

    model = torch._dynamo.OptimizedModule(
        model, dynamo_ctx=torch._dynamo.DynamoContext()
    )
    trainer = L.Trainer()
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )
