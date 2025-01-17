import json
import time
import torch
from tqdm import tqdm
from pathlib import Path
from typing import Optional
from sentencepiece import SentencePieceProcessor

from llama2 import ModelArgs, Transformer

class Llama:

    def __init__(self, model:Transformer, tokenizer: SentencePieceProcessor, model_args: ModelArgs):
        self.model = model
        self.tokenizer = tokenizer
        self.args = model_args

    @staticmethod
    def build(checkpoints_dir: str, tokenizer_path: str, load_model: bool, max_seq_len: int, max_batch_size: int, device: str):

        prev_time = time.time()

        if load_model:
            checkpoints = sorted(Path(checkpoints_dir).glob(".pth"))
            assert len(checkpoints) > 0, "No Checkpoints Files Found"

            chk_path = checkpoints[0]

            print(f"Loading Checkpoint {chk_path}")
            checkpoint = torch.load(chk_path, map_location='cuda')
            print(f"Loaded checkpoint in {(time.time() - prev_time):.2f}s")

            prev_time = time.time()

        with open(Path(checkpoints_dir) / "params.json", "r") as f:
            params = json.loads(f.read())
        
        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            device=device,
            **params
        )

        tokenizer = SentencePieceProcessor()
        tokenizer.load(tokenizer_path)

        model_args.vocab_size = tokenizer.vocab_size()

        if device == "cuda":
            torch.set_default_tensor_type(torch.cuda.HalfStorage)
        else:
            torch.set_default_tensor_type(torch.BFloat16Storage)

        model = Transformer(model_args).to(device)

        if load_model:
            del checkpoint["rope.freqs"]
            model.load_state_dict(checkpoint, strict=True)
            print(f"Loaded State Dict in {(time.time() - prev_time):.2f}s")
        
        return Llama(model, tokenizer, model_args)