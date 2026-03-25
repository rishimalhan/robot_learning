#!/usr/bin/env python3
"""
Tiny Transformer demo: learn from ~15 pages of text and generate a Thanksgiving summary.

- Character-level vocabulary for simplicity.
- Single Transformer decoder block (multi-head attention + feed-forward).
- Trains on next-character prediction.
- After training, feed a few “prompt” characters (e.g., “Thanksgiving:”) and let it generate a summary.


(robot_learning) rishimalhan@macbook robot_learning % python src/neural_engine/scripts/thanksgiving_transformer.py --train

Model configuration -> vocab=72, layers=3, embed_dim=384, heads=8, params=5.48M
Step 1/15000 | loss=4.4387
Step 500/15000 | loss=1.6169
Step 1000/15000 | loss=0.6086
Step 1500/15000 | loss=0.3105
Step 2000/15000 | loss=0.2183
Step 2500/15000 | loss=0.1321
Step 3000/15000 | loss=0.1187
Step 3500/15000 | loss=0.1277
Step 4000/15000 | loss=0.0964
Step 4500/15000 | loss=0.0754
Step 5000/15000 | loss=0.0805
Step 5500/15000 | loss=0.0732
Step 6000/15000 | loss=0.0680
Step 6500/15000 | loss=0.0625
Step 7000/15000 | loss=0.0663
Step 7500/15000 | loss=0.0557
Step 8000/15000 | loss=0.0537
Step 8500/15000 | loss=0.0521
Step 9000/15000 | loss=0.0553
Step 9500/15000 | loss=0.0559
Step 10000/15000 | loss=0.0516
Step 10500/15000 | loss=0.0455
Step 11000/15000 | loss=0.0516
Step 11500/15000 | loss=0.0518
Step 12000/15000 | loss=0.0503
Step 12500/15000 | loss=0.0428
Step 13000/15000 | loss=0.0435
Step 13500/15000 | loss=0.0384
Step 14000/15000 | loss=0.0407
Step 14500/15000 | loss=0.0422
Step 15000/15000 | loss=0.0410
Training complete. Model saved.
=== Thanksgiving summary ===
Hello Linkedin Thanksgiving briefings coming up: to mend day of feasting and celebration to thank God in times of plenty. As an annual celebration of the harvest and its bounty, moreover, Thanksgiving falls under a category of festivals that spans cultures, continents and millennia. In ancient times, the Egyptians, Greeks and Romans feasted and paid tribute to their gods after the fall harvest. Thanksgiving also bears a resemblance to the ancient Jewish harvest festival of Sukkot. Finally, historians have noted that Native Americans had a rich tradition of commemorating the fall harvest with feasting and merrymaking long before Europeans set foot on America’s shores sement days of thanks during their presidencies. In 1817, New York became the first state to officially adopt an annual Thanksgiving holiday. Other states, especially in New
(robot_learning) rishimalhan@macbook robot_learning %
(robot_learning) rishimalhan@macbook robot_learning %

"""

import argparse
import math
import pathlib
import random
import re
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------
# Config
# ------------------
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
CURRENT_DIR = pathlib.Path(__file__).resolve().parent
DATA_PATH = CURRENT_DIR / "data_training_3.txt"  # your ~15 pages of notes
CKPT_PATH = CURRENT_DIR / "tiny_thanksgiving_transformer.pt"
CONTEXT_LEN = 128
BATCH_SIZE = 16
EMBED_DIM = 384
NUM_HEADS = 8
FF_DIM = 768
NUM_LAYERS = 3
DROPOUT = 0.1
LR = 3e-4
TOTAL_STEPS = 15000  # keep tiny so the script finishes quickly
GENERATE_LEN = 400  # number of tokens to synthesize
START_PROMPT = "Hello Linkedin! Thanksgiving briefings coming up:\n"
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
EOS_TOKEN = "<EOS>"

torch.manual_seed(42)
random.seed(42)


# ------------------
# Data loading
# ------------------
def load_text(path: pathlib.Path) -> str:
    text = path.read_text(encoding="utf-8")
    # Normalize whitespace lightly
    text = " ".join(text.split())
    return text


def tokenize(text: str) -> List[str]:
    return re.findall(r"\w+|[^\s\w]", text)


def build_vocab(tokens: List[str]) -> Tuple[List[str], dict]:
    vocab = [PAD_TOKEN, UNK_TOKEN]
    uniques = sorted(set(tokens + [EOS_TOKEN]))
    vocab.extend(uniques)
    stoi = {tok: i for i, tok in enumerate(vocab)}
    return vocab, stoi


def encode(tokens: List[str], stoi: dict) -> torch.Tensor:
    ids = [stoi.get(tok, stoi[UNK_TOKEN]) for tok in tokens]
    return torch.tensor(ids, dtype=torch.long)


def detokenize(tokens: List[str]) -> str:
    out = []
    for tok in tokens:
        if tok in {PAD_TOKEN, EOS_TOKEN}:
            continue
        if not out:
            out.append(tok)
            continue
        if re.match(r"[^\w\s]", tok):
            out[-1] += tok
        else:
            out.append(" " + tok)
    return "".join(out).strip()


def get_batches(data: torch.Tensor, block_size: int, batch_size: int):
    # simple infinite generator of random chunks
    while True:
        idx = torch.randint(0, len(data) - block_size - 1, (batch_size,))
        x = torch.stack([data[i : i + block_size] for i in idx])
        y = torch.stack([data[i + 1 : i + block_size + 1] for i in idx])
        yield x, y


# ------------------
# Model
# ------------------
class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, context_len: int):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, EMBED_DIM)
        self.pos_emb = nn.Parameter(torch.zeros(1, context_len, EMBED_DIM))
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=EMBED_DIM,
            nhead=NUM_HEADS,
            dim_feedforward=FF_DIM,
            dropout=DROPOUT,
            batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, NUM_LAYERS)
        self.ln = nn.LayerNorm(EMBED_DIM)
        self.head = nn.Linear(EMBED_DIM, vocab_size)

    def forward(self, idx: torch.Tensor):
        """
        idx: [B, T]
        """
        B, T = idx.shape
        tok = self.token_emb(idx)  # [B, T, E]
        pos = self.pos_emb[:, :T, :]
        x = tok + pos

        # Generate causal mask so token i can only attend to <= i
        mask = torch.triu(torch.ones(T, T, device=idx.device), diagonal=1).bool()
        # TransformerDecoder expects tgt, memory. We don't have an encoder,
        # so we feed zeros as "memory" and use x as tgt. Another simple trick is
        # to treat x as both memory and target via identity.
        mem = torch.zeros(1, 1, EMBED_DIM, device=idx.device)
        x = self.transformer(x, mem.expand(B, 1, EMBED_DIM), tgt_mask=mask)
        x = self.ln(x)
        logits = self.head(x)
        return logits


# ------------------
# Training loop
# ------------------
def train():
    raw_text = load_text(DATA_PATH)
    tokens = tokenize(raw_text)
    tokens.append(EOS_TOKEN)
    vocab, stoi = build_vocab(tokens)
    itos = {i: tok for tok, i in stoi.items()}
    vocab_size = len(vocab)

    data = encode(tokens, stoi)
    batches = get_batches(data, CONTEXT_LEN, BATCH_SIZE)

    model = TransformerLM(vocab_size, CONTEXT_LEN).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(
        f"Model configuration -> vocab={vocab_size}, layers={NUM_LAYERS}, "
        f"embed_dim={EMBED_DIM}, heads={NUM_HEADS}, params={total_params/1e6:.2f}M"
    )
    optim = torch.optim.AdamW(model.parameters(), lr=LR)

    model.train()
    for step in range(1, TOTAL_STEPS + 1):
        xb, yb = next(batches)
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)

        logits = model(xb)
        loss = F.cross_entropy(logits.view(-1, vocab_size), yb.reshape(-1))

        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()

        if step % 500 == 0 or step == 1:
            print(f"Step {step}/{TOTAL_STEPS} | loss={loss.item():.4f}")

    torch.save(
        {
            "model_state": model.state_dict(),
            "stoi": stoi,
            "itos": itos,
        },
        CKPT_PATH,
    )
    print("Training complete. Model saved.")
    return model, stoi, itos


# ------------------
# Generation
# ------------------
@torch.no_grad()
def generate(model: TransformerLM, stoi: dict, itos: dict, prompt: str, max_len: int):
    model.eval()
    prompt_tokens = tokenize(prompt)
    prompt_ids = [stoi.get(tok, stoi[UNK_TOKEN]) for tok in prompt_tokens]
    if not prompt_ids:
        prompt_ids = [stoi[EOS_TOKEN]]
    generated = prompt_ids[-CONTEXT_LEN:]

    for _ in range(max_len):
        idx = torch.tensor([generated[-CONTEXT_LEN:]], device=DEVICE)
        logits = model(idx)
        logits = logits[:, -1, :] / 0.7  # temperature
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).item()
        generated.append(next_id)
        if itos[next_id] == EOS_TOKEN:
            break

    new_tokens = [itos[i] for i in generated[len(prompt_ids) :]]
    return detokenize(new_tokens)


def load_checkpoint():
    if not CKPT_PATH.exists():
        raise FileNotFoundError(
            f"No checkpoint at {CKPT_PATH}. Run with --train to create one."
        )
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    stoi = ckpt["stoi"]
    itos = ckpt["itos"]
    model = TransformerLM(len(stoi), CONTEXT_LEN).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, stoi, itos


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tiny Thanksgiving Transformer demo")
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train a new model (always runs if no checkpoint exists).",
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default=START_PROMPT,
        help="Seed text for generation.",
    )
    parser.add_argument(
        "--length",
        type=int,
        default=GENERATE_LEN,
        help="Number of characters to generate.",
    )
    args = parser.parse_args()

    if args.train or not CKPT_PATH.exists():
        model, stoi, itos = train()
    else:
        model, stoi, itos = load_checkpoint()

    summary = generate(model, stoi, itos, args.prompt, args.length)
    print("=== Thanksgiving summary ===")
    print(summary)
