"""In-context quadratic regression test for BDH.

Each sequence is a random quadratic y = a*x^2 + b*x + c with a,b,c ~ U(-1,1),
encoded as interleaved (x, y) byte tokens. The model is trained to predict
the next token autoregressively, which forces it to infer the quadratic from
earlier (x, y) pairs in order to predict later y's.

Train x's are drawn from [-1, 1]. Eval x's are drawn from [-2, 2] to measure
extrapolation outside the training domain.
"""

from contextlib import nullcontext

import bdh
import numpy as np
import torch
import torch.nn.functional as F

device = torch.device("mps")
torch.manual_seed(1337)
np.random.seed(1337)

# Quantization: we use the full 256-token vocab to bin values.
# x and y ranges must cover both the train and eval domains.
X_MIN, X_MAX = -2.0, 2.0
# With a,b,c in [-1,1] and |x| <= 2, |y| <= 1*4 + 1*2 + 1 = 7.
Y_MIN, Y_MAX = -7.0, 7.0
N_BINS = 256

POINTS_PER_SEQ = 128          # 128 (x,y) pairs -> 256-token sequences
BATCH_SIZE = 32
MAX_ITERS = 3000
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.1
LOG_FREQ = 100
EVAL_BATCHES = 16


def quantize(v, vmin, vmax):
    v = np.clip(v, vmin, vmax)
    return ((v - vmin) / (vmax - vmin) * (N_BINS - 1)).round().astype(np.int64)


def dequantize(b, vmin, vmax):
    return b.astype(np.float32) / (N_BINS - 1) * (vmax - vmin) + vmin


def make_batch(batch_size, x_low, x_high, return_truth=False):
    a = np.random.uniform(-1, 1, batch_size).astype(np.float32)
    b = np.random.uniform(-1, 1, batch_size).astype(np.float32)
    c = np.random.uniform(-1, 1, batch_size).astype(np.float32)
    xs = np.random.uniform(x_low, x_high, (batch_size, POINTS_PER_SEQ)).astype(np.float32)
    ys = a[:, None] * xs ** 2 + b[:, None] * xs + c[:, None]

    x_q = quantize(xs, X_MIN, X_MAX)
    y_q = quantize(ys, Y_MIN, Y_MAX)

    # Interleave: [x1, y1, x2, y2, ..., xN, yN]  -> length 2*POINTS_PER_SEQ.
    seq = np.stack([x_q, y_q], axis=-1).reshape(batch_size, -1)
    inp = torch.from_numpy(seq[:, :-1]).to(device)
    tgt = torch.from_numpy(seq[:, 1:]).to(device)
    if return_truth:
        return inp, tgt, xs, ys
    return inp, tgt


def y_position_mask(seq_len):
    # Input positions 0,1,2,3,... correspond to x1,y1,x2,y2,...
    # Targets are inp shifted by 1, so target at index i predicts token i+1.
    # We score only the positions where the target is a y (odd target index).
    # Target index = i+1 is odd  <=>  i is even.
    mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
    mask[0::2] = True
    return mask


@torch.no_grad()
def eval_extrapolation(model, x_low, x_high, n_batches=EVAL_BATCHES):
    """Average per-token MSE on dequantized y predictions."""
    model.eval()
    total_mse = 0.0
    total_n = 0
    for _ in range(n_batches):
        inp, tgt, _, ys_true = make_batch(BATCH_SIZE, x_low, x_high, return_truth=True)
        logits, _ = model(inp)
        # Positions whose target is a y.
        mask = y_position_mask(inp.size(1))
        y_logits = logits[:, mask, :]                  # B, POINTS_PER_SEQ, 256
        pred_bins = y_logits.argmax(dim=-1).cpu().numpy()
        pred_y = dequantize(pred_bins, Y_MIN, Y_MAX)
        # ys_true has shape (B, POINTS_PER_SEQ); each corresponds to one y-target.
        mse = ((pred_y - ys_true) ** 2).mean()
        total_mse += float(mse) * BATCH_SIZE
        total_n += BATCH_SIZE
    model.train()
    return total_mse / total_n


if __name__ == "__main__":
    config = bdh.BDHConfig()       # vocab_size=256 is already what we need
    model = bdh.BDH(config).to(device)
    # Note: torch.compile is skipped — Inductor has no MPS backend, so it
    # spends minutes in the joint-graph passes and then falls back anyway.
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    ctx = nullcontext()

    loss_acc, loss_steps = 0.0, 0
    for step in range(MAX_ITERS):
        inp, tgt = make_batch(BATCH_SIZE, x_low=-1.0, x_high=1.0)
        with ctx:
            logits, _ = model(inp)
            # Only score y-predictions, not x-predictions (x is i.i.d. uniform).
            mask = y_position_mask(inp.size(1))
            loss = F.cross_entropy(
                logits[:, mask, :].reshape(-1, N_BINS),
                tgt[:, mask].reshape(-1),
            )
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_acc += float(loss)
        loss_steps += 1

        if step % LOG_FREQ == 0:
            avg = loss_acc / max(loss_steps, 1)
            print(f"step {step:4d}/{MAX_ITERS}  train_loss={avg:.4f}")
            loss_acc, loss_steps = 0.0, 0

    print("\nTraining done. Evaluating MSE on dequantized y predictions.")
    in_domain_mse = eval_extrapolation(model, x_low=-1.0, x_high=1.0)
    extrap_mse = eval_extrapolation(model, x_low=-2.0, x_high=2.0)
    # A baseline: predicting the dataset mean of y (~0) gives MSE ~ Var(y).
    print(f"in-domain  x in [-1, 1]  MSE: {in_domain_mse:.4f}")
    print(f"extrapolat x in [-2, 2]  MSE: {extrap_mse:.4f}")
