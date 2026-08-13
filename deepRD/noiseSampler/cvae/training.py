import torch
import copy
from tqdm import tqdm
from torch import nn
import torch.optim as optim
from deepRD.noiseSampler.cvae.losses import elbo_loss, reweight_losses


"""
Training and evaluation loops for CVAE.
"""

def flatten_batch_time(x: torch.Tensor):
    """
    Flatten [B, L, D...] → [B*L, D...]
    Leave [B, D...] unchanged.
    """
    if x.dim() <= 2:
        return x
    # x: [B, L, ...]
    B, L = x.shape[:2]
    return x.reshape(B * L, *x.shape[2:])

def train_cvae(model, train_loader, val_loader=None,
               epochs=50,
               lr=1e-3,
               beta_max=1.0, 
               free_bits=0.0, 
               warmup_epochs=10,
               grad_clip=1.0, save_path=None,
               early_stopping=True, patience=10, min_delta=1e-3,
               validate_every=1,
               weights_for_training=False,
               device='cpu'):

    train_total, train_nll, train_kl = [], [], []
    val_total, val_nll, val_kl = [], [], []

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler('cuda')

    best_val_loss = float("inf")
    best_epoch = None
    best_state = None
    epochs_no_improve = 0


    # Using per-sample losses for training weights

    for epoch in range(1, epochs + 1):

        # ---- KL warm-up ----
        if epoch <= warmup_epochs:
            beta = beta_max * (epoch / warmup_epochs)
        else:
            beta = beta_max

        model.train()
        total_loss, total_nll, total_kl = 0.0, 0.0, 0.0

        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for batch in loop:
            
            if weights_for_training==True:
                per_sample=True
                r_next, c, w = [x.to(device) for x in batch]
                w = flatten_batch_time(w).squeeze(-1)

            else:
                per_sample=False
                r_next, c = [x.to(device) for x in batch]

            r_next = flatten_batch_time(r_next)
            c = flatten_batch_time(c)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda'):
                dec_out, q, p = model(r_next, c)
                loss, nll, kl = elbo_loss(
                                        r_next, dec_out, q, p, 
                                        beta,
                                        per_sample=per_sample,
                                        free_bits=free_bits
                                        )
                
                if weights_for_training==True:
                    loss, nll, kl = reweight_losses([loss, nll, kl], w)

            scaler.scale(loss).backward()

            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            total_nll += nll.item()
            total_kl += kl.item()

            loop.set_postfix(
                loss=f"{loss.item():.4f}",
                NLL=f"{nll.item():.4f}",
                KL=f"{kl.item():.4f}",
                beta=f"{beta:.3f}"
            )

        scheduler.step()

        avg_train_loss = total_loss / len(train_loader)
        avg_train_nll = total_nll / len(train_loader)
        avg_train_kl = total_kl / len(train_loader)

        train_total.append(avg_train_loss)
        train_nll.append(avg_train_nll)
        train_kl.append(avg_train_kl)

        print(
            f"Epoch {epoch}: "
            f"train_total={avg_train_loss:.4f}, "
            f"train_nll={avg_train_nll:.4f}, "
            f"train_kl={avg_train_kl:.4f}, "
            f"beta={beta:.4f}"
        )

        # ---- validation ----
        do_validate = (
            val_loader is not None and
            epoch % validate_every == 0
        )

        if do_validate:
            val_metrics = evaluate_cvae(
                                        model, 
                                        val_loader, 
                                        beta,
                                        return_parts=True, 
                                        per_sample=per_sample, 
                                        device=device
                                        )

            avg_val_loss = val_metrics["total"]
            avg_val_nll = val_metrics["nll"]
            avg_val_kl = val_metrics["kl"]

            val_total.append(avg_val_loss)
            val_nll.append(avg_val_nll)
            val_kl.append(avg_val_kl)

            print(
                f"Validation: "
                f"val_total={avg_val_loss:.4f}, "
                f"val_nll={avg_val_nll:.4f}, "
                f"val_kl={avg_val_kl:.4f}"
            )

            # ---- early stopping only AFTER warmup ----
            if early_stopping and epoch > warmup_epochs:
                improved = (best_val_loss - avg_val_loss) > min_delta

                if improved:
                    best_val_loss = avg_val_loss
                    best_epoch = epoch
                    best_state = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0

                    if save_path is not None:
                        torch.save(
                            {
                                "epoch": epoch,
                                "model_state": best_state,
                                "best_val_loss": best_val_loss,
                                "beta": beta,
                            },
                            save_path
                        )
                else:
                    epochs_no_improve += 1
                    print(f"No significant val improvement for {epochs_no_improve} epoch(s).")

                    if epochs_no_improve >= patience:
                        print(
                            f"Early stopping triggered at epoch {epoch}. "
                            f"Best epoch was {best_epoch} with val_total={best_val_loss:.4f}."
                        )
                        break

        # if no validation / no early stopping, optionally still save latest
        elif save_path is not None and not early_stopping:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "beta": beta,
                },
                save_path
            )

    # ---- restore best model if early stopping used ----
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Restored best model from epoch {best_epoch} (val_total={best_val_loss:.4f}).")

    return {
        "train_total": train_total,
        "train_nll": train_nll,
        "train_kl": train_kl,
        "val_total": val_total,
        "val_nll": val_nll,
        "val_kl": val_kl,
        "best_val_loss": best_val_loss if best_epoch is not None else None,
        "best_epoch": best_epoch,
    }

@torch.no_grad()
def evaluate_cvae(model, val_loader, beta, return_parts=False, per_sample=False, device='cpu'):
    model.eval()

    total_loss, total_nll, total_kl = 0.0, 0.0, 0.0

    for batch in val_loader:

        if per_sample==True:
            r_next, c, w = [x.to(device) for x in batch]
            w = flatten_batch_time(w).squeeze(-1)
        else:
            r_next, c = [x.to(device) for x in batch]

        r_next = flatten_batch_time(r_next)
        c = flatten_batch_time(c)

        with torch.amp.autocast('cuda'):
            dec_out, q, p = model(r_next, c)
            loss, nll, kl = elbo_loss(r_next, dec_out, q, p, beta, per_sample=per_sample)

            if per_sample==True:
                loss, nll, kl = reweight_losses([loss, nll, kl], w)

        total_loss += loss.item()
        total_nll += nll.item()
        total_kl += kl.item()

    avg_loss = total_loss / len(val_loader)
    avg_nll = total_nll / len(val_loader)
    avg_kl = total_kl / len(val_loader)

    if return_parts:
        return {
            "total": avg_loss,
            "nll": avg_nll,
            "kl": avg_kl,
        }
    else:
        return avg_loss


# ============================================================
# General model-aware training  (CVAE variants)
# ============================================================
#
# Usage
# -----
#   losses = train_model(model, train_loader, val_loader, epochs=50, ...)
#
# Model-specific kwargs (pass via **model_kwargs):
#   CVAE / CVAE_LF  →  free_bits=0.0
#   CVAE_MDN        →  lambda_gate=0.0, lambda_comp=0.0
#
# Batch format is inferred from the number of elements per batch:
#   (r_next, c)        →  standard, uniform weights
#   (r_next, c, w)     →  standard, importance weights
#   (r_next, c, w, y)  →  MDN – weights + component labels required

import math as _math


# ── MDN loss primitives ───────────────────────────────────────────────────────

def _mdn_nll_per_sample(x, logit_pi, mu, log_sig):
    """Mixture-of-diagonal-Gaussians NLL.  x:[B,D] → [B]"""
    x = x[:, None, :]                                              # [B, 1, D]
    log_var = 2.0 * log_sig
    log_prob_comp = (-0.5 * (
        (x - mu) ** 2 * torch.exp(-log_var) + log_var + _math.log(2.0 * _math.pi)
    )).sum(dim=-1)                                                  # [B, K]
    log_pi = torch.nn.functional.log_softmax(logit_pi, dim=-1)    # [B, K]
    return -torch.logsumexp(log_pi + log_prob_comp, dim=-1)        # [B]


def _mdn_gate_ce_per_sample(logit_pi, y_comp):
    """Cross-entropy over mixture gates.  logit_pi:[B,K], y_comp:[B] → [B]"""
    return torch.nn.functional.cross_entropy(logit_pi, y_comp.long(), reduction="none")


def _mdn_comp_nll_per_sample(x, mu, log_sig, component_idx):
    """NLL of x under the *assigned* component.  → [B]"""
    idx   = component_idx.long()
    batch = torch.arange(x.shape[0], device=x.device)
    mu_k    = mu[batch, idx, :]       # [B, D]
    lsig_k  = log_sig[batch, idx, :]  # [B, D]
    return (0.5 * (
        ((x - mu_k) / torch.exp(lsig_k)) ** 2
        + 2.0 * lsig_k
        + _math.log(2.0 * _math.pi)
    )).sum(dim=-1)                     # [B]


# ── batch unpacking ───────────────────────────────────────────────────────────

def _unpack_batch(batch, device):
    """
    Returns (r_next, c, w, y).
    w = None if no weight column; y = None if no component-label column.
    """
    n = len(batch)
    if n == 2:
        r_next, c = [t.to(device) for t in batch]
        return r_next, c, None, None
    if n == 3:
        r_next, c, w = [t.to(device) for t in batch]
        return r_next, c, w, None
    if n == 4:
        r_next, c, w, y = [t.to(device) for t in batch]
        return r_next, c, w, y
    raise ValueError(f"Unexpected batch length {n}; expected 2, 3, or 4.")


def _weighted_mean(tensors, w):
    """Return importance-weighted means.  w=None → plain mean."""
    if w is None:
        return [t.mean() for t in tensors]
    wsum = w.sum()
    return [(w * t).sum() / wsum for t in tensors]


# ── per-class step functions ──────────────────────────────────────────────────

def _step_cvae(model, batch, beta, device, free_bits=0.0):
    """
    Forward + loss for standard CVAE (Gaussian decoder).
    Returns (loss, {"nll": scalar_tensor, "kl": scalar_tensor}).
    """
    from deepRD.noiseSampler.cvae.losses import elbo_loss
    r_next, c, w, _ = _unpack_batch(batch, device)
    r_next = flatten_batch_time(r_next)
    c      = flatten_batch_time(c)
    if w is not None:
        w = flatten_batch_time(w).squeeze(-1)

    dec_out, q, p = model(r_next, c)
    loss, nll, kl = elbo_loss(
        r_next, dec_out, q, p, beta,
        per_sample=(w is not None),
        free_bits=free_bits,
    )
    if w is not None:
        loss, nll, kl = _weighted_mean([loss, nll, kl], w)

    return loss, {"nll": nll, "kl": kl}


def _step_mdn(model, batch, beta, device, lambda_gate=0.0, lambda_comp=0.0):
    """
    Forward + loss for CVAE_MDN (mixture-density decoder).
    Batch must be (r_next, c, w, y).
    Returns (loss, {"nll", "kl", "gate", "comp"}).
    """
    from deepRD.noiseSampler.cvae.losses import kl_diag_per_sample
    r_next, c, w, y = _unpack_batch(batch, device)
    r_next = flatten_batch_time(r_next)
    c      = flatten_batch_time(c)
    w      = flatten_batch_time(w).squeeze(-1)
    y      = flatten_batch_time(y).squeeze(-1)

    dec_out, q, p = model(r_next, c)
    q_mu, q_logv = q
    p_mu, p_logv = p
    logit_pi, mu_r, log_sig_r = dec_out

    nll  = _mdn_nll_per_sample(r_next, logit_pi, mu_r, log_sig_r)  # [B]
    kl   = kl_diag_per_sample(q_mu, q_logv, p_mu, p_logv)          # [B]
    gate = _mdn_gate_ce_per_sample(logit_pi, y)                     # [B]
    comp = _mdn_comp_nll_per_sample(r_next, mu_r, log_sig_r, y)    # [B]

    per_sample_loss = nll + beta * kl + lambda_gate * gate + lambda_comp * comp
    loss, nll_s, kl_s, gate_s, comp_s = _weighted_mean(
        [per_sample_loss, nll, kl, gate, comp], w
    )
    return loss, {"nll": nll_s, "kl": kl_s, "gate": gate_s, "comp": comp_s}


# ── dispatch ──────────────────────────────────────────────────────────────────

def _resolve_step_fn(model, model_kwargs):
    """Return a bound step_fn(model, batch, beta, device) → (loss, parts)."""
    from deepRD.noiseSampler.cvaeSampler import CVAE_MDN
    if isinstance(model, CVAE_MDN):
        lg = float(model_kwargs.get("lambda_gate") or 0.0)
        lc = float(model_kwargs.get("lambda_comp") or 0.0)
        return lambda m, b, beta, dev: _step_mdn(m, b, beta, dev, lg, lc)
    else:
        fb = float(model_kwargs.get("free_bits") or 0.0)
        return lambda m, b, beta, dev: _step_cvae(m, b, beta, dev, fb)


# ── general training loop ─────────────────────────────────────────────────────

def train_model(
    model,
    train_loader,
    val_loader=None,
    epochs=50,
    lr=1e-3,
    beta_max=1.0,
    warmup_epochs=10,
    grad_clip=1.0,
    save_path=None,
    early_stopping=True,
    patience=10,
    min_delta=1e-3,
    validate_every=1,
    device="cpu",
    **model_kwargs,
):
    """
    General training loop for CVAE variants (standard and MDN).

    All shared logic (optimizer, LR schedule, KL warmup, AMP, early stopping,
    checkpointing) lives here.  Model-specific forward/loss dispatch is handled
    automatically from the model class.

    Parameters
    ----------
    model_kwargs : forwarded to the per-class step function, e.g.
        free_bits=0.2          # for CVAE / CVAE_LF
        lambda_gate=5.0        # for CVAE_MDN
        lambda_comp=0.1        # for CVAE_MDN

    Returns
    -------
    dict with keys  train_total, train_nll, train_kl [, train_gate, train_comp],
                    val_total,   val_nll,   val_kl   [, val_gate,   val_comp],
                    best_val_loss, best_epoch.
    """
    step_fn = _resolve_step_fn(model, model_kwargs)

    use_amp    = torch.cuda.is_available() and ("cuda" in str(device))
    dev_type   = "cuda" if use_amp else "cpu"
    amp_scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history: dict = {"train_total": [], "val_total": []}
    best_val_loss  = float("inf")
    best_epoch     = None
    best_state     = None
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        beta = beta_max * min(epoch / max(warmup_epochs, 1), 1.0)

        # ---- train epoch ----
        model.train()
        epoch_sums: dict[str, float] = {}

        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for batch in loop:
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(dev_type, enabled=use_amp):
                loss, parts = step_fn(model, batch, beta, device)

            amp_scaler.scale(loss).backward()
            if grad_clip is not None:
                amp_scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            amp_scaler.step(optimizer)
            amp_scaler.update()

            epoch_sums["total"] = epoch_sums.get("total", 0.0) + loss.item()
            for k, v in parts.items():
                epoch_sums[k] = epoch_sums.get(k, 0.0) + v.item()

            loop.set_postfix(
                loss=f"{loss.item():.4f}",
                beta=f"{beta:.3f}",
                **{k: f"{v.item():.4f}" for k, v in parts.items()},
            )

        scheduler.step()
        n = len(train_loader)
        avgs = {k: v / n for k, v in epoch_sums.items()}

        history["train_total"].append(avgs["total"])
        for k, v in avgs.items():
            if k != "total":
                history.setdefault(f"train_{k}", []).append(v)

        print(
            f"Epoch {epoch}: "
            + "  ".join(f"train_{k}={v:.4f}" for k, v in avgs.items())
            + f"  beta={beta:.4f}"
        )

        # ---- validate ----
        do_validate = val_loader is not None and epoch % validate_every == 0
        if do_validate:
            val_metrics = evaluate_model(model, val_loader, beta, device=device, **model_kwargs)
            avg_val_loss = val_metrics["total"]

            history["val_total"].append(avg_val_loss)
            for k, v in val_metrics.items():
                if k != "total":
                    history.setdefault(f"val_{k}", []).append(v)

            print("Validation: " + "  ".join(f"val_{k}={v:.4f}" for k, v in val_metrics.items()))

            if early_stopping and epoch > warmup_epochs:
                improved = (best_val_loss - avg_val_loss) > min_delta
                if improved:
                    best_val_loss = avg_val_loss
                    best_epoch    = epoch
                    best_state    = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                    if save_path is not None:
                        torch.save(
                            {
                                "epoch": epoch,
                                "model_state": best_state,
                                "best_val_loss": best_val_loss,
                                "beta": beta,
                            },
                            save_path,
                        )
                else:
                    epochs_no_improve += 1
                    print(f"No significant val improvement for {epochs_no_improve} epoch(s).")
                    if epochs_no_improve >= patience:
                        print(
                            f"Early stopping triggered at epoch {epoch}. "
                            f"Best epoch was {best_epoch} with val_total={best_val_loss:.4f}."
                        )
                        break

        elif save_path is not None and not early_stopping:
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict(), "beta": beta},
                save_path,
            )

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Restored best model from epoch {best_epoch} (val_total={best_val_loss:.4f}).")

    history["best_val_loss"] = best_val_loss if best_epoch is not None else None
    history["best_epoch"]    = best_epoch
    return history


# ── general evaluation ────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_model(model, val_loader, beta, device="cpu", **model_kwargs):
    """
    Evaluation counterpart to train_model.
    Returns a dict of averaged metrics: {"total", "nll", "kl", ...}.
    Keys match whatever the step function for this model class produces.
    """
    step_fn  = _resolve_step_fn(model, model_kwargs)
    use_amp  = torch.cuda.is_available() and ("cuda" in str(device))
    dev_type = "cuda" if use_amp else "cpu"

    model.eval()
    totals: dict[str, float] = {}

    for batch in val_loader:
        with torch.amp.autocast(dev_type, enabled=use_amp):
            loss, parts = step_fn(model, batch, beta, device)

        totals["total"] = totals.get("total", 0.0) + loss.item()
        for k, v in parts.items():
            totals[k] = totals.get(k, 0.0) + v.item()

    n = len(val_loader)
    return {k: v / n for k, v in totals.items()}