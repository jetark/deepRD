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