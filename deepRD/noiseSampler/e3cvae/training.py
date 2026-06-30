import copy
import torch
import torch.optim as optim
from tqdm import tqdm

from .losses import e3_cvae_axial_loss, e3_cvae_isotropic_loss


def move_graph_batch_to_device(batch, device):
    """
    Move an E3 graph batch dictionary to a torch device.

    Tensor values are moved; non-tensor metadata such as num_graphs is kept.
    """
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def beta_schedule(epoch, beta_max=1.0, warmup_epochs=10):
    if warmup_epochs <= 0:
        return beta_max
    return beta_max * min(1.0, epoch / warmup_epochs)


def train_e3_cvae(
    model,
    train_loader,
    val_loader=None,
    epochs=50,
    lr=1e-3,
    beta_max=1.0,
    warmup_epochs=10,
    grad_clip=1.0,
    weight_decay=1e-4,
    save_path=None,
    early_stopping=True,
    patience=10,
    min_delta=1e-3,
    validate_every=1,
    device="cpu",
    free_bits=0.0,
):
    """
    Train E3DimerCVAE on graph batches.

    Each loader item must be a dict with at least:
        h_enc:       [B*2, encoder_irreps.dim]
        h_dec_base:  [B*2, decoder_base_irreps.dim]
        edge_index:  [2, 2*B]
        edge_vec:    [2*B, 3]
        edge_radial: [2*B, radial_dim]
        batch_index: [B*2]
        r_next:      [B*2, 3]
        num_graphs:  int
    """
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {
        "train_total": [],
        "train_nll": [],
        "train_kl": [],
        "val_total": [],
        "val_nll": [],
        "val_kl": [],
        "val_diagnostics": [],
        "best_val_loss": None,
        "best_epoch": None,
    }

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        beta = beta_schedule(epoch, beta_max=beta_max, warmup_epochs=warmup_epochs)
        model.train()

        total_loss = total_nll = total_kl = 0.0
        loop = tqdm(train_loader, desc=f"E3 epoch {epoch}/{epochs}")
        for batch in loop:
            batch = move_graph_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)

            outputs = model(batch)
            loss_fn = e3_cvae_isotropic_loss if model.isotropic else e3_cvae_axial_loss
            loss, nll, kl = loss_fn(outputs, batch, beta=beta, free_bits=free_bits)
            loss.backward()

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

            total_loss += loss.item()
            total_nll += nll.item()
            total_kl += kl.item()
            loop.set_postfix(
                loss=f"{loss.item():.4f}",
                NLL=f"{nll.item():.4f}",
                KL=f"{kl.item():.4f}",
                beta=f"{beta:.3f}",
            )

        scheduler.step()

        avg_loss = total_loss / len(train_loader)
        avg_nll = total_nll / len(train_loader)
        avg_kl = total_kl / len(train_loader)
        history["train_total"].append(avg_loss)
        history["train_nll"].append(avg_nll)
        history["train_kl"].append(avg_kl)

        print(
            f"Epoch {epoch}: train_total={avg_loss:.4f}, "
            f"train_nll={avg_nll:.4f}, train_kl={avg_kl:.4f}, beta={beta:.4f}"
        )

        if val_loader is not None and epoch % validate_every == 0:
            val_metrics = evaluate_e3_cvae(model, val_loader, beta=beta, device=device, free_bits=free_bits)
            history["val_total"].append(val_metrics["total"])
            history["val_nll"].append(val_metrics["nll"])
            history["val_kl"].append(val_metrics["kl"])
            history["val_diagnostics"].append(val_metrics)
            print(
                f"Validation: val_total={val_metrics['total']:.4f}, "
                f"val_nll={val_metrics['nll']:.4f}, val_kl={val_metrics['kl']:.4f}, "
                f"rmse={val_metrics['rmse']:.4f}, "
                f"logsig_para={val_metrics['log_sigma_para_mean']:.3f}, "
                f"logsig_perp={val_metrics['log_sigma_perp_mean']:.3f}"
            )

            if early_stopping and epoch > warmup_epochs:
                improved = best_val_loss - val_metrics["total"] > min_delta
                if improved:
                    best_val_loss = val_metrics["total"]
                    history["best_val_loss"] = best_val_loss
                    history["best_epoch"] = epoch
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
                            save_path,
                        )
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        print(
                            f"Early stopping at epoch {epoch}; "
                            f"best epoch was {history['best_epoch']}."
                        )
                        break
        elif save_path is not None and not early_stopping:
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict(), "beta": beta},
                save_path,
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    return history


@torch.no_grad()
def evaluate_e3_cvae(model, loader, beta=1.0, device="cpu", free_bits=0.0):
    model.eval()
    isotropic = model.isotropic
    loss_fn = e3_cvae_isotropic_loss if isotropic else e3_cvae_axial_loss

    total_loss = total_nll = total_kl = 0.0
    se_sum = 0.0
    y2_sum = 0.0
    mu2_sum = 0.0
    sample2_sum = 0.0
    vector_count = 0
    node_count = 0
    z_count = 0
    z_mu2_sum = 0.0
    z_logvar_sum = 0.0
    log_sigma_col0_sum = 0.0
    log_sigma_col1_sum = 0.0
    sigma_col0_sum = 0.0
    sigma_col1_sum = 0.0

    for batch in loader:
        batch = move_graph_batch_to_device(batch, device)
        outputs = model(batch)
        loss, nll, kl = loss_fn(outputs, batch, beta=beta, free_bits=free_bits)
        total_loss += loss.item()
        total_nll += nll.item()
        total_kl += kl.item()

        y = batch["r_next"]
        mu = outputs["mu"]
        log_sigma = outputs["log_sigma"]
        sample, _, _ = model.sample_torch(batch)

        se_sum += (mu - y).pow(2).sum().item()
        y2_sum += y.pow(2).sum().item()
        mu2_sum += mu.pow(2).sum().item()
        sample2_sum += sample.pow(2).sum().item()
        vector_count += y.numel()

        node_count += log_sigma.shape[0]
        log_sigma_col0_sum += log_sigma[:, 0].sum().item()
        sigma_col0_sum += torch.exp(log_sigma[:, 0]).sum().item()
        if not isotropic:
            log_sigma_col1_sum += log_sigma[:, 1].sum().item()
            sigma_col1_sum += torch.exp(log_sigma[:, 1]).sum().item()

        z_mu = outputs["z_mu"]
        z_logvar = outputs["z_logvar"]
        z_count += z_mu.numel()
        z_mu2_sum += z_mu.pow(2).sum().item()
        z_logvar_sum += z_logvar.sum().item()

    n_batches = len(loader)
    metrics = {
        "total": total_loss / n_batches,
        "nll": total_nll / n_batches,
        "kl": total_kl / n_batches,
        "rmse": (se_sum / max(vector_count, 1)) ** 0.5,
        "target_rms": (y2_sum / max(vector_count, 1)) ** 0.5,
        "mu_rms": (mu2_sum / max(vector_count, 1)) ** 0.5,
        "sample_rms": (sample2_sum / max(vector_count, 1)) ** 0.5,
        "z_mu_rms": (z_mu2_sum / max(z_count, 1)) ** 0.5,
        "z_logvar_mean": z_logvar_sum / max(z_count, 1),
    }
    if isotropic:
        metrics["log_sigma_mean"] = log_sigma_col0_sum / max(node_count, 1)
        metrics["sigma_mean"] = sigma_col0_sum / max(node_count, 1)
        # Keep para/perp keys with same value for backward compat with logging
        metrics["log_sigma_para_mean"] = metrics["log_sigma_mean"]
        metrics["log_sigma_perp_mean"] = metrics["log_sigma_mean"]
    else:
        metrics["log_sigma_para_mean"] = log_sigma_col0_sum / max(node_count, 1)
        metrics["log_sigma_perp_mean"] = log_sigma_col1_sum / max(node_count, 1)
        metrics["sigma_para_mean"] = sigma_col0_sum / max(node_count, 1)
        metrics["sigma_perp_mean"] = sigma_col1_sum / max(node_count, 1)
    return metrics
