"""Perdida y exactitudes acumuladas sobre una particion completa."""
import torch
import torch.nn.functional as F


@torch.no_grad()
def evaluate(model, loader, device="cpu"):
    """Modo evaluacion, sin gradientes y con teacher forcing.

    Acumula sumas y cuentas sobre TODA la particion (no promedia lotes).
    El silencio (0) es una categoria valida: no hay ignore_index.
    """
    model.eval()
    loss_sum, notes_ok, steps_ok, n_notes, n_steps = 0.0, 0, 0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x, y)                                   # (B,16,4,128)
        v = logits.size(-1)
        loss_sum += F.cross_entropy(logits.reshape(-1, v), y.reshape(-1),
                                    reduction="sum").item()
        hit = logits.argmax(-1) == y                           # (B,16,4)
        notes_ok += hit.sum().item()
        steps_ok += hit.all(-1).sum().item()
        n_notes += hit.numel()
        n_steps += hit.size(0) * hit.size(1)
    return {"loss": loss_sum / n_notes,
            "note_acc": notes_ok / n_notes,
            "step_acc": steps_ok / n_steps}
