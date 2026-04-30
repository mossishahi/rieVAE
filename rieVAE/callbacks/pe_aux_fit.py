"""PEAuxFitCallback.

Fits the post-hoc PE auxiliary head A_psi on FROZEN encoder outputs at
``on_train_end``, then computes ``delta_pe_aux_sup = sup_i ||A_psi(mu_i)
- Psi(x_i)||`` on the full active sample. Replaces the pre-Phase-3
trainer's ``_fit_pe_aux_posthoc``.

Only fires when the model has ``use_pe=True``; otherwise no-op.
"""
from __future__ import annotations

from typing import Optional

import torch

try:
    import pytorch_lightning as pl
    _PL_AVAILABLE = True
except ImportError:
    _PL_AVAILABLE = False
    pl = None  # type: ignore[assignment]


if _PL_AVAILABLE:

    class PEAuxFitCallback(pl.Callback):
        """Post-hoc PE auxiliary head fit (Cor.~cor:pe_euclidean)."""

        def __init__(
            self,
            n_steps: int = 2000,
            lr: float = 1e-3,
            batch_size: int = 512,
        ) -> None:
            super().__init__()
            self.n_steps = int(n_steps)
            self.lr = float(lr)
            self.batch_size = int(batch_size)
            self.delta_pe_aux_sup: Optional[float] = None

        def on_train_end(self, trainer, pl_module):
            model = pl_module.model
            if not getattr(model, "use_pe", False):
                return
            if not hasattr(model, "aux_pe_head") or model.aux_pe_head is None:
                return
            dm = getattr(trainer, "datamodule", None)
            if dm is None or getattr(dm, "artefacts", None) is None:
                return
            artefacts = dm.artefacts
            if artefacts.pe_feat is None:
                return

            device = next(model.parameters()).device
            x_active = artefacts.x_active.to(device)
            pe_feat = artefacts.pe_feat.to(device, x_active.dtype)
            n_active = int(artefacts.n_active)

            # Step 1: cache frozen mu (encoder run once).
            model.eval()
            with torch.no_grad():
                mu_frozen, _ = model.encode_nodes(
                    x_active, pe_feat=pe_feat, alpha_pe=1.0,
                )
            mu_frozen = mu_frozen.detach()

            # Step 2: freeze everything except aux_pe_head.
            for name, p in model.named_parameters():
                p.requires_grad_("aux_pe_head" in name)

            opt = torch.optim.Adam(
                [p for p in model.aux_pe_head.parameters()],
                lr=self.lr,
            )

            model.train()
            batch_sz = self.batch_size if self.batch_size > 0 else n_active
            batch_sz = min(batch_sz, n_active)
            for _ in range(self.n_steps):
                if batch_sz < n_active:
                    idx = torch.randperm(n_active, device=device)[:batch_sz]
                    mu_b = mu_frozen[idx]
                    pe_b = pe_feat[idx].detach()
                else:
                    mu_b = mu_frozen
                    pe_b = pe_feat.detach()
                pe_hat = model.aux_pe_head(mu_b)
                loss = (pe_hat - pe_b).pow(2).sum(dim=-1).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

            for p in model.parameters():
                p.requires_grad_(True)

            # Step 3: certified delta_pe_aux_sup in eval mode.
            model.eval()
            with torch.no_grad():
                pe_hat_all = model.aux_pe_head(mu_frozen)
                residuals = (pe_hat_all - pe_feat.to(pe_hat_all.dtype)).norm(dim=-1)
                self.delta_pe_aux_sup = float(residuals.max().item())
            model.train()
            print(
                f"[pe_posthoc] fitted A_psi in {self.n_steps} steps "
                f"(lr={self.lr})  delta_pe_aux_sup = "
                f"{self.delta_pe_aux_sup:.4f}",
                flush=True,
            )

else:  # pragma: no cover

    class PEAuxFitCallback:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError(
                "rieVAE.callbacks.PEAuxFitCallback requires "
                "pytorch_lightning."
            )
