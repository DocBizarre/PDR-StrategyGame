"""
trainer.py — Entraînement AlphaZero pour UTTTNetLight
======================================================
Prend des batches du ReplayBuffer et optimise le réseau sur deux pertes :
  loss_value  = MSE(v_prédit, z_réel)
  loss_policy = CrossEntropy(π_prédit, π_MCTS)
  loss_total  = loss_value + loss_policy

Usage
─────
  from bot_mcts     import LightEvaluator
  from replay_buffer import ReplayBuffer
  from trainer       import Trainer

  ev      = LightEvaluator()
  buf     = ReplayBuffer()
  trainer = Trainer(ev, buf)

  # Après chaque partie de self-play :
  metrics = trainer.train_step(batch_size=256)
  print(metrics)   # {"loss": 1.23, "loss_v": 0.45, "loss_p": 0.78}
"""

from __future__ import annotations

import time
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from bot_mcts      import LightEvaluator
from replay_buffer import ReplayBuffer


class Trainer:
    """
    Optimise UTTTNetLight à partir des exemples du ReplayBuffer.

    Hyperparamètres
    ───────────────
      lr           : taux d'apprentissage initial (défaut : 1e-3)
      weight_decay : régularisation L2 (défaut : 1e-4)
      batch_size   : exemples par step (défaut : 256)
      grad_clip    : norme max du gradient (défaut : 1.0)
    """

    def __init__(
        self,
        evaluator:    LightEvaluator,
        buffer:       ReplayBuffer,
        lr:           float = 1e-3,
        weight_decay: float = 1e-4,
        grad_clip:    float = 1.0,
        lr_min:       float = 1e-5,
        total_steps:  int   = 100_000,
    ):
        self.ev         = evaluator
        self.buf        = buffer
        self.device     = evaluator.device
        self.grad_clip  = grad_clip

        self.optimizer = Adam(
            evaluator.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=total_steps, eta_min=lr_min
        )

        self._step     = 0
        self._history: list = []   # métriques par step

    # ── Step d'entraînement ───────────────────────────────────────────────

    def train_step(self, batch_size: int = 256) -> Dict[str, float]:
        """
        Tire un batch, fait un forward + backward + step.
        Retourne les métriques du step.
        """
        if not self.buf.is_ready:
            return {}

        self.ev.set_train()
        states_np, pis_np, zs_np = self.buf.sample(batch_size)

        states = torch.from_numpy(states_np).to(self.device)   # (B, 20, 9, 9)
        pis    = torch.from_numpy(pis_np).to(self.device)      # (B, 81)
        zs     = torch.from_numpy(zs_np).to(self.device)       # (B,)

        # Forward
        log_probs, values = self.ev.model(states)   # (B, 81), (B,)

        # Pertes
        loss_v = F.mse_loss(values, zs)
        loss_p = -(pis * log_probs).sum(dim=1).mean()   # cross-entropy avec cible souple
        loss   = loss_v + loss_p

        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.ev.model.parameters(), self.grad_clip)
        self.optimizer.step()
        self.scheduler.step()

        self._step += 1
        self.ev.set_eval()

        metrics = {
            "step":   self._step,
            "loss":   float(loss.item()),
            "loss_v": float(loss_v.item()),
            "loss_p": float(loss_p.item()),
            "lr":     self.scheduler.get_last_lr()[0],
        }
        self._history.append(metrics)
        return metrics

    def train_epoch(
        self,
        n_steps:    int = 100,
        batch_size: int = 256,
        log_every:  int = 10,
    ) -> Dict[str, float]:
        """
        Lance n_steps steps consécutifs.
        Retourne les métriques moyennées sur l'epoch.
        """
        losses, losses_v, losses_p = [], [], []
        t0 = time.time()

        for i in range(1, n_steps + 1):
            m = self.train_step(batch_size)
            if not m: continue
            losses.append(m["loss"]); losses_v.append(m["loss_v"]); losses_p.append(m["loss_p"])
            if log_every and i % log_every == 0:
                print(f"    step {self._step:6d}  "
                      f"loss={m['loss']:.4f}  "
                      f"v={m['loss_v']:.4f}  "
                      f"p={m['loss_p']:.4f}  "
                      f"lr={m['lr']:.2e}")

        elapsed = time.time() - t0
        summary = {
            "loss":    float(np.mean(losses))   if losses else 0.0,
            "loss_v":  float(np.mean(losses_v)) if losses_v else 0.0,
            "loss_p":  float(np.mean(losses_p)) if losses_p else 0.0,
            "elapsed": elapsed,
            "steps":   len(losses),
        }
        return summary

    # ── Utilitaires ───────────────────────────────────────────────────────

    @property
    def step(self) -> int:
        return self._step

    @property
    def history(self) -> list:
        return self._history

    def last_metrics(self) -> Optional[Dict]:
        return self._history[-1] if self._history else None
