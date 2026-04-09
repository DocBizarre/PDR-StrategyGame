"""
replay_buffer.py — Buffer de replay pour l'entraînement AlphaZero
==================================================================
Stocke les triplets (état, π, z) produits par le self-play.

  état : tenseur (20, 9, 9) encodé depuis UTTTState
  π    : distribution de visites MCTS (81,) — cible policy head
  z    : résultat final de la partie depuis la perspective du joueur
         qui jouait dans cet état (+1 victoire, -1 défaite, 0 nul)

Usage
─────
  buf = ReplayBuffer(max_size=50_000)

  # Remplissage (depuis self_play.py)
  buf.push(state_tensor, pi, z)

  # Entraînement (depuis trainer.py)
  states, pis, zs = buf.sample(batch_size=256)
"""

from __future__ import annotations

import random
from collections import deque
from typing import Tuple

import numpy as np


class ReplayBuffer:
    """
    Buffer circulaire FIFO.
    Quand max_size est atteint, les exemples les plus anciens sont effacés.
    """

    def __init__(self, max_size: int = 100_000):
        self._buffer: deque = deque(maxlen=max_size)
        self.max_size = max_size

    # ── Alimentation ──────────────────────────────────────────────────────

    def push(
        self,
        state: np.ndarray,   # (20, 9, 9)  float32
        pi:    np.ndarray,   # (81,)       float32
        z:     float,        # +1 / -1 / 0
    ) -> None:
        """Ajoute un exemple au buffer."""
        self._buffer.append((state.copy(), pi.copy(), float(z)))

    def push_game(self, examples: list) -> None:
        """
        Ajoute tous les exemples d'une partie d'un coup.
        examples : liste de (state_tensor, pi, z)
        """
        for state, pi, z in examples:
            self.push(state, pi, z)

    # ── Échantillonnage ───────────────────────────────────────────────────

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Tire batch_size exemples uniformément sans remise.
        Retourne (states, pis, zs) sous forme de tableaux numpy.

        states : (B, 20, 9, 9)
        pis    : (B, 81)
        zs     : (B,)
        """
        batch_size = min(batch_size, len(self._buffer))
        batch      = random.sample(self._buffer, batch_size)
        states, pis, zs = zip(*batch)
        return (
            np.stack(states).astype(np.float32),
            np.stack(pis).astype(np.float32),
            np.array(zs,     dtype=np.float32),
        )

    # ── Utilitaires ───────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._buffer)

    @property
    def is_ready(self) -> bool:
        """True si le buffer contient assez d'exemples pour entraîner."""
        return len(self._buffer) >= 512

    def clear(self) -> None:
        self._buffer.clear()

    def __repr__(self) -> str:
        return f"ReplayBuffer({len(self._buffer)}/{self.max_size})"
