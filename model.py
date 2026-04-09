"""
model.py — Architecture ResNet dual-head + évaluateur neuronal optimisé
========================================================================
Optimisations par rapport à la version de base :
  1. Cache de transposition  : évite de réévaluer le même état deux fois
  2. Inférence combinée      : value + policy en un seul forward au lieu de deux
  3. torch.compile           : fusion des opérateurs CUDA (~2x plus rapide)
  4. Warmup automatique      : élimine la latence du premier appel

Dépendances : torch, numpy
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from game import UTTTState, decode_state_string


# ─────────────────────────────────────────────────────────────────────────────
# ENCODAGE ÉTAT → TENSEUR
# ─────────────────────────────────────────────────────────────────────────────

def state_to_tensor(s: str) -> np.ndarray:
    board, meta_board, active_idx, current_player = decode_state_string(s)
    tensor = np.zeros((20, 9, 9), dtype=np.float32)

    for sub in range(9):
        r_sub, c_sub = divmod(sub, 3)
        for cell in range(9):
            r_cell, c_cell = divmod(cell, 3)
            val = board[sub * 9 + cell]
            row = r_sub * 3 + r_cell
            col = c_sub * 3 + c_cell
            if val == 1:
                tensor[sub, row, col] = 1.0
            elif val == 2:
                tensor[sub + 9, row, col] = 1.0

    if active_idx == -1:
        tensor[18] = 1.0
    else:
        r_sub, c_sub = divmod(active_idx, 3)
        for cell in range(9):
            r_cell, c_cell = divmod(cell, 3)
            tensor[18, r_sub * 3 + r_cell, c_sub * 3 + c_cell] = 1.0

    tensor[19] = float(current_player == 1)
    return tensor


# ─────────────────────────────────────────────────────────────────────────────
# ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    def __init__(self, num_filters: int):
        super().__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(num_filters)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class UTTTNet(nn.Module):
    def __init__(
        self,
        in_channels: int    = 20,
        num_filters: int    = 256,
        num_res_blocks: int = 10,
        policy_size: int    = 81,
    ):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, num_filters, 3, padding=1, bias=False),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace=True),
        )
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(num_filters) for _ in range(num_res_blocks)]
        )
        self.policy_conv = nn.Conv2d(num_filters, 2, 1, bias=False)
        self.policy_bn   = nn.BatchNorm2d(2)
        self.policy_fc   = nn.Linear(2 * 9 * 9, policy_size)

        self.value_conv  = nn.Conv2d(num_filters, 1, 1, bias=False)
        self.value_bn    = nn.BatchNorm2d(1)
        self.value_fc1   = nn.Linear(9 * 9, 256)
        self.value_fc2   = nn.Linear(256, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.res_blocks(x)

        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        policy_logprob = F.log_softmax(self.policy_fc(p), dim=1)

        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v)).squeeze(1)

        return policy_logprob, value


# ─────────────────────────────────────────────────────────────────────────────
# ÉVALUATEUR NEURONAL OPTIMISÉ
# ─────────────────────────────────────────────────────────────────────────────

class NeuralEvaluator:
    """
    Encapsule UTTTNet avec trois optimisations majeures :

    1. Cache de transposition
       Chaque état est identifié par sa string (93 chars).
       Si le même état est rencontré deux fois dans l'arbre alpha-bêta,
       le réseau n'est appelé qu'une seule fois.
       Le cache est vidé entre chaque coup (clear_cache()).

    2. Inférence combinée (evaluate_and_policy)
       Un seul forward retourne value ET policy.
       Évite de doubler les appels réseau quand les deux sont nécessaires.

    3. torch.compile
       Fusionne les opérateurs CUDA pour réduire le temps de forward.
       Warmup automatique au chargement pour absorber le coût de compilation.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[str] = None,
        num_filters: int    = 256,
        num_res_blocks: int = 10,
        use_compile: bool   = False,
        cache_size: int     = 50_000,
    ):
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        raw = UTTTNet(
            in_channels=20,
            num_filters=num_filters,
            num_res_blocks=num_res_blocks,
            policy_size=81,
        ).to(self.device)
        raw.load_state_dict(
            torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        )
        raw.eval()

        # torch.compile : fusionne les kernels CUDA (~2x speedup sur GPU)
        if use_compile and self.device.type == "cuda":
            try:
                self.model = torch.compile(raw, mode="reduce-overhead")
                print("[NeuralEvaluator] torch.compile activé (reduce-overhead)")
            except Exception:
                self.model = raw
                print("[NeuralEvaluator] torch.compile non disponible, mode standard")
        else:
            self.model = raw

        # Cache de transposition : string d'état → (value, policy_logprobs)
        self._cache: dict = {}
        self._cache_size  = cache_size
        self._hits        = 0
        self._misses      = 0

        print(f"[NeuralEvaluator] Modèle chargé depuis '{checkpoint_path}' sur {self.device}")
        self._warmup()

    def _warmup(self, n: int = 5):
        """Lance n forwards à vide pour déclencher la compilation JIT."""
        dummy = torch.zeros(1, 20, 9, 9, device=self.device)
        with torch.no_grad():
            for _ in range(n):
                self.model(dummy)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        print(f"[NeuralEvaluator] Warmup terminé ({n} forwards)")

    # ── Cache ─────────────────────────────────────────────────────────────

    def clear_cache(self):
        """Vide le cache entre chaque coup pour éviter une croissance infinie."""
        self._cache.clear()
        self._hits   = 0
        self._misses = 0

    def cache_stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "hits":     self._hits,
            "misses":   self._misses,
            "total":    total,
            "hit_rate": self._hits / total if total > 0 else 0.0,
            "size":     len(self._cache),
        }

    # ── Inférence interne ─────────────────────────────────────────────────

    @torch.no_grad()
    def _forward(self, state_str: str) -> Tuple[float, np.ndarray]:
        """Forward avec cache. Retourne (value, policy_logprobs)."""
        if state_str in self._cache:
            self._hits += 1
            return self._cache[state_str]

        self._misses += 1
        tensor = state_to_tensor(state_str)
        x = torch.from_numpy(tensor).unsqueeze(0).to(self.device)
        policy_lp, value = self.model(x)

        result = (
            float(value.item()),
            policy_lp.squeeze(0).cpu().numpy(),
        )

        # Évite une croissance illimitée
        if len(self._cache) >= self._cache_size:
            keys = list(self._cache.keys())
            for k in keys[: self._cache_size // 2]:
                del self._cache[k]

        self._cache[state_str] = result
        return result

    # ── API publique ──────────────────────────────────────────────────────

    def evaluate(self, state: UTTTState) -> float:
        """Valeur ∈ [-1, 1] depuis la perspective du joueur courant."""
        value, _ = self._forward(state.to_string())
        return value

    def policy_logprobs(self, state: UTTTState) -> np.ndarray:
        """Log-probabilités (81,) du policy head."""
        _, logprobs = self._forward(state.to_string())
        return logprobs

    def evaluate_and_policy(self, state: UTTTState) -> Tuple[float, np.ndarray]:
        """Retourne (value, policy_logprobs) en un seul forward."""
        return self._forward(state.to_string())
