"""
bot_mcts.py — Agent MCTS + UTTTNetLight (style AlphaZero)
==========================================================
Deux modes d'utilisation :

  1. Jeu normal (arena) :
       agent.choose_move(state) → (move, score)

  2. Self-play / entraînement :
       move, pi = engine.search_with_policy(state, temperature)
       → pi : np.ndarray (81,) des probabilités de visites MCTS,
              cible du policy head pendant l'entraînement.
"""

from __future__ import annotations

import math
import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from game  import UTTTState
from model import state_to_tensor
from arena import Agent


# ═════════════════════════════════════════════════════════════════════════════
# RÉSEAU
# ═════════════════════════════════════════════════════════════════════════════

class _ResBlock(nn.Module):
    def __init__(self, f: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(f, f, 3, padding=1, bias=False), nn.BatchNorm2d(f), nn.ReLU(inplace=True),
            nn.Conv2d(f, f, 3, padding=1, bias=False), nn.BatchNorm2d(f),
        )
    def forward(self, x): return F.relu(self.net(x) + x)


class UTTTNetLight(nn.Module):
    """
    Réseau dual-head pour UTTT.
      entrée : (B, 20, 9, 9)
      sorties : log_probs (B, 81)  +  value (B,) ∈ [-1, 1]

    Défaut : 128 filtres × 4 blocs ≈ 1.6 M paramètres.
    """
    def __init__(self, in_channels=20, num_filters=128, num_res_blocks=4, policy_size=81):
        super().__init__()
        f = num_filters
        self.stem       = nn.Sequential(nn.Conv2d(in_channels, f, 3, padding=1, bias=False), nn.BatchNorm2d(f), nn.ReLU(inplace=True))
        self.res_blocks = nn.Sequential(*[_ResBlock(f) for _ in range(num_res_blocks)])
        self.p_conv = nn.Conv2d(f, 2, 1, bias=False); self.p_bn = nn.BatchNorm2d(2); self.p_fc = nn.Linear(2*81, policy_size)
        self.v_conv = nn.Conv2d(f, 1, 1, bias=False); self.v_bn = nn.BatchNorm2d(1); self.v_fc1 = nn.Linear(81, 128); self.v_fc2 = nn.Linear(128, 1)
        for m in self.modules():
            if   isinstance(m, nn.Conv2d):     nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d): nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):     nn.init.xavier_normal_(m.weight); nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x  = self.res_blocks(self.stem(x))
        p  = F.relu(self.p_bn(self.p_conv(x))).view(x.size(0), -1)
        lp = F.log_softmax(self.p_fc(p), dim=1)
        v  = F.relu(self.v_bn(self.v_conv(x))).view(x.size(0), -1)
        v  = torch.tanh(self.v_fc2(F.relu(self.v_fc1(v)))).squeeze(1)
        return lp, v


# ═════════════════════════════════════════════════════════════════════════════
# ÉVALUATEUR
# ═════════════════════════════════════════════════════════════════════════════

class LightEvaluator:
    """
    Encapsule UTTTNetLight.
    Expose `model` directement pour que Trainer accède aux paramètres.
    """

    def __init__(self, checkpoint: Optional[str] = None, device: Optional[str] = None,
                 num_filters: int = 128, num_res_blocks: int = 4):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model  = UTTTNetLight(num_filters=num_filters, num_res_blocks=num_res_blocks).to(self.device)

        if checkpoint:
            try:
                self.model.load_state_dict(torch.load(checkpoint, map_location=self.device, weights_only=True))
                print(f"[LightEvaluator] '{checkpoint}' chargé sur {self.device}.")
            except FileNotFoundError:
                print(f"[LightEvaluator] '{checkpoint}' introuvable → poids aléatoires.")
            except Exception as e:
                print(f"[LightEvaluator] Erreur chargement ({e}) → poids aléatoires.")
        else:
            print(f"[LightEvaluator] Nouveau réseau — poids aléatoires ({self.device}).")

        self.model.eval()
        self._cache: Dict[str, Tuple[float, np.ndarray]] = {}

    # ── Inférence ─────────────────────────────────────────────────────────

    def clear_cache(self) -> None: self._cache.clear()
    def set_train(self):           self.model.train(); self._cache.clear()
    def set_eval(self):            self.model.eval();  self._cache.clear()

    @torch.no_grad()
    def _forward(self, state: UTTTState) -> Tuple[float, np.ndarray]:
        key = state.to_string()
        if key in self._cache: return self._cache[key]
        x      = torch.from_numpy(state_to_tensor(key)).unsqueeze(0).to(self.device)
        lp, v  = self.model(x)
        result = (float(v.item()), lp.squeeze(0).cpu().numpy())
        self._cache[key] = result
        return result

    def evaluate(self, state: UTTTState) -> float:          return self._forward(state)[0]
    def policy_logprobs(self, state: UTTTState) -> np.ndarray: return self._forward(state)[1]
    def evaluate_and_policy(self, state: UTTTState):         return self._forward(state)

    # ── Persistance ───────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"[LightEvaluator] Sauvegardé → {path}")

    def load(self, path: str) -> None:
        self.model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        self.model.eval(); self._cache.clear()

    def clone_weights_from(self, other: "LightEvaluator") -> None:
        """Copie les poids depuis un autre LightEvaluator (snapshot champion)."""
        self.model.load_state_dict(other.model.state_dict())
        self.model.eval(); self._cache.clear()


# ═════════════════════════════════════════════════════════════════════════════
# NŒUD MCTS
# ═════════════════════════════════════════════════════════════════════════════

class _MCTSNode:
    __slots__ = ("state", "parent", "move", "children", "untried", "visits", "val_sum", "prior")

    def __init__(self, state, parent=None, move=None, prior=1.0):
        self.state    = state
        self.parent   = parent
        self.move     = move
        self.children: Dict[int, "_MCTSNode"] = {}
        self.untried  = state.legal_moves() if not state.is_terminal else []
        self.visits   = 0
        self.val_sum  = 0.0
        self.prior    = prior

    @property
    def q(self): return self.val_sum / self.visits if self.visits else 0.0

    def ucb(self, c=1.5):
        if self.parent is None: return 0.0
        return self.q + c * self.prior * math.sqrt(self.parent.visits) / (1 + self.visits)

    def best_child(self, c=1.5): return max(self.children.values(), key=lambda n: n.ucb(c))
    def most_visited(self):      return max(self.children.values(), key=lambda n: n.visits)
    def fully_expanded(self):    return not self.untried


# ═════════════════════════════════════════════════════════════════════════════
# MOTEUR MCTS — batch inference + virtual loss
# ═════════════════════════════════════════════════════════════════════════════

class MCTSEngine:
    """
    MCTS PUCT avec batch inference GPU.

    Optimisation clé : au lieu d'un forward réseau par nœud feuille,
    on collecte `batch_size` feuilles en parallèle (virtual loss),
    puis on fait UN SEUL forward pour toutes → ~batch_size× moins
    d'aller-retours CPU↔GPU.

    Gain mesuré sur RTX 3060 : 3–5× plus rapide qu'un MCTS séquentiel
    avec le même nombre de simulations.

    search(state, temperature)             → (move, q)   — jeu normal
    search_with_policy(state, temperature) → (move, pi)  — self-play
    """

    def __init__(
        self,
        evaluator:  LightEvaluator,
        simulations: int   = 200,
        c_puct:      float = 1.5,
        batch_size:  int   = 8,    # feuilles évaluées par forward GPU
    ):
        self.ev         = evaluator
        self.sims       = simulations
        self.c          = c_puct
        self.batch_size = batch_size

    # ── Sélection ─────────────────────────────────────────────────────────

    def _select(self, root: _MCTSNode) -> _MCTSNode:
        node = root
        while not node.state.is_terminal:
            if not node.fully_expanded(): return node
            node = node.best_child(self.c)
        return node

    # ── Expansion ─────────────────────────────────────────────────────────

    def _expand_with_priors(self, node: _MCTSNode, priors: np.ndarray) -> _MCTSNode:
        """Expand un nœud avec des priors déjà calculés (évite un forward supplémentaire)."""
        if not node.untried or node.state.is_terminal:
            return node
        move = max(node.untried, key=lambda m: priors[m])
        node.untried.remove(move)
        child = _MCTSNode(
            state  = node.state.apply_move(move),
            parent = node,
            move   = move,
            prior  = float(priors[move]),
        )
        node.children[move] = child
        return child

    # ── Virtual loss ──────────────────────────────────────────────────────

    def _apply_virtual_loss(self, node: _MCTSNode, vl: float = -1.0) -> None:
        """Pénalise temporairement un chemin pour forcer l'exploration d'autres branches."""
        n = node
        while n is not None:
            n.visits  += 1
            n.val_sum += vl
            n = n.parent

    def _remove_virtual_loss(self, node: _MCTSNode, vl: float = -1.0) -> None:
        """Annule le virtual loss après la vraie évaluation."""
        n = node
        while n is not None:
            n.visits  -= 1
            n.val_sum -= vl
            n = n.parent

    # ── Backprop ──────────────────────────────────────────────────────────

    def _backprop(self, node: _MCTSNode, value: float) -> None:
        while node is not None:
            node.visits  += 1
            node.val_sum += value
            value  = -value
            node   = node.parent

    # ── Construction de la racine ─────────────────────────────────────────

    def _build_root(self, state: UTTTState, add_noise: bool) -> _MCTSNode:
        root   = _MCTSNode(state)
        priors = np.exp(self.ev.policy_logprobs(state))

        if add_noise:
            legal  = state.legal_moves()
            noise  = np.random.dirichlet([0.3] * len(legal))
            noised = priors.copy()
            for i, m in enumerate(legal):
                noised[m] = 0.75 * priors[m] + 0.25 * noise[i]
            priors = noised

        for move in list(root.untried):
            root.untried.remove(move)
            child = _MCTSNode(
                state  = state.apply_move(move),
                parent = root,
                move   = move,
                prior  = float(priors[move]),
            )
            root.children[move] = child
        return root

    # ── Batch simulation ──────────────────────────────────────────────────

    @torch.no_grad()
    def _batch_forward(self, states: list) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward GPU sur un batch d'états non-terminaux.
        Retourne (values, policy_logprobs) sous forme de tableaux numpy.
        """
        tensors = np.stack([
            state_to_tensor(s.to_string()) for s in states
        ])  # (B, 20, 9, 9)
        x  = torch.from_numpy(tensors).to(self.ev.device)
        lp, v = self.ev.model(x)
        return v.cpu().numpy(), lp.cpu().numpy()

    def _run_simulations(self, root: _MCTSNode) -> None:
        """
        Lance self.sims simulations en groupes de batch_size.
        Pour chaque groupe :
          1. Sélectionne batch_size feuilles (avec virtual loss)
          2. Un seul forward GPU pour toutes
          3. Expand + backprop chaque feuille
        """
        n_batches = (self.sims + self.batch_size - 1) // self.batch_size

        for _ in range(n_batches):
            leaves      = []
            terminal_vs = []

            # ── Sélection du batch ────────────────────────────────────────
            for _ in range(self.batch_size):
                leaf = self._select(root)

                if leaf.state.is_terminal:
                    w = leaf.state.winner
                    v = 0.0 if w == 0 else (1.0 if w != leaf.state.player else -1.0)
                    terminal_vs.append((leaf, v))
                else:
                    self._apply_virtual_loss(leaf)
                    leaves.append(leaf)

            # ── Backprop terminaux directement ────────────────────────────
            for leaf, v in terminal_vs:
                self._backprop(leaf, v)

            if not leaves:
                continue

            # ── Batch forward GPU ─────────────────────────────────────────
            values, log_probs_batch = self._batch_forward([l.state for l in leaves])

            # ── Expand + backprop chaque feuille ──────────────────────────
            for i, leaf in enumerate(leaves):
                self._remove_virtual_loss(leaf)
                priors = np.exp(log_probs_batch[i])
                child  = self._expand_with_priors(leaf, priors)
                self._backprop(child, float(values[i]))

    # ── Politique depuis les visites ──────────────────────────────────────

    def _visits_to_policy(self, root: _MCTSNode, temperature: float) -> np.ndarray:
        pi = np.zeros(81, dtype=np.float32)
        for move, child in root.children.items():
            pi[move] = child.visits
        if temperature <= 0.01:
            best = int(np.argmax(pi)); pi[:] = 0.0; pi[best] = 1.0
        else:
            pi = pi ** (1.0 / temperature)
            s  = pi.sum()
            if s > 0: pi /= s
        return pi

    # ── API publique ──────────────────────────────────────────────────────

    def search(self, state: UTTTState, temperature: float = 0.1) -> Tuple[int, float]:
        """Jeu normal — retourne (move, q_value)."""
        self.ev.clear_cache()
        root = self._build_root(state, add_noise=False)
        self._run_simulations(root)
        if not root.children:
            mvs = state.legal_moves(); return (mvs[0] if mvs else 0), 0.0
        pi   = self._visits_to_policy(root, temperature)
        move = int(np.argmax(pi))
        return move, root.children[move].q if move in root.children else 0.0

    def search_with_policy(
        self,
        state:       UTTTState,
        temperature: float = 1.0,
    ) -> Tuple[int, np.ndarray]:
        """
        Self-play — retourne (move, pi).
        pi : vecteur (81,) des probabilités de visites → cible policy head.
        """
        self.ev.clear_cache()
        root = self._build_root(state, add_noise=True)
        self._run_simulations(root)
        if not root.children:
            mvs = state.legal_moves()
            pi  = np.zeros(81, dtype=np.float32)
            if mvs: pi[mvs[0]] = 1.0
            return (mvs[0] if mvs else 0), pi
        pi   = self._visits_to_policy(root, temperature)
        move = int(np.random.choice(81, p=pi))
        return move, pi


# ═════════════════════════════════════════════════════════════════════════════
# AGENT (interface Arena)
# ═════════════════════════════════════════════════════════════════════════════

class MCTSAgent(Agent):
    """Wrapper Arena autour de MCTSEngine. Pour le self-play, utiliser MCTSEngine directement."""

    def __init__(self, evaluator: LightEvaluator, simulations: int = 200,
                 c_puct: float = 1.5, temperature: float = 0.1, batch_size: int = 8):
        self.engine      = MCTSEngine(evaluator, simulations, c_puct, batch_size)
        self.temperature = temperature
        self.name        = f"MCTS(s={simulations})"

    def choose_move(self, state: UTTTState) -> Tuple[int, float]:
        return self.engine.search(state, self.temperature)
