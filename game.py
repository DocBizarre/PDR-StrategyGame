"""
game.py — Logique pure du jeu Ultimate Tic-Tac-Toe
====================================================
Aucune dépendance ML. Seul numpy est requis.
"""

import numpy as np
from typing import List

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────────────────────────────────────

WIN_LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),
    (0, 3, 6), (1, 4, 7), (2, 5, 8),
    (0, 4, 8), (2, 4, 6),
]


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def check_winner_small(cells: np.ndarray) -> int:
    """
    Vérifie si une grille 3×3 (9 cases) est gagnée.
    Retourne 1, 2 ou 0 (pas encore gagné).
    """
    for a, b, c in WIN_LINES:
        if cells[a] != 0 and cells[a] == cells[b] == cells[c]:
            return int(cells[a])
    return 0


def decode_state_string(s: str):
    """
    Parse la string d'état (93 chars, format dataset markstanl/u3t).

    s[0:81]  → plateau principal  (0=vide, 1=joueur1, 2=joueur2)
    s[81:90] → meta_board (état des 9 sous-grilles)
    s[90]    → active board index (0–8, ou autre si toutes libres)
    s[91]    → current player (1 ou 2)
    s[92]    → profondeur (ignorée)

    Retourne (board, meta_board, active_idx, current_player).
    """
    n = len(s)
    assert n >= 81, f"String trop courte : {n} chars"

    board      = np.array([int(c) for c in s[0:81]], dtype=np.int8)
    meta_board = np.array([int(c) for c in s[81:90]], dtype=np.int8) if n >= 90 else np.zeros(9, dtype=np.int8)

    active_idx = -1
    if n >= 91:
        raw = int(s[90])
        active_idx = raw if 0 <= raw <= 8 else -1

    current_player = None
    if n >= 92:
        cp = int(s[91])
        current_player = cp if cp in (1, 2) else None

    if current_player not in (1, 2):
        n_p1 = int((board == 1).sum())
        n_p2 = int((board == 2).sum())
        current_player = 1 if n_p1 == n_p2 else 2

    return board, meta_board, active_idx, current_player


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAT DU JEU
# ─────────────────────────────────────────────────────────────────────────────

class UTTTState:
    """
    État complet d'une partie Ultimate Tic-Tac-Toe.

    Attributs
    ---------
    board      : np.ndarray (81,)  cases 0/1/2
    meta_board : np.ndarray (9,)   état des sous-grilles (0=libre, 1/2=gagné, 3=nul)
    active_idx : int               sous-grille active (-1 = toutes libres)
    player     : int               joueur courant (1 ou 2)
    """

    __slots__ = ("board", "meta_board", "active_idx", "player", "_winner")

    def __init__(self):
        self.board      = np.zeros(81, dtype=np.int8)
        self.meta_board = np.zeros(9,  dtype=np.int8)
        self.active_idx = -1
        self.player     = 1
        self._winner    = 0

    # ── Construction ──────────────────────────────────────────────────────

    @classmethod
    def from_string(cls, s: str) -> "UTTTState":
        """Construit un état depuis une string du dataset (93 chars)."""
        state = cls()
        board, meta_board, active_idx, current_player = decode_state_string(s)
        state.board      = board
        state.meta_board = meta_board
        state.active_idx = active_idx
        state.player     = current_player
        state._winner    = check_winner_small(meta_board)
        return state

    @classmethod
    def initial(cls) -> "UTTTState":
        """Plateau vide, joueur 1 commence."""
        return cls()

    # ── Sérialisation ─────────────────────────────────────────────────────

    def to_string(self) -> str:
        """Sérialise l'état en string 93 chars (format dataset)."""
        board_str  = "".join(str(int(c)) for c in self.board)
        meta_str   = "".join(str(int(c)) for c in self.meta_board)
        active_str = str(self.active_idx if self.active_idx != -1 else 9)
        return board_str + meta_str + active_str + str(self.player) + "0"

    # ── Coups légaux ──────────────────────────────────────────────────────

    def legal_moves(self) -> List[int]:
        """
        Retourne les indices globaux (0–80) des coups légaux.
        Un coup est légal si la case est vide ET dans une sous-grille active
        ET cette sous-grille n'est pas encore terminée.
        """
        if self._winner != 0:
            return []

        sub_range = range(9) if self.active_idx == -1 else [self.active_idx]
        moves = []
        for sub in sub_range:
            if self.meta_board[sub] != 0:
                continue
            base = sub * 9
            for cell in range(9):
                if self.board[base + cell] == 0:
                    moves.append(base + cell)
        return moves

    # ── Application d'un coup ─────────────────────────────────────────────

    def apply_move(self, move: int) -> "UTTTState":
        """
        Retourne un NOUVEL état après le coup (index global 0–80).
        L'état courant n'est pas modifié.
        """
        new            = UTTTState.__new__(UTTTState)
        new.board      = self.board.copy()
        new.meta_board = self.meta_board.copy()

        sub  = move // 9
        cell = move % 9

        new.board[move] = self.player

        # Mise à jour meta_board pour cette sous-grille
        sub_cells  = new.board[sub * 9: sub * 9 + 9]
        sub_winner = check_winner_small(sub_cells)
        if sub_winner != 0:
            new.meta_board[sub] = sub_winner
        elif np.all(sub_cells != 0):
            new.meta_board[sub] = 3  # nulle

        # Prochaine sous-grille active
        new.active_idx = cell if new.meta_board[cell] == 0 else -1

        new._winner = check_winner_small(new.meta_board)
        new.player  = 3 - self.player  # alterne 1 ↔ 2

        return new

    # ── Propriétés ────────────────────────────────────────────────────────

    @property
    def is_terminal(self) -> bool:
        return self._winner != 0 or len(self.legal_moves()) == 0

    @property
    def winner(self) -> int:
        """0 = en cours ou nul, 1 ou 2 = vainqueur."""
        return self._winner

    def __repr__(self) -> str:
        return (f"UTTTState(player={self.player}, "
                f"active={self.active_idx}, winner={self._winner})")
