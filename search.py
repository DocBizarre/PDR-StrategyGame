"""
search.py — Alpha-bêta avec Zobrist, table de transposition et iterative deepening
====================================================================================
Améliorations par rapport à la version précédente :
  1. Iterative deepening (ID) : on cherche depth=1, 2, …, N en réutilisant
     la TT entre chaque passe → move ordering progressivement meilleur,
     tt_hits enfin non nuls.
  2. top_k=None par défaut : tous les coups légaux sont explorés.
     Le policy head sert uniquement au move ordering, sans troncature.
  3. La TT n'est plus vidée entre les passes ID (seulement entre les coups).

API publique inchangée :
  best_move(state, evaluator, depth, top_k) → (move, score)
"""

import numpy as np
from typing import Tuple, List, Optional

from game  import UTTTState, check_winner_small
from model import NeuralEvaluator


# ─────────────────────────────────────────────────────────────────────────────
# TABLES DE ZOBRIST
# ─────────────────────────────────────────────────────────────────────────────

_RNG = np.random.RandomState(seed=0xDEADBEEF)

# board[case][joueur] — case 0..80, joueur 0=J1, 1=J2
ZOBRIST_BOARD: np.ndarray = _RNG.randint(
    0, 2**63, size=(81, 2), dtype=np.uint64
)

# meta[sous-grille][état] — état 0=J1 gagné, 1=J2 gagné, 2=nul
ZOBRIST_META: np.ndarray = _RNG.randint(
    0, 2**63, size=(9, 3), dtype=np.uint64
)

# active[idx] — idx 0..8 pour une sous-grille spécifique, 9 pour "toutes"
ZOBRIST_ACTIVE: np.ndarray = _RNG.randint(
    0, 2**63, size=(10,), dtype=np.uint64
)

# XOR avec cette valeur pour basculer le joueur courant
ZOBRIST_SIDE: int = int(_RNG.randint(0, 2**63, dtype=np.uint64))


def zobrist_full(state: UTTTState) -> int:
    """Calcule le hash Zobrist complet d'un état (utilisé une seule fois à la racine)."""
    h = np.uint64(0)

    for i in range(81):
        v = state.board[i]
        if v == 1:
            h ^= ZOBRIST_BOARD[i, 0]
        elif v == 2:
            h ^= ZOBRIST_BOARD[i, 1]

    for sg in range(9):
        m = state.meta_board[sg]
        if m == 1:
            h ^= ZOBRIST_META[sg, 0]
        elif m == 2:
            h ^= ZOBRIST_META[sg, 1]
        elif m == 3:
            h ^= ZOBRIST_META[sg, 2]

    active_key = state.active_idx if state.active_idx != -1 else 9
    h ^= ZOBRIST_ACTIVE[active_key]

    if state.player == 2:
        h ^= np.uint64(ZOBRIST_SIDE)

    return int(h)


def zobrist_update(
    h: int,
    move: int,
    player: int,
    old_meta_sg: int,
    new_meta_sg: int,
    sg: int,
    old_active: int,
    new_active: int,
) -> int:
    """Met à jour le hash Zobrist de façon incrémentale après un coup — O(1)."""
    zh = np.uint64(h)

    zh ^= ZOBRIST_BOARD[move, player - 1]

    if old_meta_sg != new_meta_sg:
        if old_meta_sg in (1, 2, 3):
            zh ^= ZOBRIST_META[sg, old_meta_sg - 1]
        if new_meta_sg in (1, 2, 3):
            zh ^= ZOBRIST_META[sg, new_meta_sg - 1]

    old_ak = old_active if old_active != -1 else 9
    new_ak = new_active if new_active != -1 else 9
    if old_ak != new_ak:
        zh ^= ZOBRIST_ACTIVE[old_ak]
        zh ^= ZOBRIST_ACTIVE[new_ak]

    zh ^= np.uint64(ZOBRIST_SIDE)

    return int(zh)


# ─────────────────────────────────────────────────────────────────────────────
# TABLE DE TRANSPOSITION
# ─────────────────────────────────────────────────────────────────────────────

EXACT       = 0
LOWER_BOUND = 1   # fail-high : score >= beta
UPPER_BOUND = 2   # fail-low  : score <= alpha


class TranspositionTable:
    """Table de transposition à remplacement par profondeur."""

    __slots__ = ("_table", "_max_size")

    def __init__(self, max_size: int = 1 << 20):
        self._table: dict[int, tuple[int, int, float, Optional[int]]] = {}
        self._max_size = max_size

    def probe(self, key: int, depth: int, alpha: float, beta: float
              ) -> Tuple[Optional[float], Optional[int]]:
        """
        Retourne (score, best_move) si exploitable,
        (None, best_move) si seul le best_move est utile pour le move ordering,
        (None, None) sinon.
        """
        entry = self._table.get(key)
        if entry is None:
            return None, None

        tt_depth, tt_flag, tt_score, tt_move = entry

        if tt_depth < depth:
            return None, tt_move   # entrée trop peu profonde mais best_move utile

        if tt_flag == EXACT:
            return tt_score, tt_move
        elif tt_flag == LOWER_BOUND and tt_score >= beta:
            return tt_score, tt_move
        elif tt_flag == UPPER_BOUND and tt_score <= alpha:
            return tt_score, tt_move

        return None, tt_move

    def store(self, key: int, depth: int, flag: int, score: float,
              best_move: Optional[int]) -> None:
        """Stocke un résultat. Ne remplace pas une entrée plus profonde."""
        existing = self._table.get(key)
        if existing is not None and existing[0] > depth:
            return

        if len(self._table) >= self._max_size and key not in self._table:
            items = sorted(self._table.items(), key=lambda x: x[1][0])
            for k, _ in items[: self._max_size // 4]:
                del self._table[k]

        self._table[key] = (depth, flag, score, best_move)

    def clear(self) -> None:
        self._table.clear()

    def __len__(self) -> int:
        return len(self._table)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def terminal_value(state: UTTTState, maximizing_player: int) -> float:
    w = state.winner
    if w == 0:
        return 0.0
    return 1.0 if w == maximizing_player else -1.0


def _order_moves(
    state: UTTTState,
    evaluator: NeuralEvaluator,
    top_k: Optional[int],
    tt_move: Optional[int] = None,
) -> List[int]:
    """
    Ordonne les coups pour maximiser les coupures alpha-bêta.

    Priorité :
      1. Coup TT (meilleur coup de la recherche précédente)
      2. Tri par policy head (score neuronal)
      3. Troncature top_k si spécifié (None = tous les coups)
    """
    moves = state.legal_moves()

    if len(moves) <= 1:
        return moves

    log_probs = evaluator.policy_logprobs(state)
    moves = sorted(moves, key=lambda m: log_probs[m], reverse=True)

    if top_k is not None and len(moves) > top_k:
        moves = moves[:top_k]

    # Le coup TT en tête, même s'il n'est pas dans le top-K
    if tt_move is not None:
        if tt_move in moves:
            moves.remove(tt_move)
            moves.insert(0, tt_move)
        elif top_k is None:
            pass   # tt_move déjà dans moves (trié)

    return moves


# ─────────────────────────────────────────────────────────────────────────────
# ALPHA-BÊTA
# ─────────────────────────────────────────────────────────────────────────────

def _alphabeta(
    state: UTTTState,
    zobrist_hash: int,
    depth: int,
    alpha: float,
    beta: float,
    maximizing: bool,
    maximizing_player: int,
    evaluator: NeuralEvaluator,
    tt: TranspositionTable,
    top_k: Optional[int],
    stats: dict,
) -> float:
    """Alpha-bêta avec TT et hachage Zobrist incrémental."""

    if state.is_terminal:
        stats["terminals"] += 1
        return terminal_value(state, maximizing_player)

    if depth == 0:
        stats["evals"] += 1
        raw_value = evaluator.evaluate(state)
        return raw_value if state.player == maximizing_player else -raw_value

    tt_score, tt_move = tt.probe(zobrist_hash, depth, alpha, beta)
    if tt_score is not None:
        stats["tt_hits"] += 1
        return tt_score

    moves = _order_moves(state, evaluator, top_k, tt_move)
    stats["nodes"] += 1

    best_mv      = moves[0]
    original_alpha = alpha

    if maximizing:
        value = -float("inf")
        for move in moves:
            child = state.apply_move(move)
            sg    = move // 9
            child_hash = zobrist_update(
                h=zobrist_hash, move=move, player=state.player,
                old_meta_sg=int(state.meta_board[sg]),
                new_meta_sg=int(child.meta_board[sg]),
                sg=sg,
                old_active=state.active_idx,
                new_active=child.active_idx,
            )
            score = _alphabeta(
                child, child_hash, depth - 1, alpha, beta,
                False, maximizing_player, evaluator, tt, top_k, stats,
            )
            if score > value:
                value   = score
                best_mv = move
            if value > alpha:
                alpha = value
            if alpha >= beta:
                stats["cutoffs"] += 1
                break
    else:
        value = float("inf")
        for move in moves:
            child = state.apply_move(move)
            sg    = move // 9
            child_hash = zobrist_update(
                h=zobrist_hash, move=move, player=state.player,
                old_meta_sg=int(state.meta_board[sg]),
                new_meta_sg=int(child.meta_board[sg]),
                sg=sg,
                old_active=state.active_idx,
                new_active=child.active_idx,
            )
            score = _alphabeta(
                child, child_hash, depth - 1, alpha, beta,
                True, maximizing_player, evaluator, tt, top_k, stats,
            )
            if score < value:
                value   = score
                best_mv = move
            if value < beta:
                beta = value
            if alpha >= beta:
                stats["cutoffs"] += 1
                break

    # Stocker dans la TT
    if value <= original_alpha:
        flag = UPPER_BOUND
    elif value >= beta:
        flag = LOWER_BOUND
    else:
        flag = EXACT

    tt.store(zobrist_hash, depth, flag, value, best_mv)
    return value


# ─────────────────────────────────────────────────────────────────────────────
# ITERATIVE DEEPENING
# ─────────────────────────────────────────────────────────────────────────────

def _iterative_deepening(
    state: UTTTState,
    root_hash: int,
    max_depth: int,
    evaluator: NeuralEvaluator,
    tt: TranspositionTable,
    top_k: Optional[int],
    stats: dict,
    verbose: bool,
) -> Tuple[int, float]:
    """
    Recherche par approfondissement itératif de depth=1 jusqu'à max_depth.

    À chaque passe :
      - La TT est conservée (les entrées des passes précédentes guident
        le move ordering des passes suivantes).
      - Le meilleur coup trouvé à depth=d est utilisé en tête à depth=d+1.

    Retourne (best_move, best_score) à la profondeur maximale.
    """
    maximizing_player = state.player
    best_mv    = state.legal_moves()[0]
    best_score = -float("inf")

    # Ordonnancement initial par policy head (avant toute recherche)
    log_probs = evaluator.policy_logprobs(state)
    root_moves = sorted(state.legal_moves(), key=lambda m: log_probs[m], reverse=True)
    if top_k is not None and len(root_moves) > top_k:
        root_moves = root_moves[:top_k]

    for depth in range(1, max_depth + 1):
        depth_stats = {"nodes": 0, "evals": 0, "terminals": 0,
                       "tt_hits": 0, "cutoffs": 0}

        current_best_mv    = root_moves[0]
        current_best_score = -float("inf")
        alpha = -float("inf")
        beta  =  float("inf")

        # Placer le meilleur coup de la passe précédente en tête
        ordered_moves = list(root_moves)
        if best_mv in ordered_moves:
            ordered_moves.remove(best_mv)
            ordered_moves.insert(0, best_mv)

        for move in ordered_moves:
            child = state.apply_move(move)
            sg    = move // 9
            child_hash = zobrist_update(
                h=root_hash, move=move, player=state.player,
                old_meta_sg=int(state.meta_board[sg]),
                new_meta_sg=int(child.meta_board[sg]),
                sg=sg,
                old_active=state.active_idx,
                new_active=child.active_idx,
            )
            score = _alphabeta(
                child, child_hash, depth - 1, alpha, beta,
                False, maximizing_player, evaluator, tt, top_k, depth_stats,
            )
            if score > current_best_score:
                current_best_score = score
                current_best_mv    = move
            if current_best_score > alpha:
                alpha = current_best_score

        best_mv    = current_best_mv
        best_score = current_best_score

        # Accumuler les stats globales
        for k in stats:
            stats[k] += depth_stats[k]

        if verbose:
            print(
                f"  [ID depth={depth:2d}]  "
                f"best={best_mv}  score={best_score:+.3f}  "
                f"nodes={depth_stats['nodes']}  evals={depth_stats['evals']}  "
                f"tt_hits={depth_stats['tt_hits']}  "
                f"cutoffs={depth_stats['cutoffs']}  "
                f"tt_size={len(tt)}"
            )

    return best_mv, best_score


# ─────────────────────────────────────────────────────────────────────────────
# INTERFACE PUBLIQUE
# ─────────────────────────────────────────────────────────────────────────────

# TT globale — conservée entre les passes ID, vidée entre les coups
_global_tt = TranspositionTable(max_size=1 << 20)


def best_move(
    state: UTTTState,
    evaluator: NeuralEvaluator,
    depth: int = 4,
    top_k: Optional[int] = None,   # None = tous les coups (recommandé avec ID)
    verbose: bool = True,
    **kwargs,
) -> Tuple[int, float]:
    """
    Calcule le meilleur coup par iterative deepening alpha-bêta
    avec hachage Zobrist et table de transposition.

    Paramètres
    ----------
    state     : état courant du jeu
    evaluator : évaluateur neuronal (value + policy heads)
    depth     : profondeur maximale de recherche
    top_k     : coups explorés par nœud (None = tous — recommandé)
                Mettre une valeur entière pour limiter la largeur.
    verbose   : afficher les statistiques par passe ID

    Retourne
    --------
    (best_move_index, best_score)
    """
    evaluator.clear_cache()
    _global_tt.clear()

    moves = state.legal_moves()
    if not moves:
        raise ValueError("Aucun coup légal disponible.")
    if len(moves) == 1:
        return moves[0], 0.0

    root_hash = zobrist_full(state)
    stats = {"nodes": 0, "evals": 0, "terminals": 0, "tt_hits": 0, "cutoffs": 0}

    best_mv, best_score = _iterative_deepening(
        state, root_hash, depth, evaluator, _global_tt, top_k, stats, verbose,
    )

    if verbose:
        print(
            f"  [Search total]  depth={depth}  top_k={top_k}  "
            f"nodes={stats['nodes']}  evals={stats['evals']}  "
            f"terminals={stats['terminals']}  "
            f"tt_hits={stats['tt_hits']}  cutoffs={stats['cutoffs']}  "
            f"tt_size={len(_global_tt)}"
        )

    return best_mv, best_score


# ─────────────────────────────────────────────────────────────────────────────
# ALIAS DE COMPATIBILITÉ
# ─────────────────────────────────────────────────────────────────────────────

def alphabeta(
    state: UTTTState,
    evaluator: NeuralEvaluator,
    depth: int = 4,
    top_k: Optional[int] = None,
    **kwargs,
) -> Tuple[int, float]:
    """Alias de best_move pour compatibilité arrière."""
    return best_move(state, evaluator, depth=depth, top_k=top_k, verbose=False, **kwargs)
