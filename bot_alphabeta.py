"""
bot_alphabeta.py — Agent alpha-bêta + iterative deepening
==========================================================
Encapsule NeuralEvaluator (UTTTNet 10×256) et la recherche alpha-bêta
définie dans search.py.

Usage
─────
  from bot_alphabeta import AlphaBetaAgent
  from model         import NeuralEvaluator
  from arena         import Arena
  from bot_random    import RandomAgent

  ev    = NeuralEvaluator("models/best_uttt_model.pth")
  agent = AlphaBetaAgent(ev, depth=4)

  # Coup isolé
  move, score = agent.choose_move(state)

  # Tournoi
  report = Arena(agent, RandomAgent()).run_verbose(n_games=20)
  report.print_summary()

Paramètres
──────────
  evaluator : NeuralEvaluator   — value head + policy head
  depth     : int               — profondeur maximale (iterative deepening)
  top_k     : int | None        — coups explorés par nœud (None = tous)
"""

from typing import Optional, Tuple

from game   import UTTTState
from model  import NeuralEvaluator
from search import best_move as _alphabeta
from arena  import Agent


class AlphaBetaAgent(Agent):
    """
    Bot alpha-bêta avec iterative deepening, table de transposition Zobrist
    et move ordering par policy head neuronal.

    Réseau : UTTTNet (10 blocs résiduels, 256 filtres)
    Stratégie : alpha-bêta  (voir search.py)
    """

    def __init__(
        self,
        evaluator: NeuralEvaluator,
        depth: int            = 4,
        top_k: Optional[int] = None,
    ):
        self.evaluator = evaluator
        self.depth     = depth
        self.top_k     = top_k
        self.name      = f"AlphaBeta(d={depth})"

    def choose_move(self, state: UTTTState) -> Tuple[int, float]:
        return _alphabeta(
            state,
            self.evaluator,
            depth=self.depth,
            top_k=self.top_k,
            verbose=False,
        )
