"""
bot_random.py — Agent aléatoire (baseline)
==========================================
Choisit un coup légal au hasard avec distribution uniforme.
Sert de baseline pour évaluer la qualité des autres agents.

Usage
─────
  from bot_random import RandomAgent
  from arena      import Arena

  agent = RandomAgent()
  move, score = agent.choose_move(state)   # score toujours 0.0

  # Exemple : benchmark d'un agent contre Random
  report = Arena(mon_agent, RandomAgent()).run_verbose(n_games=20)
"""

import random
from typing import Tuple

from game  import UTTTState
from arena import Agent


class RandomAgent(Agent):
    """Joue un coup légal uniformément au hasard. Score toujours 0.0."""

    name = "Random"

    def choose_move(self, state: UTTTState) -> Tuple[int, float]:
        return random.choice(state.legal_moves()), 0.0
