# Ultimate Tic-Tac-Toe — IA Python

IA jouant à l'Ultimate Tic-Tac-Toe (UTTT) via un réseau neuronal dual-head (AlphaZero-style).

## Structure

```
game.py           — Logique pure du jeu (UTTTState, coups légaux, règles)
model.py          — Architecture ResNet (UTTTNet, 10×256) + NeuralEvaluator
search.py         — Alpha-bêta avec iterative deepening, Zobrist, table de transposition
bot_alphabeta.py  — Agent alpha-bêta (utilise NeuralEvaluator + search.py)
bot_mcts.py       — Agent MCTS PUCT + réseau léger UTTTNetLight (4×128)
bot_random.py     — Agent aléatoire (baseline)
arena.py          — Moteur de tournoi : fait s'affronter deux agents
replay_buffer.py  — Buffer circulaire de triplets (état, π, z) pour l'entraînement MCTS
trainer.py        — Optimisation du réseau (loss value + loss policy) pour le MCTS
self_play.py      — Boucle AlphaZero : self-play → entraînement → évaluation pour le MCTS
run_training.py   — Script de lancement calibré RTX 3060/3070 (~2h)
```

## Agents disponibles

| Agent | Fichier | Description |
|---|---|---|
| `AlphaBetaAgent` | `bot_alphabeta.py` | Alpha-bêta + policy head neuronal |
| `MCTSAgent` | `bot_mcts.py` | MCTS PUCT guidé par réseau léger |
| `RandomAgent` | `bot_random.py` | Baseline uniforme |

## Utilisation rapide

```python
from model         import NeuralEvaluator
from bot_alphabeta import AlphaBetaAgent
from bot_random    import RandomAgent
from arena         import Arena

ev     = NeuralEvaluator("models/best_uttt_model.pth")
report = Arena(AlphaBetaAgent(ev, depth=3), RandomAgent()).run_verbose(n_games=20)
report.print_summary()
```

```bash
# CLI
python arena.py benchmark --games 20 --depth 3
python arena.py battle    --games 10 --simulations 200
python run_training.py                # entraînement AlphaZero (~2h)
python run_training.py --resume       # reprend depuis models/alphazero/best.pth
```

## Entraînement AlphaZero

La boucle (`self_play.py`) alterne trois phases :
1. **Self-play** — le réseau joue contre lui-même via MCTS → remplit le `ReplayBuffer`
2. **Entraînement** — `Trainer` optimise sur les triplets `(état, π, z)`
3. **Évaluation** — le challenger affronte le champion ; promu si win rate ≥ 55 %
