# Ultimate Tic-Tac-Toe

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

## Installation

```bash
# Python 3.10+ recommandé ( faire attention avec les compatibilités torch des python trop récents)
pip install numpy torch matplotlib
# Tkinter est surement inclus avec la votre distribution Python (sinon : apt install python3-tk)
```

Placez les checkpoints (`.pth`) attendus dans `models/` :

```
models/
├── best_uttt_model.pth        # réseau supervisé (UTTTNet 10×256)
└── alphazero/
    ├── best.pth               # meilleur réseau self-play (UTTTNetLight 4×128)
    └── iter_XXXX.pth          # snapshots périodiques
```

---

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
python run_training.py                # entraînement AlphaZero (~2h on peut pousser sur plus)
python run_training.py --resume       # reprend depuis models/alphazero/best.pth
```

## Entraînement AlphaZero

La boucle (`self_play.py`) alterne trois phases :
1. **Self-play** — le réseau joue contre lui-même via MCTS → remplit le `ReplayBuffer`
2. **Entraînement** — `Trainer` optimise sur les triplets `(état, π, z)`
3. **Évaluation** — le challenger affronte le champion ; promu si win rate ≥ 55 %



## 1. Interface graphique

Lance la fenêtre Tkinter à 4 onglets — **Jouer**, **Match**, **Entraînement**, **Debug**.

```bash
python ui.py
```

- **Jouer** — humain contre AlphaBeta ou MCTS, choix du camp (X/J1 ou O/J2), profondeur/simulations réglables, sélection du checkpoint via `…`.
- **Match** — deux bots s'affrontent sur N parties avec plateau live, barre de progression et log.
- **Entraînement** — courbes loss et win-rate lues depuis les logs de self-play.
- **Debug** — heatmap des scores par coup, mode pas-à-pas, auto-play, retour arrière.

---

## 2. Tournois en ligne de commande — `arena.py`

Trois sous-commandes :

```bash
# AlphaBeta vs Random (baseline de sanité)
python arena.py benchmark --games 20 --depth 3

# AlphaBeta vs MCTS (comparaison des deux moteurs)
python arena.py battle --games 10 --depth 3 --simulations 200 \
    --checkpoint1 models/best_uttt_model.pth \
    --checkpoint2 models/alphazero/best.pth

# Batterie de tests qualité du modèle (symétrie, dispersion, policy, mini-bench)
python arena.py eval --games 20 --depth 3
```

Options communes : `--checkpoint1`, `--device {cpu,cuda}`, `--seed`, `--verbose`.

---

## 3. Entraînement AlphaZero — self-play

### Lancement rapide (préréglages RTX 3070 (ma carte graphique), ~30 itérations / ~2 h)

```bash
python run_training.py
python run_training.py --resume              # reprend depuis models/alphazero/best.pth
python run_training.py --iterations 60       # surcharge ponctuelle
```

### Boucle complète paramétrable — `self_play.py`

```bash
python self_play.py \
    --simulations 200 \
    --games-per-iter 50 \
    --iterations 200 \
    --train-steps 200 \
    --batch-size 256 \
    --eval-games 20 --eval-every 5 --win-threshold 0.55 \
    --checkpoint-dir models/alphazero
```

Options principales : `--buffer-size`, `--lr`, `--weight-decay`, `--num-filters`, `--num-res-blocks`, `--temp-cutoff`, `--device`, `--resume CHEMIN.pth`. `python self_play.py --help` pour la liste complète.

La boucle alterne self-play → entraînement → évaluation challenger vs champion. Un checkpoint est écrit à chaque itération ; le challenger n'est promu que s'il dépasse le seuil de victoires.

---

## 4. Rapports et statistiques (cette partie était notamment pour générer les statistiques pour la soutenance et pour le déboggage)

### Rapport de présentation — `rapport.py`

Produit un fichier texte structuré + un JSON de données brutes à partir de tous les tests (parties, évaluateur, moteur de recherche, temps).

```bash
python rapport.py --no-eval                                    # sans checkpoint (factice)
python rapport.py --checkpoint models/best_uttt_model.pth      # UTTTNet 10×256
python rapport.py --checkpoint models/alphazero/best.pth --light   # UTTTNetLight
python rapport.py --checkpoint models/best.pth --games 50 --depth 4 --compare-depths
```

Sorties : `rapport_presentation.txt`, `rapport_data.json`. Options de saut : `--skip-duel`, `--skip-parties`, `--skip-eval`, `--skip-search`, `--skip-temps`.

### Tableaux de bord visuels — `graphiques.py`

Génère 3 figures PNG (12 graphiques) : analyse du jeu, moteur de recherche, qualité du modèle.

```bash
python graphiques.py
python graphiques.py --checkpoint models/best_uttt_model.pth
python graphiques.py --checkpoint models/alphazero/best.pth --light
python graphiques.py --games 300 --positions 60 --seed 7
```

Sorties par défaut : `stats_jeu.png`, `stats_alphabeta.png`, `stats_modele.png` (renommables via `--out-jeu`, `--out-ab`, `--out-modele`).

### Comparaison de tous les checkpoints — `model_stats.py`

Compare automatiquement tous les `.pth` d'un dossier (auto-détection d'architecture), produit un CSV récapitulatif, un round-robin, une matrice de win-rates et un classement Elo approximatif.

```bash
python model_stats.py
python model_stats.py --models-dir models/alphazero --games 20
python model_stats.py --include models/best_uttt_model.pth --games 10 --no-round-robin
```

Options : `--max-models`, `--rr-games`, `--outdir stats_output`, `--device`.

### Vérification qualité du réseau — `quality_checker.py`

Teste la cohérence des scores (+1 gagnant, -1 perdant, ~0 nul) sur des parties réelles et des positions synthétiques.

```bash
python quality_checker.py
python quality_checker.py --checkpoint models/alphazero/best.pth
python quality_checker.py --games 50 --states 40 --continuous --interval 120 \
    --output models/alphazero/quality.json
```

Le mode `--continuous` ré-évalue automatiquement toutes les N secondes — pratique pour surveiller un entraînement en cours.

---

## 5. Outils de diagnostic

### `test_depth.py` — les scores varient-ils avec la profondeur ?

Vérifie que l'alpha-bêta explore réellement l'arbre (sinon : bug dans `search.py` ou dans l'évaluateur).

```bash
python test_depth.py
python test_depth.py --checkpoint models/best_uttt_model.pth
```

### `test_symmetry.py` — le réseau distingue-t-il les deux joueurs ?

Si `v(état, J1) ≈ -v(état, J2)`, le réseau est symétrique. Si `v(J1) ≈ v(J2)`, le canal d'entrée 19 (joueur courant) est ignoré — signe d'un bug d'entraînement.

```bash
python test_symmetry.py --checkpoint models/alphazero/iter_0003.pth
python test_symmetry.py --checkpoint models/best_uttt_model.pth
```

---


## Notes pratiques

- CUDA est détecté automatiquement. Forcez le CPU avec `--device cpu` si besoin.
- Les checkpoints AlphaZero (UTTTNetLight) et supervisés (UTTTNet) sont **incompatibles entre évaluateurs** : utilisez `NeuralEvaluator` pour le second, `LightEvaluator` pour le premier. `model_stats.py` et `quality_checker.py` auto-détectent l'architecture.
- Un `Ctrl+C` pendant l'entraînement préserve le dernier checkpoint écrit sur disque.
- `run_stats.py` est vide dans le dépôt — il peut être supprimé sans impact.
