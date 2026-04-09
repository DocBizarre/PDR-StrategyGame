"""
run_training.py — Entraînement AlphaZero calibré RTX 3060/3070, ~1-2h
======================================================================
Estimation :
  - ~100 sims/coup  × 25 parties  ≈ 3-4 min de self-play / iter
  - 100 steps d'entraînement      ≈ 10 sec / iter
  - évaluation tous les 3 iter    ≈ 1 min tous les 12 min
  → ~30 itérations en 2h

Résultat attendu :
  - Itérations 1-5  : le réseau sort du hasard (loss_p descend)
  - Itérations 5-15 : il apprend des patterns basiques
  - Itérations 15-30: win rate vs Random commence à monter
  - Pour battre AlphaBeta(depth=3) : compter 200+ itérations (nuit/weekend)

Usage :
  python run_training.py           # lance avec les paramètres ci-dessous
  python run_training.py --resume  # reprend depuis models/alphazero/best.pth
"""

import argparse
import random
import numpy as np
import torch

from self_play import run_alphazero

# ── Paramètres ────────────────────────────────────────────────────────────────

CFG = dict(
    # Self-play
    simulations     = 100,      # sims MCTS/coup  — 100 = bon compromis vitesse/qualité
    games_per_iter  = 25,       # parties/iter    — 25 × ~40 coups = ~1000 exemples/iter
    temp_cutoff     = 10,       # exploration pendant les 10 premiers coups

    # Buffer
    buffer_size     = 50_000,   # ~50 itérations d'historique
    min_buffer      = 500,      # démarre l'entraînement dès 500 exemples

    # Entraînement
    train_steps     = 100,      # steps/iter
    batch_size      = 256,
    lr              = 1e-3,
    weight_decay    = 1e-4,

    # Évaluation
    eval_games      = 16,       # 16 parties = assez pour estimer le WR
    eval_sims       = 50,       # sims réduits pendant l'éval pour aller plus vite
    win_threshold   = 0.55,     # 55% pour promouvoir le challenger
    eval_every      = 3,        # évalue toutes les 3 itérations

    # Réseau (léger pour aller vite)
    num_filters     = 128,
    num_res_blocks  = 4,

    # Durée
    iterations      = 30,       # ~2h sur RTX 3060/3070

    # Persistance
    checkpoint_dir  = "models/alphazero",
    device          = "cuda",
)

# ── Lancement ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true",
                        help="Reprend depuis models/alphazero/best.pth")
    parser.add_argument("--iterations", type=int, default=None,
                        help="Surcharge le nombre d'itérations")
    parser.add_argument("--simulations", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if args.resume:
        CFG["resume"] = "models/alphazero/best.pth"
    if args.iterations:
        CFG["iterations"] = args.iterations
    if args.simulations:
        CFG["simulations"] = args.simulations
    if args.device:
        CFG["device"] = args.device

    # Reproductibilité
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        print(f"GPU détecté : {torch.cuda.get_device_name(0)}")
        print(f"VRAM       : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠  CUDA non disponible — entraînement sur CPU (très lent)")
        CFG["device"] = "cpu"

    print(f"\nEstimation : ~{CFG['iterations'] * 4} min "
          f"({CFG['iterations']} itérations × ~4 min/iter)")
    print("Ctrl+C pour interrompre proprement — le dernier checkpoint est sauvegardé.\n")

    run_alphazero(**CFG)
