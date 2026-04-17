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
    # Self-play — avec batch MCTS on peut se permettre plus de simulations
    # pour le même temps : 100 sims séquentiels ≈ 300 sims batch_size=8
    simulations     = 300,      # augmenté grâce au batch inference
    games_per_iter  = 25,
    temp_cutoff     = 10,

    # Buffer
    buffer_size     = 50_000,
    min_buffer      = 500,

    # Entraînement
    train_steps     = 100,
    batch_size      = 256,
    lr              = 1e-3,
    weight_decay    = 1e-4,

    # Évaluation
    eval_games      = 16,
    eval_sims       = 100,      # sims réduits à l'éval
    win_threshold   = 0.55,
    eval_every      = 3,

    # Réseau
    num_filters     = 128,
    num_res_blocks  = 4,

    # Durée — même temps qu'avant mais meilleure qualité
    iterations      = 30,

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
