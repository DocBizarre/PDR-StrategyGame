"""
self_play.py — Boucle d'entraînement AlphaZero pour UTTT
=========================================================
Répète indéfiniment :
  1. Self-play   — le réseau joue contre lui-même via MCTS
                   → remplit le ReplayBuffer de triplets (état, π, z)
  2. Entraînement — optimise le réseau sur un epoch du buffer
  3. Évaluation   — le nouveau réseau affronte le champion actuel
                   si win_rate > seuil → il devient le nouveau champion

Usage
─────
  python self_play.py
  python self_play.py --simulations 200 --games-per-iter 50 --iterations 100
  python self_play.py --checkpoint models/alphazero_best.pth --resume
"""

from __future__ import annotations

import argparse
import copy
import os
import random
import time
from typing import List, Tuple

import numpy as np
import torch

from game          import UTTTState
from model         import state_to_tensor
from bot_mcts      import LightEvaluator, MCTSEngine, MCTSAgent
from replay_buffer import ReplayBuffer
from trainer       import Trainer
from arena         import Arena


# ═════════════════════════════════════════════════════════════════════════════
# GÉNÉRATION DE PARTIES (self-play)
# ═════════════════════════════════════════════════════════════════════════════

def play_self_play_game(
    engine:      MCTSEngine,
    temp_cutoff: int   = 15,    # coups avant de passer en mode greedy
    temp_high:   float = 1.0,   # température exploration (début)
    temp_low:    float = 0.1,   # température exploitation (fin)
) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    """
    Joue une partie complète en self-play.

    Pour chaque coup :
      - MCTS produit (move, pi) avec search_with_policy()
      - L'état est encodé en tenseur (20, 9, 9)
      - Le résultat z est assigné en fin de partie

    Retourne une liste de (state_tensor, pi, z) prêts pour le ReplayBuffer.
    """
    state    = UTTTState.initial()
    examples = []   # (state_tensor, pi, player_who_played)
    move_num = 0

    while not state.is_terminal:
        move_num += 1
        temp = temp_high if move_num <= temp_cutoff else temp_low

        # MCTS → politique de visites
        move, pi = engine.search_with_policy(state, temperature=temp)

        # Encode l'état courant
        tensor = state_to_tensor(state.to_string())

        examples.append((tensor, pi, state.player))
        state = state.apply_move(move)

    # Résultat final
    winner = state.winner   # 0=nul, 1=J1, 2=J2

    # Assigne z depuis la perspective de chaque joueur
    result = []
    for tensor, pi, player in examples:
        if winner == 0:
            z = 0.0
        elif winner == player:
            z = 1.0
        else:
            z = -1.0
        result.append((tensor, pi, z))

    return result


# ═════════════════════════════════════════════════════════════════════════════
# ÉVALUATION CHAMPION vs CHALLENGER
# ═════════════════════════════════════════════════════════════════════════════

def evaluate_challenger(
    champion:    LightEvaluator,
    challenger:  LightEvaluator,
    simulations: int   = 100,
    n_games:     int   = 20,
    win_threshold: float = 0.55,
) -> Tuple[bool, float]:
    """
    Fait s'affronter champion vs challenger via l'Arena.
    Retourne (challenger_wins, win_rate_challenger).
    """
    champ_agent      = MCTSAgent(champion,   simulations=simulations, temperature=0.1)
    challenger_agent = MCTSAgent(challenger, simulations=simulations, temperature=0.1)

    report = Arena(challenger_agent, champ_agent).run(n_games=n_games, alternate=True)
    wr     = report.win_rate(challenger_agent.name)

    print(f"  Challenger WR : {wr*100:.1f}%  "
          f"(V={report.wins(challenger_agent.name)}  "
          f"D={report.draws()}  "
          f"L={report.wins(champ_agent.name)}  "
          f"/ {n_games})")

    return wr >= win_threshold, wr


# ═════════════════════════════════════════════════════════════════════════════
# BOUCLE PRINCIPALE
# ═════════════════════════════════════════════════════════════════════════════

def run_alphazero(
    # Self-play
    simulations:     int   = 200,
    games_per_iter:  int   = 50,
    temp_cutoff:     int   = 15,
    # Buffer
    buffer_size:     int   = 100_000,
    min_buffer:      int   = 1_000,
    # Entraînement
    train_steps:     int   = 200,
    batch_size:      int   = 256,
    lr:              float = 1e-3,
    weight_decay:    float = 1e-4,
    # Évaluation
    eval_games:      int   = 20,
    eval_sims:       int   = 100,
    win_threshold:   float = 0.55,
    eval_every:      int   = 5,     # évalue tous les N itérations
    # Réseau
    num_filters:     int   = 128,
    num_res_blocks:  int   = 4,
    # Persistance
    checkpoint_dir:  str   = "models/alphazero",
    resume:          str   = None,
    # Durée
    iterations:      int   = 200,
    device:          str   = None,
):
    os.makedirs(checkpoint_dir, exist_ok=True)

    # ── Initialisation ────────────────────────────────────────────────────

    # Réseau courant (celui qu'on entraîne)
    current = LightEvaluator(
        checkpoint     = resume,
        device         = device,
        num_filters    = num_filters,
        num_res_blocks = num_res_blocks,
    )

    # Champion (snapshot du meilleur réseau connu)
    champion = LightEvaluator(
        device         = device,
        num_filters    = num_filters,
        num_res_blocks = num_res_blocks,
    )
    champion.clone_weights_from(current)

    buffer  = ReplayBuffer(max_size=buffer_size)
    trainer = Trainer(current, buffer, lr=lr, weight_decay=weight_decay,
                      total_steps=iterations * train_steps)

    engine  = MCTSEngine(current, simulations=simulations)

    print(f"\n{'═'*60}")
    print(f"  AlphaZero UTTT")
    print(f"  Device         : {current.device}")
    print(f"  Simulations    : {simulations}")
    print(f"  Parties/iter   : {games_per_iter}")
    print(f"  Steps/epoch    : {train_steps}")
    print(f"  Buffer max     : {buffer_size}")
    print(f"  Checkpoint dir : {checkpoint_dir}")
    print(f"{'═'*60}\n")

    # ── Boucle principale ─────────────────────────────────────────────────

    best_wr = 0.0

    for iteration in range(1, iterations + 1):
        t_iter = time.time()
        print(f"{'─'*60}")
        print(f"  ITÉRATION {iteration}/{iterations}")
        print(f"{'─'*60}")

        # ── 1. SELF-PLAY ─────────────────────────────────────────────────
        print(f"\n  [1/3] Self-play ({games_per_iter} parties, {simulations} sims/coup)")
        t0 = time.time()
        n_examples = 0

        for g in range(1, games_per_iter + 1):
            examples = play_self_play_game(engine, temp_cutoff=temp_cutoff)
            buffer.push_game(examples)
            n_examples += len(examples)

            if g % max(1, games_per_iter // 5) == 0:
                print(f"    Partie {g:3d}/{games_per_iter}  "
                      f"({len(examples)} coups)  "
                      f"buffer={len(buffer)}")

        print(f"  → {n_examples} exemples ajoutés en {time.time()-t0:.1f}s  "
              f"(buffer total : {len(buffer)})")

        # ── 2. ENTRAÎNEMENT ──────────────────────────────────────────────
        if len(buffer) < min_buffer:
            print(f"\n  [2/3] Entraînement ignoré (buffer={len(buffer)} < {min_buffer})")
        else:
            print(f"\n  [2/3] Entraînement ({train_steps} steps, batch={batch_size})")
            t0 = time.time()
            summary = trainer.train_epoch(n_steps=train_steps, batch_size=batch_size,
                                          log_every=train_steps // 5)
            print(f"  → loss={summary['loss']:.4f}  "
                  f"v={summary['loss_v']:.4f}  "
                  f"p={summary['loss_p']:.4f}  "
                  f"({time.time()-t0:.1f}s)")

        # ── 3. ÉVALUATION ────────────────────────────────────────────────
        if iteration % eval_every == 0 and len(buffer) >= min_buffer:
            print(f"\n  [3/3] Évaluation challenger vs champion ({eval_games} parties)")
            promoted, wr = evaluate_challenger(
                champion, current,
                simulations   = eval_sims,
                n_games       = eval_games,
                win_threshold = win_threshold,
            )

            if promoted:
                print(f"  ✓ Challenger promu ! WR={wr*100:.1f}% ≥ {win_threshold*100:.0f}%")
                champion.clone_weights_from(current)
                path = os.path.join(checkpoint_dir, f"best_iter{iteration:04d}.pth")
                current.save(path)
                current.save(os.path.join(checkpoint_dir, "best.pth"))
                best_wr = wr
            else:
                print(f"  ~ Champion conservé. WR={wr*100:.1f}% < {win_threshold*100:.0f}%")
                # Remet les poids du champion (le challenger n'était pas meilleur)
                current.clone_weights_from(champion)
        else:
            print(f"\n  [3/3] Évaluation dans {eval_every - (iteration % eval_every)} itérations")

        # ── Checkpoint périodique ─────────────────────────────────────────
        path = os.path.join(checkpoint_dir, f"iter_{iteration:04d}.pth")
        current.save(path)

        elapsed = time.time() - t_iter
        print(f"\n  Itération {iteration} terminée en {elapsed:.1f}s  |  best WR={best_wr*100:.1f}%\n")

    print("═" * 60)
    print(f"  Entraînement terminé. Meilleur WR : {best_wr*100:.1f}%")
    print(f"  Checkpoints → {checkpoint_dir}/")
    print("═" * 60)


# ═════════════════════════════════════════════════════════════════════════════
# POINT D'ENTRÉE
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AlphaZero self-play pour UTTT")

    # Self-play
    parser.add_argument("--simulations",    type=int,   default=200,
                        help="Simulations MCTS par coup (défaut : 200)")
    parser.add_argument("--games-per-iter", type=int,   default=50,
                        help="Parties de self-play par itération (défaut : 50)")
    parser.add_argument("--temp-cutoff",    type=int,   default=15,
                        help="Coup à partir duquel on passe en température basse (défaut : 15)")

    # Buffer
    parser.add_argument("--buffer-size",    type=int,   default=100_000)
    parser.add_argument("--min-buffer",     type=int,   default=1_000,
                        help="Exemples minimum avant de commencer l'entraînement")

    # Entraînement
    parser.add_argument("--train-steps",    type=int,   default=200)
    parser.add_argument("--batch-size",     type=int,   default=256)
    parser.add_argument("--lr",             type=float, default=1e-3)
    parser.add_argument("--weight-decay",   type=float, default=1e-4)

    # Évaluation
    parser.add_argument("--eval-games",     type=int,   default=20)
    parser.add_argument("--eval-sims",      type=int,   default=100)
    parser.add_argument("--win-threshold",  type=float, default=0.55,
                        help="Win rate minimum pour promouvoir le challenger (défaut : 0.55)")
    parser.add_argument("--eval-every",     type=int,   default=5,
                        help="Évalue tous les N itérations (défaut : 5)")

    # Réseau
    parser.add_argument("--num-filters",    type=int,   default=128)
    parser.add_argument("--num-res-blocks", type=int,   default=4)

    # Persistance
    parser.add_argument("--checkpoint-dir", default="models/alphazero")
    parser.add_argument("--resume",         default=None,
                        help="Reprendre depuis un checkpoint .pth")

    # Durée
    parser.add_argument("--iterations",     type=int,   default=200)
    parser.add_argument("--device",         default=None)
    parser.add_argument("--seed",           type=int,   default=42)

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    run_alphazero(
        simulations     = args.simulations,
        games_per_iter  = args.games_per_iter,
        temp_cutoff     = args.temp_cutoff,
        buffer_size     = args.buffer_size,
        min_buffer      = args.min_buffer,
        train_steps     = args.train_steps,
        batch_size      = args.batch_size,
        lr              = args.lr,
        weight_decay    = args.weight_decay,
        eval_games      = args.eval_games,
        eval_sims       = args.eval_sims,
        win_threshold   = args.win_threshold,
        eval_every      = args.eval_every,
        num_filters     = args.num_filters,
        num_res_blocks  = args.num_res_blocks,
        checkpoint_dir  = args.checkpoint_dir,
        resume          = args.resume,
        iterations      = args.iterations,
        device          = args.device,
    )
