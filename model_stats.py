"""
model_stats.py — Statistiques et graphiques sur les différents modèles UTTT
============================================================================
Compare automatiquement tous les checkpoints disponibles (AlphaZero itératifs
+ modèle supervisé de base) et produit :

  1. Tableau récapitulatif  (CSV + print)
       - nb de paramètres, taille fichier, architecture détectée
       - win rate vs Random
       - longueur moyenne des parties
       - temps moyen par coup
  2. Round-robin entre tous les modèles
       - matrice des win rates (heatmap)
       - classement Elo approximatif
  3. Graphiques  (PNG dans stats_output/)
       - progression du win rate vs Random au fil des itérations AlphaZero
       - courbe Elo au fil des itérations
       - barres comparatives finales
       - heatmap round-robin

Utilisation
───────────
  python model_stats.py
  python model_stats.py --models-dir models/alphazero --games 20
  python model_stats.py --include models/best_uttt_model.pth --games 10 --no-round-robin

L'auto-détection d'architecture (Light vs full NeuralEvaluator) se fait en
inspectant les clés et les shapes du state_dict — pas besoin de configurer.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

# Matplotlib en mode non-interactif (utilisable sans display)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from game       import UTTTState
from arena      import Arena, Agent
from bot_random import RandomAgent


# ═════════════════════════════════════════════════════════════════════════════
# DÉTECTION D'ARCHITECTURE
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class CheckpointInfo:
    path:        str
    label:       str                       # nom court pour l'affichage
    kind:        str                       # "light" | "full"
    num_filters: int
    num_blocks:  int
    num_params:  int
    file_size:   int                       # octets
    iter_num:    Optional[int] = None      # pour les itér. AlphaZero
    # Rempli par run_benchmarks :
    win_rate_vs_random: float = 0.0
    avg_game_len:       float = 0.0
    avg_move_time_ms:   float = 0.0
    elo:                float = 1000.0
    rr_wins:  dict = field(default_factory=dict)   # label -> win rate


def inspect_checkpoint(path: str) -> CheckpointInfo:
    """Lit un .pth et en déduit l'architecture + métadonnées."""
    sd = torch.load(path, map_location="cpu", weights_only=True)
    keys = list(sd.keys())

    # L'architecture Light de bot_mcts utilise typiquement des clés différentes
    # (ex. "p_conv", ".net.0.") de la version NeuralEvaluator (model.py).
    is_light = any(("p_conv" in k) or (".net.0." in k) for k in keys)

    # Nombre de filtres = 1ère dim de la conv d'entrée
    stem_w = sd.get("stem.0.weight")
    if stem_w is None:
        # fallback : première conv trouvée
        for k, v in sd.items():
            if "weight" in k and v.ndim == 4:
                stem_w = v
                break
    num_filters = int(stem_w.shape[0]) if stem_w is not None else -1

    # Nombre de blocs résiduels = plus grand index res_blocks.N
    block_ids = [
        int(k.split(".")[1]) for k in keys
        if k.startswith("res_blocks.") and k.split(".")[1].isdigit()
    ]
    num_blocks = (max(block_ids) + 1) if block_ids else 0

    num_params = sum(v.numel() for v in sd.values() if hasattr(v, "numel"))
    file_size  = os.path.getsize(path)

    # Extrait un numéro d'itération depuis le nom de fichier (iter_0017.pth…)
    iter_num = None
    base = os.path.basename(path)
    for token in base.replace(".", "_").split("_"):
        if token.isdigit():
            iter_num = int(token)
            break

    return CheckpointInfo(
        path=path,
        label=os.path.splitext(base)[0],
        kind="light" if is_light else "full",
        num_filters=num_filters,
        num_blocks=num_blocks,
        num_params=num_params,
        file_size=file_size,
        iter_num=iter_num,
    )


# ═════════════════════════════════════════════════════════════════════════════
# CHARGEMENT D'AGENTS (avec imports paresseux pour éviter les cycles)
# ═════════════════════════════════════════════════════════════════════════════

def build_agent(info: CheckpointInfo, device: Optional[str] = None) -> Agent:
    """
    Construit un Agent prêt à jouer depuis un checkpoint.

    - Checkpoint "light" → LightEvaluator + MCTSAgent (peu de simulations
      pour rester rapide sur les benchmarks).
    - Checkpoint "full"  → NeuralEvaluator + AlphaBetaAgent (profondeur 2).
    """
    if info.kind == "light":
        from bot_mcts import LightEvaluator, MCTSAgent
        ev = LightEvaluator(
            checkpoint     = info.path,
            device         = device,
            num_filters    = info.num_filters,
            num_res_blocks = info.num_blocks,
        )
        agent = MCTSAgent(ev, simulations=50, temperature=0.1)
    else:
        from model         import NeuralEvaluator
        from bot_alphabeta import AlphaBetaAgent
        ev = NeuralEvaluator(
            checkpoint_path=info.path,
            device=device,
            num_filters=info.num_filters or 256,
            num_res_blocks=info.num_blocks or 10,
        )
        agent = AlphaBetaAgent(ev, depth=2)

    agent.name = info.label
    return agent


# ═════════════════════════════════════════════════════════════════════════════
# BENCHMARKS
# ═════════════════════════════════════════════════════════════════════════════

def bench_vs_random(agent: Agent, n_games: int) -> tuple[float, float, float]:
    """
    Fait jouer `agent` contre RandomAgent.
    Retourne (win_rate, avg_game_len, avg_move_time_ms).

    On mesure le temps total d'une run et on divise par le nombre estimé de
    coups joués par l'agent (≈ moitié des coups totaux).
    """
    opponent = RandomAgent()

    # On utilise Arena.run() qui alterne les couleurs automatiquement.
    t0 = time.time()
    report = Arena(agent, opponent).run(n_games=n_games, alternate=True)
    elapsed = time.time() - t0

    wr = report.win_rate(agent.name)

    # Longueur moyenne : certaines versions d'Arena la stockent dans le report.
    # On tente plusieurs attributs raisonnables sans planter sinon.
    avg_len = 0.0
    for attr in ("avg_length", "mean_length", "avg_moves"):
        if hasattr(report, attr):
            try:
                avg_len = float(getattr(report, attr))
                break
            except Exception:
                pass
    if avg_len == 0.0 and hasattr(report, "games"):
        try:
            lens = [len(g) for g in report.games]
            avg_len = float(np.mean(lens))
        except Exception:
            avg_len = 0.0

    # Temps moyen / coup de l'agent (estimation : la moitié des coups joués)
    est_agent_moves = max(1.0, (avg_len or 40.0) * n_games / 2.0)
    avg_ms = 1000.0 * elapsed / est_agent_moves

    return wr, avg_len, avg_ms


def round_robin(infos: list[CheckpointInfo], agents: dict[str, Agent], n_games: int):
    """
    Tournoi toutes-contre-toutes.
    Remplit info.rr_wins[other_label] = win_rate de info contre other.
    """
    labels = [i.label for i in infos]

    for i, a_info in enumerate(infos):
        for j, b_info in enumerate(infos):
            if i == j:
                a_info.rr_wins[b_info.label] = 0.5   # soi-même = neutre
                continue
            if b_info.label in a_info.rr_wins:
                continue   # déjà joué dans l'autre sens

            A = agents[a_info.label]
            B = agents[b_info.label]
            report = Arena(A, B).run(n_games=n_games, alternate=True)
            wr_a = report.win_rate(A.name)
            a_info.rr_wins[b_info.label] = wr_a
            b_info.rr_wins[a_info.label] = 1.0 - wr_a
            print(f"    {a_info.label:<25} vs {b_info.label:<25} "
                  f"→ {wr_a*100:5.1f}%")

    # Calcul Elo approximatif depuis les win rates globaux
    # Elo simple : 1000 + 400 * (mean_wr - 0.5) / 0.5 ajusté logistiquement.
    for info in infos:
        wrs = [v for k, v in info.rr_wins.items() if k != info.label]
        mean_wr = float(np.mean(wrs)) if wrs else 0.5
        # Transformation logit bornée
        eps = 1e-3
        p = min(max(mean_wr, eps), 1 - eps)
        info.elo = 1000.0 + 400.0 * np.log10(p / (1 - p))


# ═════════════════════════════════════════════════════════════════════════════
# GRAPHIQUES
# ═════════════════════════════════════════════════════════════════════════════

def plot_all(infos: list[CheckpointInfo], outdir: str, did_rr: bool):
    os.makedirs(outdir, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 110, "savefig.bbox": "tight"})

    labels = [i.label for i in infos]
    wrs    = [i.win_rate_vs_random * 100 for i in infos]

    # ── 1. Barres : win rate vs Random ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.6), 4))
    bars = ax.bar(labels, wrs, color="#4C9AFF", edgecolor="#1f4e99")
    ax.axhline(50, color="gray", linestyle="--", linewidth=0.8, label="hasard")
    ax.set_ylabel("Win rate vs Random (%)")
    ax.set_title("Performance des modèles contre un adversaire aléatoire")
    ax.set_ylim(0, 105)
    for b, v in zip(bars, wrs):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.0f}%",
                ha="center", fontsize=8)
    plt.xticks(rotation=35, ha="right")
    ax.legend()
    plt.savefig(os.path.join(outdir, "winrate_vs_random.png"))
    plt.close(fig)

    # ── 2. Progression des checkpoints AlphaZero (ordonnés par itération) ─
    az = sorted(
        [i for i in infos if i.iter_num is not None],
        key=lambda x: x.iter_num
    )
    if len(az) >= 2:
        xs = [i.iter_num for i in az]
        ys = [i.win_rate_vs_random * 100 for i in az]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(xs, ys, "o-", color="#2d8f47", linewidth=2, markersize=6)
        ax.axhline(50, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Itération AlphaZero")
        ax.set_ylabel("Win rate vs Random (%)")
        ax.set_title("Progression du self-play au fil des itérations")
        ax.set_ylim(0, 105)
        ax.grid(alpha=0.3)
        plt.savefig(os.path.join(outdir, "alphazero_progression.png"))
        plt.close(fig)

    # ── 3. Paramètres vs performance ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    sizes = [i.num_params / 1e6 for i in infos]
    ax.scatter(sizes, wrs, s=80, c="#c75b5b", edgecolor="black")
    for i, info in enumerate(infos):
        ax.annotate(info.label, (sizes[i], wrs[i]),
                    fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Paramètres (millions)")
    ax.set_ylabel("Win rate vs Random (%)")
    ax.set_title("Taille du modèle vs performance")
    ax.grid(alpha=0.3)
    plt.savefig(os.path.join(outdir, "params_vs_winrate.png"))
    plt.close(fig)

    # ── 4. Temps moyen par coup ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.6), 4))
    times = [i.avg_move_time_ms for i in infos]
    ax.bar(labels, times, color="#e0a458", edgecolor="#7c5418")
    ax.set_ylabel("Temps moyen / coup (ms)")
    ax.set_title("Coût d'inférence")
    plt.xticks(rotation=35, ha="right")
    plt.savefig(os.path.join(outdir, "move_time.png"))
    plt.close(fig)

    # ── 5. Heatmap round-robin + barres Elo ───────────────────────────────
    if did_rr and len(infos) >= 2:
        n = len(infos)
        mat = np.zeros((n, n))
        for i, a in enumerate(infos):
            for j, b in enumerate(infos):
                mat[i, j] = a.rr_wins.get(b.label, 0.5) * 100

        fig, ax = plt.subplots(figsize=(1 + n * 0.7, 1 + n * 0.7))
        im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=100)
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{mat[i,j]:.0f}",
                        ha="center", va="center", fontsize=7,
                        color="black" if 25 < mat[i, j] < 75 else "white")
        ax.set_title("Matrice round-robin (win rate ligne vs colonne, %)")
        plt.colorbar(im, ax=ax, fraction=0.046)
        plt.savefig(os.path.join(outdir, "round_robin.png"))
        plt.close(fig)

        # Elo
        order = sorted(infos, key=lambda x: x.elo, reverse=True)
        fig, ax = plt.subplots(figsize=(max(6, n * 0.6), 4))
        ax.bar([i.label for i in order], [i.elo for i in order],
               color="#7e57c2", edgecolor="#3a2465")
        ax.axhline(1000, color="gray", linestyle="--", linewidth=0.8)
        ax.set_ylabel("Elo approximatif")
        ax.set_title("Classement Elo (dérivé du round-robin)")
        plt.xticks(rotation=35, ha="right")
        for i, info in enumerate(order):
            ax.text(i, info.elo + 5, f"{info.elo:.0f}",
                    ha="center", fontsize=8)
        plt.savefig(os.path.join(outdir, "elo_ranking.png"))
        plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# EXPORT CSV + AFFICHAGE
# ═════════════════════════════════════════════════════════════════════════════

def export_csv(infos: list[CheckpointInfo], path: str):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "kind", "iter", "filters", "blocks",
                    "params", "file_size_kb",
                    "winrate_vs_random", "avg_game_len",
                    "avg_move_time_ms", "elo"])
        for i in infos:
            w.writerow([
                i.label, i.kind, i.iter_num or "",
                i.num_filters, i.num_blocks, i.num_params,
                round(i.file_size / 1024, 1),
                round(i.win_rate_vs_random, 4),
                round(i.avg_game_len, 2),
                round(i.avg_move_time_ms, 2),
                round(i.elo, 1),
            ])


def print_table(infos: list[CheckpointInfo]):
    print(f"\n{'Label':<25} {'Kind':<6} {'Params':>9} {'vs Rand':>9} "
          f"{'Len':>6} {'ms/cp':>8} {'Elo':>7}")
    print("─" * 76)
    for i in sorted(infos, key=lambda x: x.elo, reverse=True):
        print(f"{i.label:<25} {i.kind:<6} {i.num_params/1e6:>7.2f}M "
              f"{i.win_rate_vs_random*100:>7.1f}% "
              f"{i.avg_game_len:>6.1f} "
              f"{i.avg_move_time_ms:>7.1f} "
              f"{i.elo:>7.0f}")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def discover_checkpoints(models_dir: str, extra: list[str]) -> list[str]:
    paths = []
    if os.path.isdir(models_dir):
        paths.extend(sorted(glob.glob(os.path.join(models_dir, "*.pth"))))
    for e in extra:
        if os.path.isfile(e) and e not in paths:
            paths.append(e)
    return paths


def main():
    parser = argparse.ArgumentParser(
        description="Statistiques et graphiques sur les modèles UTTT")
    parser.add_argument("--models-dir", default="models/alphazero",
                        help="Répertoire contenant les checkpoints AlphaZero")
    parser.add_argument("--include", nargs="*", default=["models/best_uttt_model.pth"],
                        help="Checkpoints additionnels à inclure")
    parser.add_argument("--games", type=int, default=20,
                        help="Parties par match contre Random")
    parser.add_argument("--rr-games", type=int, default=10,
                        help="Parties par match en round-robin")
    parser.add_argument("--no-round-robin", action="store_true",
                        help="Désactive le tournoi round-robin")
    parser.add_argument("--max-models", type=int, default=8,
                        help="Nombre max de modèles (on garde les + récents)")
    parser.add_argument("--outdir", default="stats_output")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    paths = discover_checkpoints(args.models_dir, args.include)
    if not paths:
        print(f"Aucun checkpoint trouvé dans '{args.models_dir}' "
              f"ni dans --include. Rien à analyser.")
        return

    print(f"\n{len(paths)} checkpoint(s) trouvé(s).")
    infos = []
    for p in paths:
        try:
            info = inspect_checkpoint(p)
            infos.append(info)
            print(f"  ✓ {info.label:<25} kind={info.kind:<6} "
                  f"filters={info.num_filters} blocks={info.num_blocks} "
                  f"params={info.num_params/1e6:.2f}M")
        except Exception as e:
            print(f"  ✗ {p} : {e}")

    # On limite si trop de modèles (garde les itérations les plus récentes
    # + tous les modèles non-AlphaZero).
    if len(infos) > args.max_models:
        az  = sorted([i for i in infos if i.iter_num is not None],
                     key=lambda x: x.iter_num)
        oth = [i for i in infos if i.iter_num is None]
        keep_n = max(0, args.max_models - len(oth))
        if keep_n > 0:
            # échantillonnage régulier pour couvrir toute la progression
            idx = np.linspace(0, len(az) - 1, keep_n).astype(int)
            az = [az[k] for k in idx]
        else:
            az = []
        infos = oth + az
        print(f"  → limité à {len(infos)} modèles.")

    # ── Bench vs Random ───────────────────────────────────────────────────
    os.makedirs(args.outdir, exist_ok=True)
    agents: dict[str, Agent] = {}
    print(f"\n[1/2] Bench vs Random ({args.games} parties par modèle)")
    for info in infos:
        print(f"  → {info.label}")
        try:
            agent = build_agent(info, device=args.device)
            agents[info.label] = agent
            wr, avg_len, avg_ms = bench_vs_random(agent, args.games)
            info.win_rate_vs_random = wr
            info.avg_game_len       = avg_len
            info.avg_move_time_ms   = avg_ms
            print(f"     WR={wr*100:.1f}%  len={avg_len:.1f}  ms/coup={avg_ms:.1f}")
        except Exception as e:
            print(f"     ✗ échec : {e}")

    # ── Round-robin ───────────────────────────────────────────────────────
    did_rr = False
    if not args.no_round_robin and len(agents) >= 2:
        print(f"\n[2/2] Round-robin ({args.rr_games} parties par match)")
        loaded_infos = [i for i in infos if i.label in agents]
        round_robin(loaded_infos, agents, args.rr_games)
        did_rr = True
    else:
        print("\n[2/2] Round-robin sauté.")

    # ── Rapport ───────────────────────────────────────────────────────────
    print_table([i for i in infos if i.label in agents])
    export_csv(infos, os.path.join(args.outdir, "models_stats.csv"))
    plot_all([i for i in infos if i.label in agents], args.outdir, did_rr)
    print(f"\nRésultats → {args.outdir}/")
    print("  - models_stats.csv")
    print("  - winrate_vs_random.png")
    print("  - alphazero_progression.png  (si ≥2 itérations)")
    print("  - params_vs_winrate.png")
    print("  - move_time.png")
    if did_rr:
        print("  - round_robin.png")
        print("  - elo_ranking.png")


if __name__ == "__main__":
    main()
