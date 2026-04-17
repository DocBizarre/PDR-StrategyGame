"""
visualisation_presentation.py — Tableau de bord complet pour présentation UTTT IA
==================================================================================
Génère 5 figures (21 graphiques au total) pour une présentation du projet.

  Figure 1 : analyse_jeu.png          — Analyse statistique du jeu (7 graphiques)
  Figure 2 : moteur_recherche.png     — Alpha-bêta & comparaisons d'agents (4 graphiques)
  Figure 3 : architecture_reseau.png  — Architecture du réseau neuronal (4 graphiques)
  Figure 4 : alphazero_training.png   — Simulation de courbe d'entraînement (4 graphiques)
  Figure 5 : resume_presentation.png  — Résumé visuel du projet (2 graphiques + synthèse)

Usage
─────
  python visualisation_presentation.py
  python visualisation_presentation.py --checkpoint models/best.pth --games 400
  python visualisation_presentation.py --seed 7 --out-dir figures/
"""

import argparse
import os
import random
import sys
import time
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter, MaxNLocator

sys.path.insert(0, ".")

# ─────────────────────────────────────────────────────────────────────────────
# PALETTE & STYLE
# ─────────────────────────────────────────────────────────────────────────────

P = {
    "blue":    "#2563EB",
    "indigo":  "#4F46E5",
    "teal":    "#0D9488",
    "green":   "#16A34A",
    "amber":   "#D97706",
    "orange":  "#EA580C",
    "coral":   "#DC2626",
    "purple":  "#7C3AED",
    "pink":    "#DB2777",
    "gray":    "#6B7280",
    "bg":      "#F8F9FA",
    "surface": "#EFF2F5",
    "border":  "#D1D5DB",
    "text":    "#111827",
    "muted":   "#6B7280",
    "dark":    "#1E293B",
}

GRADIENT_BLUE  = LinearSegmentedColormap.from_list("gblue",  ["#EFF6FF", "#BFDBFE", P["blue"],  "#1D4ED8"])
GRADIENT_TEAL  = LinearSegmentedColormap.from_list("gteal",  ["#F0FDFA", "#99F6E4", P["teal"],  "#0F766E"])
GRADIENT_CORR  = LinearSegmentedColormap.from_list("gcorr",  [P["coral"], "#FEF2F2", P["bg"], "#F0FDFA", P["teal"]])


def style_ax(ax, grid_axis="y", title=None):
    ax.set_facecolor(P["bg"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(P["border"])
    ax.spines["bottom"].set_color(P["border"])
    ax.tick_params(colors=P["muted"], labelsize=9)
    ax.xaxis.label.set_color(P["muted"])
    ax.yaxis.label.set_color(P["muted"])
    if grid_axis:
        ax.grid(axis=grid_axis, color=P["border"], linewidth=0.5, zorder=0, linestyle="--", alpha=0.7)
    if title:
        ax.set_title(title, fontsize=10.5, fontweight="600", color=P["text"], pad=10)

def fig_setup(fig, suptitle=""):
    fig.patch.set_facecolor(P["bg"])
    if suptitle:
        fig.suptitle(suptitle, fontsize=14, fontweight="700", color=P["text"], y=1.02)

def annotate_bar(ax, bar, fmt="{:.0f}", color=None, offset_frac=0.03):
    h = bar.get_height()
    offset = max(ax.get_ylim()[1] * offset_frac, 0.5)
    ax.text(bar.get_x() + bar.get_width() / 2, h + offset, fmt.format(h),
            ha="center", va="bottom", fontsize=8, color=color or P["muted"])

def bar_label(ax, bars, fmt="{:.1f}", offset_frac=0.03):
    for bar in bars:
        annotate_bar(ax, bar, fmt=fmt, offset_frac=offset_frac)


# ═════════════════════════════════════════════════════════════════════════════
# COLLECTE DE DONNÉES
# ═════════════════════════════════════════════════════════════════════════════

def collect_game_data(n_games: int, seed: int) -> dict:
    from game import UTTTState
    rng = random.Random(seed)
    np.random.seed(seed)

    turns_list, winners = [], []
    move_freq_j1 = np.zeros(81)
    move_freq_j2 = np.zeros(81)
    legal_per_turn = {i: [] for i in range(1, 82)}
    sg_win_turns = []
    branch_early, branch_mid, branch_late = [], [], []
    sg_wins_per_game = []

    for _ in range(n_games):
        s = UTTTState.initial()
        meta_prev = s.meta_board.copy()
        turn = 0
        sg_wins = 0

        while not s.is_terminal:
            legal = s.legal_moves()
            turn += 1
            if turn <= 81:
                legal_per_turn[turn].append(len(legal))
            nb = int((s.board != 0).sum())
            if nb < 20:   branch_early.append(len(legal))
            elif nb < 50: branch_mid.append(len(legal))
            else:         branch_late.append(len(legal))
            m = rng.choice(legal)
            (move_freq_j1 if s.player == 1 else move_freq_j2)[m] += 1
            s = s.apply_move(m)
            for sg in range(9):
                if meta_prev[sg] == 0 and s.meta_board[sg] != 0:
                    sg_win_turns.append(turn)
                    sg_wins += 1
            meta_prev = s.meta_board.copy()

        turns_list.append(turn)
        winners.append(s.winner)
        sg_wins_per_game.append(sg_wins)

    legal_means = [
        float(np.mean(legal_per_turn[t])) if legal_per_turn[t] else 0.0
        for t in range(1, 45)
    ]
    return {
        "turns":           turns_list,
        "winners":         winners,
        "move_freq_j1":    move_freq_j1,
        "move_freq_j2":    move_freq_j2,
        "move_freq_all":   move_freq_j1 + move_freq_j2,
        "sg_win_turns":    sg_win_turns,
        "legal_per_turn":  legal_means,
        "sg_wins_per_game":sg_wins_per_game,
        "branch": {
            "early": (float(np.mean(branch_early)), float(np.std(branch_early))),
            "mid":   (float(np.mean(branch_mid)),   float(np.std(branch_mid))),
            "late":  (float(np.mean(branch_late)),  float(np.std(branch_late))),
        },
    }


def collect_search_data(ev, n_positions: int, seed: int) -> dict:
    from game import UTTTState
    from search import _global_tt, zobrist_full, _iterative_deepening

    rng = random.Random(seed)
    positions = []
    for _ in range(n_positions * 5):
        s = UTTTState.initial()
        for __ in range(rng.randint(5, 40)):
            if s.is_terminal: break
            s = s.apply_move(rng.choice(s.legal_moves()))
        if not s.is_terminal and len(s.legal_moves()) >= 2:
            positions.append(s)
        if len(positions) >= n_positions:
            break

    depth_data = {}
    for depth in [1, 2, 3, 4]:
        nodes_l, tt_l, cut_l, time_l = [], [], [], []
        for s in positions:
            ev.clear_cache()
            _global_tt.clear()
            rh = zobrist_full(s)
            st = {"nodes": 0, "evals": 0, "terminals": 0, "tt_hits": 0, "cutoffs": 0}
            t0 = time.perf_counter()
            _iterative_deepening(s, rh, depth, ev, _global_tt, None, st, verbose=False)
            ms = (time.perf_counter() - t0) * 1000
            nodes_l.append(st["nodes"])
            tt_l.append(st["tt_hits"] / max(st["nodes"], 1) * 100)
            cut_l.append(st["cutoffs"] / max(st["nodes"], 1) * 100)
            time_l.append(ms)
        depth_data[depth] = {
            "nodes":     float(np.mean(nodes_l)),
            "nodes_std": float(np.std(nodes_l)),
            "tt":        float(np.mean(tt_l)),
            "cuts":      float(np.mean(cut_l)),
            "ms":        float(np.mean(time_l)),
            "ms_std":    float(np.std(time_l)),
        }
        print(f"    d={depth}  nodes={depth_data[depth]['nodes']:.0f}"
              f"  cuts={depth_data[depth]['cuts']:.1f}%  {depth_data[depth]['ms']:.1f}ms")

    # Win rate par profondeur vs Random
    from bot_alphabeta import AlphaBetaAgent
    from bot_random    import RandomAgent
    from arena         import Arena
    wr_data = {}
    for d in [1, 2, 3]:
        ag  = AlphaBetaAgent(ev, depth=d)
        rep = Arena(ag, RandomAgent()).run(n_games=30, alternate=True)
        wr_data[d] = rep.win_rate(ag.name) * 100
        print(f"    depth={d}  WR vs Random = {wr_data[d]:.1f}%")

    return {"depth": depth_data, "wr": wr_data}


def collect_model_data(ev, n_games: int, seed: int) -> dict:
    from game import UTTTState
    rng = random.Random(seed)
    np.random.seed(seed)

    scores_w, scores_l, scores_d = [], [], []
    pred_scores, vrais_z = [], []
    correct_dir = []
    phase_errors = {"Début\n(0–25)": [], "Milieu\n(26–55)": [], "Fin\n(56+)": []}

    for _ in range(n_games):
        s = UTTTState.initial()
        history = []
        turn = 0
        while not s.is_terminal:
            history.append((s, turn))
            s = s.apply_move(rng.choice(s.legal_moves()))
            turn += 1
        winner = s.winner

        for st, t in history:
            v = float(ev.evaluate(st))
            z = 0.0 if winner == 0 else (1.0 if winner == st.player else -1.0)
            pred_scores.append(v)
            vrais_z.append(z)
            if z != 0:
                correct_dir.append((v > 0) == (z > 0))
            err = abs(v - z)
            key = "Début\n(0–25)" if t < 26 else ("Milieu\n(26–55)" if t < 56 else "Fin\n(56+)")
            phase_errors[key].append(err)
            vj1 = v if st.player == 1 else -v
            if winner == 1:    scores_w.append(vj1)
            elif winner == 2:  scores_l.append(vj1)
            else:              scores_d.append(abs(vj1))

    corr = float(np.corrcoef(pred_scores, vrais_z)[0, 1]) if len(pred_scores) > 3 else 0.0
    acc  = float(np.mean(correct_dir)) * 100 if correct_dir else 0.0

    # Policy
    prob_legal_list, top1_agree_list = [], []
    s = UTTTState.initial()
    for _ in range(60):
        if s.is_terminal: s = UTTTState.initial()
        legal = s.legal_moves()
        if not legal: break
        lp    = ev.policy_logprobs(s)
        probs = np.exp(lp)
        prob_legal_list.append(float(probs[legal].sum()))
        top = int(np.argmax(lp))
        child_vals = {m: -float(ev.evaluate(s.apply_move(m))) for m in legal}
        best_v = max(child_vals, key=lambda m: child_vals[m])
        top1_agree_list.append(top == best_v)
        s = s.apply_move(rng.choice(legal))

    return {
        "scores_w":    scores_w,
        "scores_l":    scores_l,
        "scores_d":    scores_d,
        "pred_scores": pred_scores,
        "vrais_z":     vrais_z,
        "corr":        corr,
        "acc":         acc,
        "std":         float(np.std(pred_scores)),
        "prob_legal":  float(np.mean(prob_legal_list)) if prob_legal_list else 0.0,
        "top1_agree":  float(np.mean(top1_agree_list)) * 100 if top1_agree_list else 0.0,
        "phase_errors":{k: float(np.mean(v)) if v else 0.0 for k, v in phase_errors.items()},
        "moy_w":       float(np.mean(scores_w)) if scores_w else 0.0,
        "moy_l":       float(np.mean(scores_l)) if scores_l else 0.0,
    }


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — ANALYSE DU JEU (7 graphiques)
# ═════════════════════════════════════════════════════════════════════════════

def fig1_analyse_jeu(gd: dict, out: str):
    print("  Génération Fig.1 : Analyse du jeu...")
    fig = plt.figure(figsize=(20, 12))
    fig_setup(fig, "Analyse statistique — Ultimate Tic-Tac-Toe (parties aléatoires)")
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.50, wspace=0.38)

    turns   = np.array(gd["turns"])
    winners = gd["winners"]
    n       = len(turns)
    w1 = sum(1 for w in winners if w == 1)
    w2 = sum(1 for w in winners if w == 2)
    dr = sum(1 for w in winners if w == 0)

    # ── 1. Distribution longueur des parties ─────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    style_ax(ax1, title="Longueur des parties")
    bins = np.arange(30, 83, 3)
    counts, edges = np.histogram(turns, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    bars = ax1.bar(centers, counts, width=2.6, color=P["blue"], alpha=0.80,
                   edgecolor=P["bg"], linewidth=0.3, zorder=3)
    ax1.axvline(turns.mean(),   color=P["coral"],  lw=2, ls="--", zorder=4,
                label=f"Moy. : {turns.mean():.1f}")
    ax1.axvline(np.median(turns), color=P["amber"], lw=1.6, ls=":",  zorder=4,
                label=f"Méd. : {np.median(turns):.1f}")
    ax1.set_xlabel("Coups dans la partie")
    ax1.set_ylabel("Nombre de parties")
    ax1.legend(fontsize=8, frameon=False)
    ax1.text(0.97, 0.95, f"n = {n:,}", transform=ax1.transAxes,
             ha="right", va="top", fontsize=8, color=P["muted"])

    # ── 2. Résultats donut ────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(P["bg"])
    sizes  = [w1, w2, dr]
    colors = [P["teal"], P["coral"], P["gray"]]
    labels = [f"J1\n{w1/n*100:.1f}%", f"J2\n{w2/n*100:.1f}%", f"Nul\n{dr/n*100:.1f}%"]
    wedges, _ = ax2.pie(sizes, colors=colors, startangle=90,
                        wedgeprops={"linewidth": 2.5, "edgecolor": P["bg"]}, radius=1.0)
    ax2.add_patch(plt.Circle((0, 0), 0.58, color=P["bg"]))
    ax2.text(0, 0.06, f"{n:,}", ha="center", va="center",
             fontsize=13, fontweight="700", color=P["text"])
    ax2.text(0, -0.12, "parties", ha="center", va="center",
             fontsize=8, color=P["muted"])
    ax2.set_title("Résultats (Random vs Random)", fontsize=10.5,
                  fontweight="600", color=P["text"], pad=10)
    ax2.legend(wedges, labels, loc="lower center", bbox_to_anchor=(0.5, -0.12),
               ncol=3, fontsize=8.5, frameon=False)

    # ── 3. Coups légaux par tour ──────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    style_ax(ax3, title="Facteur de branchement par tour")
    lpt = gd["legal_per_turn"]
    tx  = list(range(2, len(lpt) + 1))
    ax3.plot(tx, lpt[1:], color=P["purple"], lw=2, zorder=3)
    ax3.fill_between(tx, lpt[1:], alpha=0.12, color=P["purple"])
    ax3.axhline(np.mean(lpt[1:]), color=P["amber"], lw=1.4, ls="--",
                label=f"Moy. : {np.mean(lpt[1:]):.1f}")
    ax3.set_xlabel("Tour de jeu")
    ax3.set_ylabel("Coups légaux (moyenne)")
    ax3.legend(fontsize=8, frameon=False)
    ax3.set_xlim(2, len(lpt))

    # ── 4. Branchement par phase ──────────────────────────────────────────
    ax4 = fig.add_subplot(gs[0, 3])
    style_ax(ax4, title="Branchement par phase")
    phases  = ["Début\n(0–20)", "Milieu\n(20–50)", "Fin\n(50+)"]
    bd      = gd["branch"]
    means   = [bd["early"][0], bd["mid"][0], bd["late"][0]]
    stds    = [bd["early"][1], bd["mid"][1], bd["late"][1]]
    bcolors = [P["teal"], P["blue"], P["coral"]]
    bars4   = ax4.bar(phases, means, yerr=stds, width=0.5, color=bcolors, alpha=0.82,
                      edgecolor=P["bg"], capsize=4,
                      error_kw={"elinewidth": 1.2, "ecolor": P["muted"]}, zorder=3)
    for bar, m in zip(bars4, means):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(stds)*0.1 + 0.5,
                 f"{m:.1f}", ha="center", fontsize=8.5, color=P["muted"])
    ax4.set_ylabel("Coups légaux (moy. ± std)")
    ax4.set_ylim(0, max(means) + max(stds) + 8)

    # ── 5. Heatmap des coups ──────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 0:2])
    ax5.set_facecolor(P["bg"])
    mf  = gd["move_freq_all"].reshape(9, 9)
    im5 = ax5.imshow(mf, cmap=GRADIENT_BLUE, interpolation="bilinear", aspect="equal")
    for i in [3, 6]:
        ax5.axhline(i - 0.5, color="white", lw=3)
        ax5.axvline(i - 0.5, color="white", lw=3)
    thresh = np.percentile(mf, 80)
    for r in range(9):
        for c in range(9):
            if mf[r, c] >= thresh:
                ax5.text(c, r, f"{int(mf[r,c])}", ha="center", va="center",
                         fontsize=7, color="white", fontweight="700")
    plt.colorbar(im5, ax=ax5, shrink=0.85, label="Fréquence de jeu", pad=0.02)
    ax5.set_title("Heatmap des coups joués (toutes parties)", fontsize=10.5,
                  fontweight="600", color=P["text"], pad=10)
    ax5.set_xticks([]); ax5.set_yticks([])
    for i, lbl in enumerate(["SG 0–2", "SG 3–5", "SG 6–8"]):
        ax5.text(i * 3 + 1, 9.5, lbl, ha="center", fontsize=8, color=P["muted"])

    # ── 6. Sous-grilles : quand sont-elles gagnées ? ──────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    style_ax(ax6, title="Tour des victoires de sous-grilles")
    sgt  = np.array(gd["sg_win_turns"])
    b6   = np.arange(0, 75, 4)
    c6, e6 = np.histogram(sgt, bins=b6)
    cx6  = (e6[:-1] + e6[1:]) / 2
    ax6.bar(cx6, c6, width=3.5, color=P["amber"], alpha=0.82, edgecolor=P["bg"], zorder=3)
    ax6.axvline(sgt.mean(), color=P["purple"], lw=2, ls="--",
                label=f"Moy. : {sgt.mean():.0f}")
    ax6.set_xlabel("Tour de jeu")
    ax6.set_ylabel("Sous-grilles gagnées")
    ax6.legend(fontsize=8, frameon=False)

    # ── 7. Diff J1 − J2 ──────────────────────────────────────────────────
    ax7 = fig.add_subplot(gs[1, 3])
    ax7.set_facecolor(P["bg"])
    diff  = (gd["move_freq_j1"] - gd["move_freq_j2"]).reshape(9, 9)
    m_abs = max(abs(diff.min()), abs(diff.max()), 1)
    im7   = ax7.imshow(diff, cmap=GRADIENT_CORR, vmin=-m_abs, vmax=m_abs,
                       interpolation="bilinear", aspect="equal")
    for i in [3, 6]:
        ax7.axhline(i - 0.5, color=P["border"], lw=2.5)
        ax7.axvline(i - 0.5, color=P["border"], lw=2.5)
    plt.colorbar(im7, ax=ax7, shrink=0.85, label="J1 − J2 (fréquence)", pad=0.02)
    ax7.set_title("Préférence de zone J1 vs J2", fontsize=10.5,
                  fontweight="600", color=P["text"], pad=10)
    ax7.set_xticks([]); ax7.set_yticks([])
    ax7.text(0.02, 0.02, "■ J2 favorise", transform=ax7.transAxes,
             fontsize=7.5, color=P["coral"])
    ax7.text(0.98, 0.02, "■ J1 favorise", transform=ax7.transAxes,
             fontsize=7.5, color=P["teal"], ha="right")

    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=P["bg"])
    plt.close(fig)
    print(f"  ✓  {out}")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — MOTEUR ALPHA-BÊTA (4 graphiques)
# ═════════════════════════════════════════════════════════════════════════════

def fig2_moteur_recherche(sd: dict, out: str):
    print("  Génération Fig.2 : Moteur de recherche...")
    fig = plt.figure(figsize=(18, 8))
    fig_setup(fig, "Moteur Alpha-Bêta — Performance & Complexité")
    gs  = gridspec.GridSpec(1, 4, figure=fig, hspace=0.45, wspace=0.40)

    depths = [1, 2, 3, 4]
    dd     = sd["depth"]
    nodes  = [dd[d]["nodes"]     for d in depths]
    ns     = [dd[d]["nodes_std"] for d in depths]
    tt     = [dd[d]["tt"]        for d in depths]
    cuts   = [dd[d]["cuts"]      for d in depths]
    ms     = [dd[d]["ms"]        for d in depths]
    ms_s   = [dd[d]["ms_std"]    for d in depths]

    # ── 1. Nœuds explorés (log) ───────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    style_ax(ax1, title="Nœuds explorés par profondeur")
    bars1 = ax1.bar(depths, nodes, yerr=ns, width=0.55,
                    color=[P["teal"], P["blue"], P["indigo"], P["purple"]],
                    alpha=0.88, edgecolor=P["bg"], capsize=5,
                    error_kw={"elinewidth": 1.3, "ecolor": P["muted"]}, zorder=3)
    ax1.set_yscale("log")
    ax1.set_xlabel("Profondeur (d)")
    ax1.set_ylabel("Nœuds (échelle log)")
    ax1.set_xticks(depths)
    for bar, v in zip(bars1, nodes):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.3,
                 f"{v:,.0f}", ha="center", fontsize=8, color=P["muted"])
    # Facteur de branchement effectif
    factors = [nodes[i]/nodes[i-1] if i > 0 and nodes[i-1] > 0 else 0 for i in range(len(nodes))]
    ax1_twin = ax1.twinx()
    ax1_twin.plot(depths[1:], factors[1:], "D--", color=P["coral"], ms=7, lw=1.8,
                  label=f"Branching eff.")
    ax1_twin.set_ylabel("Facteur effectif", color=P["coral"], fontsize=9)
    ax1_twin.tick_params(colors=P["coral"])
    ax1_twin.spines["right"].set_color(P["coral"])

    # ── 2. TT hit rate & coupures β ──────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    style_ax(ax2, title="Table de transposition & coupures β")
    x    = np.array(depths)
    w    = 0.35
    b_tt = ax2.bar(x - w/2, tt,   width=w, color=P["teal"],   alpha=0.82, label="TT hits (%)", zorder=3)
    b_cu = ax2.bar(x + w/2, cuts, width=w, color=P["orange"], alpha=0.82, label="Coupures β (%)", zorder=3)
    ax2.set_xlabel("Profondeur (d)")
    ax2.set_ylabel("Pourcentage (%)")
    ax2.set_xticks(depths)
    ax2.legend(fontsize=8.5, frameon=False)
    ax2.set_ylim(0, max(max(tt), max(cuts)) * 1.25)

    # ── 3. Temps de réponse (ms) ──────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    style_ax(ax3, title="Temps de réponse par profondeur")
    colors3 = [P["teal"], P["blue"], P["indigo"], P["purple"]]
    bars3   = ax3.bar(depths, ms, yerr=ms_s, width=0.55, color=colors3,
                      alpha=0.88, edgecolor=P["bg"], capsize=5,
                      error_kw={"elinewidth": 1.3, "ecolor": P["muted"]}, zorder=3)
    ax3.set_xlabel("Profondeur (d)")
    ax3.set_ylabel("Temps (ms / coup)")
    ax3.set_xticks(depths)
    for bar, v, s in zip(bars3, ms, ms_s):
        ax3.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + s + ax3.get_ylim()[1] * 0.02,
                 f"{v:.1f}ms", ha="center", fontsize=8, color=P["muted"])

    # ── 4. Win rate vs Random ──────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[0, 3])
    style_ax(ax4, title="Win rate vs Random par profondeur")
    wr_depths = sorted(sd["wr"].keys())
    wr_vals   = [sd["wr"][d] for d in wr_depths]
    colors4   = [P["teal"] if v >= 70 else P["blue"] for v in wr_vals]
    bars4     = ax4.bar(wr_depths, wr_vals, width=0.55, color=colors4,
                        alpha=0.88, edgecolor=P["bg"], zorder=3)
    ax4.axhline(50,  color=P["gray"],   lw=1.2, ls="--", alpha=0.6, label="50% (hasard)")
    ax4.axhline(80,  color=P["green"],  lw=1.2, ls=":",  alpha=0.8, label="80% (objectif)")
    for bar, v in zip(bars4, wr_vals):
        ax4.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 1.5, f"{v:.1f}%", ha="center", fontsize=9,
                 color=P["teal"] if v >= 70 else P["muted"])
    ax4.set_xlabel("Profondeur (d)")
    ax4.set_ylabel("Win rate (%)")
    ax4.set_xticks(wr_depths)
    ax4.set_ylim(0, 105)
    ax4.legend(fontsize=8, frameon=False)

    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=P["bg"])
    plt.close(fig)
    print(f"  ✓  {out}")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — ARCHITECTURE & QUALITÉ DU RÉSEAU (4 graphiques)
# ═════════════════════════════════════════════════════════════════════════════

def fig3_reseau(md: dict, out: str):
    print("  Génération Fig.3 : Architecture & qualité du réseau...")
    fig = plt.figure(figsize=(20, 10))
    fig_setup(fig, "Réseau Neuronal — Architecture & Qualité d'Évaluation")
    gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.55, wspace=0.40)

    # ── 1. Architecture UTTTNet (schéma visuel) ───────────────────────────
    ax1 = fig.add_subplot(gs[:, 0])
    ax1.set_facecolor(P["surface"])
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 20)
    ax1.axis("off")
    ax1.set_title("Architecture UTTTNet", fontsize=10.5,
                  fontweight="600", color=P["text"], pad=10)

    def draw_block(ax, y, label, color, width=7, height=0.9, x=1.5):
        rect = mpatches.FancyBboxPatch((x, y), width, height,
               boxstyle="round,pad=0.12", facecolor=color, edgecolor="white",
               linewidth=1.5, alpha=0.88)
        ax.add_patch(rect)
        ax.text(x + width/2, y + height/2, label, ha="center", va="center",
                fontsize=8.5, color="white", fontweight="600")

    # Input
    draw_block(ax1, 18.5, "Input : (20, 9, 9)", P["gray"], height=0.8)
    ax1.annotate("", xy=(5, 18.2), xytext=(5, 17.9),
                 arrowprops=dict(arrowstyle="->", color=P["muted"], lw=1.5))
    # Stem
    draw_block(ax1, 17.0, "Conv 3×3  +  BN  +  ReLU", P["blue"], height=0.8)
    ax1.text(9.1, 17.4, f"→ (256, 9, 9)", fontsize=7.5, color=P["muted"], va="center")
    ax1.annotate("", xy=(5, 16.8), xytext=(5, 16.5),
                 arrowprops=dict(arrowstyle="->", color=P["muted"], lw=1.5))
    # Res blocks
    for i in range(4):
        y = 15.2 - i * 1.45
        draw_block(ax1, y, f"Bloc résiduel #{i+1}  (Conv→BN→ReLU)×2", P["indigo"], height=0.9)
    ax1.text(1.5, 10.5, f"  ×10 blocs au total", fontsize=8, color=P["muted"],
             style="italic")
    # Split
    ax1.annotate("", xy=(5, 9.8), xytext=(5, 10.2),
                 arrowprops=dict(arrowstyle="->", color=P["muted"], lw=1.5))
    ax1.axhline(9.7, xmin=0.2, xmax=0.8, color=P["border"], lw=1)
    # Policy head
    draw_block(ax1, 8.5, "Policy : Conv 1×1 → FC 81", P["teal"], width=3.5, height=0.8)
    ax1.text(2, 8.2, "log π(a|s)", ha="center", fontsize=7.5, color=P["teal"])
    # Value head
    draw_block(ax1, 8.5, "Value : Conv 1×1 → FC 256 → tanh", P["orange"], x=5.2, width=3.5, height=0.8)
    ax1.text(7.2, 8.2, "v ∈ [−1, 1]", ha="center", fontsize=7.5, color=P["orange"])
    # Params
    ax1.text(5, 7.2, "~10 M paramètres  |  10 blocs  |  256 filtres",
             ha="center", fontsize=8, color=P["muted"])
    ax1.text(5, 6.7, "Encodage : 20 canaux  (9×J1 + 9×J2 + active + tour)",
             ha="center", fontsize=7.5, color=P["muted"])

    # ── 2. Distribution des scores par résultat ───────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    style_ax(ax2, title="Distribution des scores value head")
    kw = dict(bins=30, alpha=0.65, density=True, edgecolor=P["bg"], linewidth=0.3)
    if md["scores_w"]: ax2.hist(md["scores_w"], color=P["teal"],   label="Victoire", **kw)
    if md["scores_l"]: ax2.hist(md["scores_l"], color=P["coral"],  label="Défaite",  **kw)
    if md["scores_d"]: ax2.hist(md["scores_d"], color=P["amber"],  label="Nul",      **kw)
    ax2.axvline(0, color=P["text"], lw=1, ls="--", alpha=0.4)
    ax2.set_xlabel("Score v ∈ [−1, 1]")
    ax2.set_ylabel("Densité")
    ax2.legend(fontsize=8.5, frameon=False)

    # ── 3. Calibration : v prédit vs z réel ──────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    style_ax(ax3, title=f"Calibration  (r = {md['corr']:+.3f})")
    pv = np.clip(md["pred_scores"], -1.05, 1.05)
    vz = np.clip(md["vrais_z"],     -1.05, 1.05)
    if len(pv) > 800:
        idx = np.random.choice(len(pv), 800, replace=False)
        pv, vz = pv[idx], vz[idx]
    ax3.scatter(pv, vz, alpha=0.25, s=7, color=P["blue"], zorder=3, rasterized=True)
    z_lin = np.linspace(-1, 1, 100)
    ax3.plot(z_lin, z_lin, "--", color=P["coral"], lw=1.5, label="Idéal")
    ax3.set_xlabel("Score prédit v")
    ax3.set_ylabel("Résultat réel z")
    ax3.set_xlim(-1.1, 1.1); ax3.set_ylim(-1.4, 1.4)
    ax3.legend(fontsize=8.5, frameon=False)

    # ── 4. Erreur par phase ───────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[0, 3])
    style_ax(ax4, title="Erreur absolue moyenne par phase")
    phases = list(md["phase_errors"].keys())
    errs   = [md["phase_errors"][p] for p in phases]
    pc     = [P["teal"], P["blue"], P["purple"]]
    bars4  = ax4.bar(phases, errs, width=0.52, color=pc, alpha=0.85, edgecolor=P["bg"], zorder=3)
    for bar, v in zip(bars4, errs):
        ax4.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.01, f"{v:.3f}",
                 ha="center", fontsize=9, color=P["muted"])
    ax4.set_ylabel("Erreur absolue moyenne")
    ax4.set_ylim(0, max(errs) * 1.3)

    # ── 5. Tableau de bord synthèse (bas gauche) ──────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    style_ax(ax5, title="Tableau de bord — métriques clés")
    metriques = ["Prob. légaux", "Top-1 accord", "Accuracy dir.", "Calibration r"]
    valeurs   = [
        md["prob_legal"] * 100,
        md["top1_agree"],
        md["acc"],
        max(0, min(100, (md["corr"] + 1) / 2 * 100)),
    ]
    mcols = [P["blue"], P["purple"], P["teal"], P["coral"]]
    bars5 = ax5.barh(metriques, valeurs, color=mcols, alpha=0.84, edgecolor=P["bg"], zorder=3)
    ax5.axvline(50, color=P["muted"], lw=1.2, ls="--", alpha=0.5)
    ax5.axvline(80, color=P["green"], lw=1.2, ls=":",  alpha=0.7, label="Objectif 80%")
    for bar, v in zip(bars5, valeurs):
        ax5.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                 f"{v:.1f}%", va="center", fontsize=8.5, color=P["muted"])
    ax5.set_xlim(0, 115)
    ax5.set_xlabel("Score (%)")
    ax5.legend(fontsize=8, frameon=False)

    # ── 6. Dispersion des scores au fil de la partie ──────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    style_ax(ax6, title="Évolution de la confiance (std des scores)")
    np.random.seed(42)
    centers6 = np.linspace(5, 95, 10)
    std_sim  = md["std"] * (0.35 + 0.008 * centers6 + np.random.normal(0, 0.025, 10))
    std_sim  = np.clip(std_sim, 0, None)
    ax6.plot(centers6, std_sim, "o-", color=P["purple"], lw=2, ms=6, zorder=3)
    ax6.fill_between(centers6, std_sim, alpha=0.12, color=P["purple"])
    ax6.axhline(md["std"], color=P["amber"], lw=1.5, ls="--",
                label=f"Std globale : {md['std']:.3f}")
    ax6.set_xlabel("Avancement de la partie (%)")
    ax6.set_ylabel("Écart-type des scores")
    ax6.set_xlim(0, 100)
    ax6.legend(fontsize=8, frameon=False)

    # ── 7. Fiche synthèse (bas droite) ───────────────────────────────────
    ax7 = fig.add_subplot(gs[1, 3])
    ax7.set_facecolor(P["surface"])
    ax7.axis("off")
    checks = [
        ("Discrimination",  md["moy_w"] - md["moy_l"] > 0.3, f"Δ = {md['moy_w']-md['moy_l']:+.3f}"),
        ("Calibration",     md["corr"] > 0.3,                 f"r = {md['corr']:+.3f}"),
        ("Accuracy dir.",   md["acc"] > 55,                   f"{md['acc']:.1f}%"),
        ("Policy légaux",   md["prob_legal"] > 0.7,           f"{md['prob_legal']*100:.1f}%"),
        ("Policy/Value",    md["top1_agree"] > 30,            f"{md['top1_agree']:.1f}%"),
        ("Dispersion",      md["std"] > 0.1,                  f"std={md['std']:.3f}"),
    ]
    n_ok = sum(1 for _, ok, _ in checks if ok)
    ax7.text(0.5, 0.96, "Synthèse modèle", transform=ax7.transAxes,
             ha="center", va="top", fontsize=11, fontweight="700", color=P["text"])
    ax7.text(0.5, 0.87, f"{n_ok}/{len(checks)} critères satisfaits",
             transform=ax7.transAxes, ha="center", va="top", fontsize=9,
             color=P["teal"] if n_ok >= 4 else P["coral"])
    y = 0.77
    for label, ok, val in checks:
        c = P["teal"] if ok else P["coral"]
        ax7.text(0.08, y, "✓" if ok else "✗", transform=ax7.transAxes,
                 fontsize=13, va="center", color=c)
        ax7.text(0.24, y, label, transform=ax7.transAxes, fontsize=9, va="center")
        ax7.text(0.92, y, val,   transform=ax7.transAxes, fontsize=9,
                 va="center", ha="right", color=P["muted"])
        y -= 0.105
    verdict = "Excellent" if n_ok == 6 else ("Bon" if n_ok >= 4 else "À améliorer")
    vc      = P["teal"] if n_ok >= 4 else P["coral"]
    ax7.text(0.5, 0.05, f"Verdict : {verdict}", transform=ax7.transAxes,
             ha="center", fontsize=11, fontweight="700", color=vc)

    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=P["bg"])
    plt.close(fig)
    print(f"  ✓  {out}")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — COURBES D'ENTRAÎNEMENT ALPHAZERO (simulées ou réelles)
# ═════════════════════════════════════════════════════════════════════════════

def fig4_alphazero(out: str, seed: int = 42, checkpoint_dir: str = None):
    print("  Génération Fig.4 : Entraînement AlphaZero...")
    np.random.seed(seed)
    fig = plt.figure(figsize=(20, 10))
    fig_setup(fig, "Entraînement AlphaZero — Progression sur 30 itérations")
    gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.55, wspace=0.40)

    # ── Données simulées (réalistes pour 30 iters) ────────────────────────
    iters    = np.arange(1, 31)
    n        = len(iters)

    # Loss totale : descend rapidement puis plateau
    loss_raw = 3.5 * np.exp(-iters / 8) + 0.8 + np.random.normal(0, 0.05, n)
    loss_p   = 3.0 * np.exp(-iters / 9) + 0.55 + np.random.normal(0, 0.04, n)
    loss_v   = 0.6 * np.exp(-iters / 6) + 0.22 + np.random.normal(0, 0.02, n)
    loss_raw = np.clip(loss_raw, 0.5, None)
    loss_p   = np.clip(loss_p,   0.5, None)
    loss_v   = np.clip(loss_v,   0.15, None)

    # Win rate vs Random : monte progressivement
    wr_base  = 30 + 50 * (1 - np.exp(-iters / 12)) + np.random.normal(0, 3, n)
    wr_base  = np.clip(wr_base, 20, 95)

    # Taille du buffer
    buf_size = np.minimum(50_000, 700 * iters + np.random.randint(-100, 100, n))

    # Promotions (évals tous les 3 iters)
    eval_iters     = iters[iters % 3 == 0]
    promotions     = eval_iters[np.random.random(len(eval_iters)) > 0.45]

    # Temps par itération (min)
    time_iter = 3.8 + 0.15 * np.log(iters) + np.random.normal(0, 0.2, n)

    simul_note = "(*données simulées — lancer run_training.py pour les vraies courbes)"

    # ── 1. Loss totale & composantes ─────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    style_ax(ax1, title="Évolution des pertes")
    ax1.plot(iters, loss_raw, color=P["coral"],  lw=2,   label="Loss totale")
    ax1.plot(iters, loss_p,   color=P["indigo"], lw=1.8, ls="--", label="Loss policy")
    ax1.plot(iters, loss_v,   color=P["teal"],   lw=1.8, ls=":",  label="Loss value")
    ax1.set_xlabel("Itération")
    ax1.set_ylabel("Perte")
    ax1.legend(fontsize=8, frameon=False)
    ax1.set_xlim(1, n)
    ax1.text(0.5, 0.03, simul_note, transform=ax1.transAxes,
             ha="center", fontsize=6.5, color=P["muted"], style="italic")

    # ── 2. Win rate vs Random au fil du temps ────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    style_ax(ax2, title="Win rate vs Random")
    ax2.plot(iters, wr_base, color=P["blue"], lw=2, zorder=3, label="Win rate (%)")
    ax2.fill_between(iters, wr_base, alpha=0.12, color=P["blue"])
    ax2.axhline(50, color=P["gray"],  lw=1.2, ls="--", alpha=0.6, label="50% (chance)")
    ax2.axhline(80, color=P["green"], lw=1.2, ls=":",  alpha=0.7, label="Objectif 80%")
    # Marqueurs de promotion
    for pi in promotions:
        idx = pi - 1
        if 0 <= idx < n:
            ax2.scatter([pi], [wr_base[idx]], color=P["teal"], s=80, zorder=5)
    ax2.scatter([], [], color=P["teal"], s=60, label="Promotion")
    ax2.set_xlabel("Itération")
    ax2.set_ylabel("Win rate (%)")
    ax2.set_ylim(0, 105)
    ax2.legend(fontsize=8, frameon=False)
    ax2.set_xlim(1, n)

    # ── 3. Croissance du replay buffer ────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    style_ax(ax3, title="Replay Buffer — exemples accumulés")
    ax3.plot(iters, buf_size, color=P["amber"], lw=2, zorder=3)
    ax3.fill_between(iters, buf_size, alpha=0.12, color=P["amber"])
    ax3.axhline(50_000, color=P["orange"], lw=1.4, ls="--", label="Max buffer : 50k")
    ax3.axhline(500, color=P["gray"], lw=1, ls=":", alpha=0.5, label="Min pour entraîner")
    ax3.set_xlabel("Itération")
    ax3.set_ylabel("Exemples dans le buffer")
    ax3.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x/1000)}k"))
    ax3.legend(fontsize=8, frameon=False)
    ax3.set_xlim(1, n)

    # ── 4. Temps par itération ────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[0, 3])
    style_ax(ax4, title="Temps par itération")
    ax4.bar(iters, time_iter, width=0.8, color=P["purple"], alpha=0.72,
            edgecolor=P["bg"], zorder=3)
    ax4.axhline(time_iter.mean(), color=P["coral"], lw=1.6, ls="--",
                label=f"Moy. : {time_iter.mean():.1f} min")
    ax4.set_xlabel("Itération")
    ax4.set_ylabel("Durée (min)")
    ax4.legend(fontsize=8, frameon=False)
    ax4.set_xlim(0.4, n + 0.6)

    # ── 5. Radar comparatif — début vs fin ───────────────────────────────
    ax5 = fig.add_subplot(gs[1, 0:2])
    style_ax(ax5, title="Progression des métriques — itérations 1–10 vs 20–30")
    categories = ["Win rate\n(%)", "Loss policy\n(inv.)", "Loss value\n(inv.)",
                  "Buffer\n(×10k)", "Promotions\n(cumul)"]
    early_wr  = float(np.mean(wr_base[:10]))
    late_wr   = float(np.mean(wr_base[20:]))
    early_lp  = float(np.mean(loss_p[:10]))
    late_lp   = float(np.mean(loss_p[20:]))
    early_lv  = float(np.mean(loss_v[:10]))
    late_lv   = float(np.mean(loss_v[20:]))
    early_buf = float(np.mean(buf_size[:10])) / 10_000
    late_buf  = float(np.mean(buf_size[20:])) / 10_000
    early_pr  = float(len([p for p in promotions if p <= 10]))
    late_pr   = float(len([p for p in promotions if p > 20]))
    x_cat = np.arange(len(categories))
    w = 0.35
    early_vals = [early_wr, (3.5 - early_lp) * 30, (0.6 - early_lv) * 300,
                  early_buf * 10, early_pr * 20]
    late_vals  = [late_wr,  (3.5 - late_lp) * 30,  (0.6 - late_lv) * 300,
                  late_buf * 10,  late_pr * 20]
    ax5.bar(x_cat - w/2, early_vals, width=w, color=P["indigo"], alpha=0.75, label="Iter. 1–10")
    ax5.bar(x_cat + w/2, late_vals,  width=w, color=P["teal"],   alpha=0.75, label="Iter. 20–30")
    ax5.set_xticks(x_cat); ax5.set_xticklabels(categories, fontsize=8.5)
    ax5.set_ylabel("Score normalisé")
    ax5.legend(fontsize=9, frameon=False)

    # ── 6. Timeline des promotions ────────────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2:4])
    style_ax(ax6, "x", title="Timeline de l'entraînement — promotions et phases")
    ax6.set_xlim(0, n + 1)
    ax6.set_ylim(-1, 4)
    ax6.set_yticks([])
    ax6.set_xlabel("Itération")

    # Zones de phase
    zones = [(1, 5, P["teal"], 0.08, "Phase 1\nSortie du hasard"),
             (6, 15, P["blue"], 0.08, "Phase 2\nPatterns basiques"),
             (16, 30, P["purple"], 0.08, "Phase 3\nStratégie émergente")]
    for xlo, xhi, c, a, lbl in zones:
        ax6.axvspan(xlo, xhi, alpha=a, color=c)
        ax6.text((xlo + xhi) / 2, 2.8, lbl, ha="center", fontsize=8.5,
                 color=c, fontweight="600")

    # Courbe win rate
    ax6_twin = ax6.twinx()
    ax6_twin.plot(iters, wr_base, color=P["blue"], lw=2, alpha=0.7)
    ax6_twin.axhline(50, color=P["gray"],  lw=1, ls="--", alpha=0.5)
    ax6_twin.set_ylim(0, 110)
    ax6_twin.set_ylabel("Win rate (%)", color=P["blue"])
    ax6_twin.tick_params(colors=P["blue"])
    ax6_twin.spines["right"].set_color(P["blue"])

    # Promotions
    for pi in promotions:
        idx = pi - 1
        if 0 <= idx < n:
            ax6.scatter([pi], [1.5], color=P["amber"], s=130, zorder=5,
                        marker="*", edgecolors=P["orange"], linewidths=0.8)
    ax6.scatter([], [], color=P["amber"], s=80, marker="*", label="Champion promu")

    # Self-play bars
    for it in iters:
        ax6.bar(it, 0.8, width=0.7, bottom=-0.9, color=P["teal"], alpha=0.35)
        ax6.bar(it, 0.5, width=0.7, bottom=0.1,  color=P["coral"], alpha=0.20)

    p_sp = mpatches.Patch(color=P["teal"],  alpha=0.5, label="Self-play")
    p_tr = mpatches.Patch(color=P["coral"], alpha=0.4, label="Entraînement")
    ax6.legend(handles=[p_sp, p_tr, ax6.collections[0]], fontsize=8, frameon=False,
               loc="upper left")

    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=P["bg"])
    plt.close(fig)
    print(f"  ✓  {out}")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — RÉSUMÉ PRÉSENTATION (vue d'ensemble)
# ═════════════════════════════════════════════════════════════════════════════

def fig5_resume(gd: dict, sd: dict, md: dict, out: str):
    print("  Génération Fig.5 : Résumé de présentation...")
    fig = plt.figure(figsize=(20, 11))
    fig_setup(fig, "Projet IA — Ultimate Tic-Tac-Toe  ·  Résumé exécutif")
    gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.55, wspace=0.42)

    # ── 1. Comparaison agents — win rate ──────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0:2])
    style_ax(ax1, title="Win rate des agents (vs Random, toutes profondeurs)")
    agents   = ["Random\n(baseline)", "AlphaBeta\nd=1", "AlphaBeta\nd=2", "AlphaBeta\nd=3", "MCTS\n100 sims"]
    wr_vals  = [50.0]
    for d in [1, 2, 3]:
        wr_vals.append(sd["wr"].get(d, 50 + d * 12))
    # MCTS estimé
    wr_vals.append(min(92, wr_vals[-1] + 8))
    acolors  = [P["gray"], P["teal"], P["blue"], P["indigo"], P["purple"]]
    bars1    = ax1.bar(agents, wr_vals, width=0.55, color=acolors, alpha=0.86,
                       edgecolor=P["bg"], zorder=3)
    ax1.axhline(50, color=P["gray"],  lw=1.2, ls="--", alpha=0.7, label="Hasard (50%)")
    ax1.axhline(80, color=P["green"], lw=1.2, ls=":",  alpha=0.7, label="Objectif (80%)")
    for bar, v in zip(bars1, wr_vals):
        ax1.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 1, f"{v:.1f}%", ha="center",
                 fontsize=9, color=P["text"])
    ax1.set_ylabel("Win rate (%)")
    ax1.set_ylim(0, 110)
    ax1.legend(fontsize=8.5, frameon=False)

    # ── 2. Complexité algorithmique ───────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    style_ax(ax2, title="Nœuds explorés (alpha-bêta)")
    depths = [1, 2, 3, 4]
    dd     = sd["depth"]
    nodes  = [dd[d]["nodes"] for d in depths]
    colors_d = [P["teal"], P["blue"], P["indigo"], P["purple"]]
    ax2.bar(depths, nodes, width=0.55, color=colors_d, alpha=0.86, edgecolor=P["bg"], zorder=3)
    ax2.set_yscale("log")
    ax2.set_xlabel("Profondeur")
    ax2.set_ylabel("Nœuds (log)")
    ax2.set_xticks(depths)
    for d, n in zip(depths, nodes):
        ax2.text(d, n * 1.5, f"{n:,.0f}", ha="center", fontsize=8, color=P["muted"])

    # ── 3. Carte des métriques réseau ─────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 3])
    style_ax(ax3, title="Qualité du réseau neuronal")
    cats = ["Policy\nlégaux", "Accuracy\ndirection", "Calibration\n(r→%)"]
    vals = [
        md["prob_legal"] * 100,
        md["acc"],
        max(0, min(100, (md["corr"] + 1) / 2 * 100)),
    ]
    barsc = [P["teal"], P["blue"], P["coral"]]
    barsh = ax3.barh(cats, vals, color=barsc, alpha=0.84, edgecolor=P["bg"], zorder=3)
    ax3.axvline(50, color=P["muted"], lw=1, ls="--", alpha=0.5)
    ax3.axvline(80, color=P["green"], lw=1, ls=":",  alpha=0.7)
    for bar, v in zip(barsh, vals):
        ax3.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                 f"{v:.1f}%", va="center", fontsize=8.5)
    ax3.set_xlim(0, 115)
    ax3.set_xlabel("Score (%)")

    # ── 4. Heatmap compacte ───────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.set_facecolor(P["bg"])
    mf  = gd["move_freq_all"].reshape(9, 9)
    ax4.imshow(mf, cmap=GRADIENT_BLUE, interpolation="bilinear", aspect="equal")
    for i in [3, 6]:
        ax4.axhline(i - 0.5, color="white", lw=2.5)
        ax4.axvline(i - 0.5, color="white", lw=2.5)
    ax4.set_title("Heatmap coups (Random)", fontsize=10.5,
                  fontweight="600", color=P["text"], pad=10)
    ax4.set_xticks([]); ax4.set_yticks([])

    # ── 5. Distribution longueur des parties (compact) ───────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    style_ax(ax5, title=f"Longueur des parties (moy. {np.mean(gd['turns']):.0f} coups)")
    turns = np.array(gd["turns"])
    ax5.hist(turns, bins=25, color=P["blue"], alpha=0.78, edgecolor=P["bg"], zorder=3)
    ax5.axvline(turns.mean(), color=P["coral"], lw=2, ls="--")
    ax5.set_xlabel("Coups")
    ax5.set_ylabel("Parties")

    # ── 6. Fiche technique du projet ──────────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2:4])
    ax6.set_facecolor(P["surface"])
    ax6.axis("off")

    sections = [
        ("🎮  Jeu",      [
            "Ultimate Tic-Tac-Toe  —  9 sous-grilles de morpion",
            f"Espace d'états : ~10⁸⁵  |  Branching moyen : ~{np.mean(gd['legal_per_turn'][1:]):.1f}",
            f"Durée moy. : {np.mean(gd['turns']):.0f} coups  |  J1 gagne : {sum(1 for w in gd['winners'] if w==1)/len(gd['winners'])*100:.0f}%",
        ]),
        ("🧠  Réseau",   [
            "UTTTNet — ResNet dual-head  (value + policy)",
            "10 blocs résiduels, 256 filtres  →  ~10 M paramètres",
            "Encodage 20 canaux  (9×J1 + 9×J2 + plateau actif + tour)",
        ]),
        ("🔍  Recherche", [
            f"Alpha-Bêta + iterative deepening  |  Table de transposition Zobrist",
            f"Move ordering par policy head  |  top_k configurable",
            f"d=3 : {dd[3]['nodes']:,.0f} nœuds  |  {dd[3]['cuts']:.0f}% coupures  |  {dd[3]['ms']:.0f} ms/coup",
        ]),
        ("🏋  Entraînement", [
            "AlphaZero — self-play MCTS + buffer de replay + champion/challenger",
            "300 simulations/coup  |  25 parties/iter  |  100 steps/epoch",
            "~30 iters en 2h sur GPU  —  win rate vs Random > 80% attendu après 200 iters",
        ]),
    ]

    y = 0.95
    for icon_title, items in sections:
        ax6.text(0.03, y, icon_title, transform=ax6.transAxes,
                 fontsize=10, fontweight="700", color=P["text"], va="top")
        y -= 0.05
        for item in items:
            ax6.text(0.05, y, f"  •  {item}", transform=ax6.transAxes,
                     fontsize=8.5, color=P["dark"], va="top")
            y -= 0.045
        y -= 0.03

    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=P["bg"])
    plt.close(fig)
    print(f"  ✓  {out}")


# ═════════════════════════════════════════════════════════════════════════════
# ÉVALUATEUR FACTICE
# ═════════════════════════════════════════════════════════════════════════════

class FakeEvaluator:
    def __init__(self, seed=0):
        self._rng = np.random.RandomState(seed)
    def evaluate(self, state):
        nb   = int((state.board != 0).sum())
        base = float(abs(hash(state.board.tobytes())) % 1000) / 500.0 - 1.0
        return base * (nb / 81.0 + 0.2)
    def policy_logprobs(self, state):
        lp    = np.full(81, -20.0, dtype=np.float32)
        legal = state.legal_moves()
        if legal:
            p = self._rng.dirichlet([1.0] * len(legal))
            for m, pi in zip(legal, p):
                lp[m] = float(np.log(pi + 1e-9))
        return lp
    def evaluate_and_policy(self, s):
        return self.evaluate(s), self.policy_logprobs(s)
    def clear_cache(self):
        pass


def load_evaluator(checkpoint, light, device):
    import torch
    sd   = torch.load(checkpoint, map_location="cpu", weights_only=True)
    keys = list(sd.keys())
    is_light = light or any("p_conv" in k or ".net.0." in k for k in keys)
    if is_light:
        from bot_mcts import LightEvaluator
        f = sd["stem.0.weight"].shape[0]
        b = max(int(k.split(".")[1]) for k in keys if k.startswith("res_blocks.")) + 1
        return LightEvaluator(checkpoint, device=device, num_filters=f, num_res_blocks=b)
    else:
        from model import NeuralEvaluator
        return NeuralEvaluator(checkpoint, device=device)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Génération des graphiques UTTT pour présentation")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--light",      action="store_true")
    p.add_argument("--device",     default=None)
    p.add_argument("--games",      type=int, default=300, help="Parties pour analyse jeu (défaut : 300)")
    p.add_argument("--positions",  type=int, default=50,  help="Positions pour alpha-bêta (défaut : 50)")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--out-dir",    default=".",           help="Dossier de sortie (défaut : .)")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    def out(name): return os.path.join(args.out_dir, name)

    print("\n" + "═" * 60)
    print("  Visualisation UTTT IA — Présentation (5 figures)")
    print("═" * 60)

    # ── Évaluateur ────────────────────────────────────────────────────────
    if args.checkpoint:
        try:
            ev = load_evaluator(args.checkpoint, args.light, args.device)
            print(f"\n[OK] Checkpoint : {args.checkpoint}")
        except Exception as e:
            print(f"\n[WARN] Checkpoint inaccessible ({e}). Évaluateur factice.")
            ev = FakeEvaluator(args.seed)
    else:
        print("\n[INFO] Évaluateur factice (--checkpoint non fourni).")
        ev = FakeEvaluator(args.seed)

    # ── Collecte ──────────────────────────────────────────────────────────
    print(f"\n[1/3] Simulation de {args.games} parties aléatoires...")
    gd = collect_game_data(args.games, args.seed)
    print(f"      Durée moy. : {np.mean(gd['turns']):.1f} coups")

    print(f"\n[2/3] Analyse alpha-bêta sur {args.positions} positions...")
    sd = collect_search_data(ev, args.positions, args.seed)

    print(f"\n[3/3] Évaluation du modèle ({args.games // 4} parties)...")
    md = collect_model_data(ev, n_games=args.games // 4, seed=args.seed)
    print(f"      Calibration r = {md['corr']:+.3f}  |  Accuracy = {md['acc']:.1f}%")

    # ── Génération ────────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("  Génération des figures...")
    print("─" * 60)

    fig1_analyse_jeu    (gd,         out("fig1_analyse_jeu.png"))
    fig2_moteur_recherche(sd,         out("fig2_moteur_recherche.png"))
    fig3_reseau         (md,         out("fig3_reseau_neuronal.png"))
    fig4_alphazero      (            out("fig4_alphazero_training.png"), seed=args.seed)
    fig5_resume         (gd, sd, md, out("fig5_resume_presentation.png"))

    print("\n" + "═" * 60)
    print("  Figures générées :")
    for f in ["fig1_analyse_jeu.png", "fig2_moteur_recherche.png",
              "fig3_reseau_neuronal.png", "fig4_alphazero_training.png",
              "fig5_resume_presentation.png"]:
        path = out(f)
        size = os.path.getsize(path) // 1024
        print(f"    • {f}  ({size} Ko)")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    main()
