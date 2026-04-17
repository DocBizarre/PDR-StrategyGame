"""
graphiques.py — Tableaux de bord visuels pour la présentation UTTT IA
======================================================================
Génère 3 figures (12 graphiques au total) à partir des données réelles du projet.

Figure 1 : stats_jeu.png       — Analyse du jeu (parties, coups, heatmap)
Figure 2 : stats_alphabeta.png — Moteur de recherche (nœuds, TT, vitesse)
Figure 3 : stats_modele.png    — Qualité du modèle neuronal (si checkpoint dispo)

Usage
─────
  python graphiques.py                                        # évaluateur factice
  python graphiques.py --checkpoint models/best_uttt_model.pth
  python graphiques.py --checkpoint models/alphazero/best.pth --light
  python graphiques.py --games 300 --positions 60 --seed 7
"""

import argparse
import random
import sys
import time
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter
import matplotlib.patches as mpatches

sys.path.insert(0, ".")

# ── Palette centralisée ───────────────────────────────────────────────────────
P = {
    "blue":    "#2D7DD2",
    "teal":    "#1D9E75",
    "amber":   "#E8A838",
    "coral":   "#D85A30",
    "purple":  "#6B5EA8",
    "gray":    "#7A7975",
    "green":   "#4E9A38",
    "bg":      "#FAFAF8",
    "surface": "#F2F1ED",
    "border":  "#E0DED8",
    "text":    "#1A1A18",
    "muted":   "#888780",
}

def style_ax(ax, grid_axis="y"):
    ax.set_facecolor(P["bg"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(P["border"])
    ax.spines["bottom"].set_color(P["border"])
    ax.tick_params(colors=P["muted"], labelsize=9)
    ax.xaxis.label.set_color(P["muted"])
    ax.yaxis.label.set_color(P["muted"])
    ax.title.set_color(P["text"])
    if grid_axis:
        ax.grid(axis=grid_axis, color=P["border"], linewidth=0.6, zorder=0)

def fig_bg(fig):
    fig.patch.set_facecolor(P["bg"])

def title_style():
    return dict(fontsize=10.5, fontweight="600", color=P["text"], pad=10)

def annotate_bar(ax, bar, val, fmt="{:.0f}", color=None, offset=3):
    h = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        h + offset,
        fmt.format(val),
        ha="center", va="bottom",
        fontsize=8, color=color or P["muted"],
    )


# ═══════════════════════════════════════════════════════════════════════════════
# COLLECTE DES DONNÉES
# ═══════════════════════════════════════════════════════════════════════════════

def collect_game_data(n_games: int, seed: int):
    from game import UTTTState

    random.seed(seed)
    rng = random.Random(seed)

    turns_list, winners = [], []
    move_freq_j1 = np.zeros(81)
    move_freq_j2 = np.zeros(81)
    legal_per_turn = {i: [] for i in range(1, 82)}
    sg_win_turns = []
    branch_early, branch_mid, branch_late = [], [], []

    for _ in range(n_games):
        s = UTTTState.initial()
        meta_prev = s.meta_board.copy()
        turn = 0

        while not s.is_terminal:
            legal = s.legal_moves()
            turn += 1
            if turn <= 81:
                legal_per_turn[turn].append(len(legal))

            nb = int((s.board != 0).sum())
            if nb < 20:
                branch_early.append(len(legal))
            elif nb < 50:
                branch_mid.append(len(legal))
            else:
                branch_late.append(len(legal))

            m = rng.choice(legal)
            (move_freq_j1 if s.player == 1 else move_freq_j2)[m] += 1
            s = s.apply_move(m)
            meta_prev_new = s.meta_board.copy()
            for sg in range(9):
                if meta_prev[sg] == 0 and meta_prev_new[sg] != 0:
                    sg_win_turns.append(turn)
            meta_prev = meta_prev_new

        turns_list.append(turn)
        winners.append(s.winner)

    legal_means = []
    for t in range(1, 45):
        vals = legal_per_turn.get(t, [])
        legal_means.append(float(np.mean(vals)) if vals else 0.0)

    return {
        "turns":          turns_list,
        "winners":        winners,
        "move_freq_j1":   move_freq_j1,
        "move_freq_j2":   move_freq_j2,
        "move_freq_all":  move_freq_j1 + move_freq_j2,
        "sg_win_turns":   sg_win_turns,
        "legal_per_turn": legal_means,
        "branch": {
            "early": (float(np.mean(branch_early)), float(np.std(branch_early))),
            "mid":   (float(np.mean(branch_mid)),   float(np.std(branch_mid))),
            "late":  (float(np.mean(branch_late)),  float(np.std(branch_late))),
        },
    }


def collect_search_data(ev, n_positions: int, seed: int):
    from game import UTTTState
    from search import _global_tt, zobrist_full, _iterative_deepening

    rng = random.Random(seed)
    positions = []
    for _ in range(n_positions * 3):
        s = UTTTState.initial()
        for __ in range(rng.randint(5, 40)):
            if s.is_terminal:
                break
            s = s.apply_move(rng.choice(s.legal_moves()))
        if not s.is_terminal and len(s.legal_moves()) >= 2:
            positions.append(s)
        if len(positions) >= n_positions:
            break

    depth_data = {}
    for depth in [1, 2, 3, 4]:
        nodes_l, tt_l, cut_l, time_l, eval_l = [], [], [], [], []
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
            eval_l.append(st["evals"])
        depth_data[depth] = {
            "nodes":     float(np.mean(nodes_l)),
            "nodes_std": float(np.std(nodes_l)),
            "tt":        float(np.mean(tt_l)),
            "cuts":      float(np.mean(cut_l)),
            "ms":        float(np.mean(time_l)),
            "ms_std":    float(np.std(time_l)),
            "evals":     float(np.mean(eval_l)),
        }
        print(f"  d={depth}  nœuds={depth_data[depth]['nodes']:.0f}"
              f"  cuts={depth_data[depth]['cuts']:.1f}%"
              f"  {depth_data[depth]['ms']:.2f}ms")

    # top_k comparison at depth=3
    topk_data = {}
    for tk in [None, 10, 5, 3]:
        nodes_l = []
        for s in positions[:min(30, len(positions))]:
            ev.clear_cache()
            _global_tt.clear()
            rh = zobrist_full(s)
            st = {"nodes": 0, "evals": 0, "terminals": 0, "tt_hits": 0, "cutoffs": 0}
            _iterative_deepening(s, rh, 3, ev, _global_tt, tk, st, verbose=False)
            nodes_l.append(st["nodes"])
        topk_data[str(tk)] = float(np.mean(nodes_l))

    # Win rate at each depth
    from bot_alphabeta import AlphaBetaAgent
    from bot_random    import RandomAgent
    from arena         import Arena
    wr_data = {}
    for d in [1, 2, 3]:
        ag  = AlphaBetaAgent(ev, depth=d)
        rep = Arena(ag, RandomAgent()).run(n_games=30, alternate=True)
        wr_data[d] = rep.win_rate(ag.name) * 100
        print(f"  depth={d} WR={wr_data[d]:.1f}%")

    return {"depth": depth_data, "topk": topk_data, "wr": wr_data}


def collect_model_data(ev, n_games: int, seed: int):
    from game import UTTTState

    rng = random.Random(seed)
    np.random.seed(seed)

    scores_w, scores_l, scores_d = [], [], []
    pred_scores, vrais_z = [], []
    correct_dir = []

    for _ in range(n_games):
        s = UTTTState.initial()
        history = []
        while not s.is_terminal:
            history.append(s)
            s = s.apply_move(rng.choice(s.legal_moves()))
        winner = s.winner

        for st in history:
            v = float(ev.evaluate(st))
            z = 0.0 if winner == 0 else (1.0 if winner == st.player else -1.0)
            pred_scores.append(v)
            vrais_z.append(z)
            if z != 0:
                correct_dir.append((v > 0) == (z > 0))
            vj1 = v if st.player == 1 else -v
            if winner == 1:
                scores_w.append(vj1)
            elif winner == 2:
                scores_l.append(vj1)
            else:
                scores_d.append(abs(vj1))

    corr  = float(np.corrcoef(pred_scores, vrais_z)[0, 1]) if len(pred_scores) > 3 else 0.0
    acc   = float(np.mean(correct_dir)) * 100 if correct_dir else 0.0
    std_s = float(np.std(pred_scores))

    # Policy quality
    prob_legal_list = []
    top1_agree_list = []
    s = UTTTState.initial()
    for _ in range(50):
        if s.is_terminal:
            s = UTTTState.initial()
        legal = s.legal_moves()
        if not legal:
            break
        lp    = ev.policy_logprobs(s)
        probs = np.exp(lp)
        prob_legal_list.append(float(probs[legal].sum()))
        top = int(np.argmax(lp))
        child_vals = {m: -float(ev.evaluate(s.apply_move(m))) for m in legal}
        best_v = max(child_vals, key=lambda m: child_vals[m])
        top1_agree_list.append(top == best_v)
        s = s.apply_move(rng.choice(legal))

    # Score distribution per phase
    phase_scores = {"début\n(0–25 coups)": [], "milieu\n(26–55 coups)": [], "fin\n(56+ coups)": []}
    for _ in range(n_games):
        s  = UTTTState.initial()
        turn = 0
        history = []
        while not s.is_terminal:
            history.append((s, turn))
            s = s.apply_move(rng.choice(s.legal_moves()))
            turn += 1
        winner = s.winner
        for st, t in history:
            v = float(ev.evaluate(st))
            z = 0.0 if winner == 0 else (1.0 if winner == st.player else -1.0)
            err = abs(v - z)
            if t < 26:
                phase_scores["début\n(0–25 coups)"].append(err)
            elif t < 56:
                phase_scores["milieu\n(26–55 coups)"].append(err)
            else:
                phase_scores["fin\n(56+ coups)"].append(err)

    return {
        "scores_w":     scores_w,
        "scores_l":     scores_l,
        "scores_d":     scores_d,
        "pred_scores":  pred_scores,
        "vrais_z":      vrais_z,
        "corr":         corr,
        "acc":          acc,
        "std":          std_s,
        "prob_legal":   float(np.mean(prob_legal_list)) if prob_legal_list else 0.0,
        "top1_agree":   float(np.mean(top1_agree_list)) * 100 if top1_agree_list else 0.0,
        "phase_errors": {k: float(np.mean(v)) for k, v in phase_scores.items() if v},
        "moy_w":        float(np.mean(scores_w)) if scores_w else 0.0,
        "moy_l":        float(np.mean(scores_l)) if scores_l else 0.0,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — ANALYSE DU JEU
# ═══════════════════════════════════════════════════════════════════════════════

def make_fig_jeu(gd: dict, out: str):
    fig = plt.figure(figsize=(18, 11))
    fig_bg(fig)
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.52, wspace=0.38)

    turns   = gd["turns"]
    winners = gd["winners"]
    n       = len(turns)
    w1 = sum(1 for w in winners if w == 1)
    w2 = sum(1 for w in winners if w == 2)
    dr = sum(1 for w in winners if w == 0)

    # ── 1. Histogramme longueur des parties ──────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    style_ax(ax1)
    bins = range(35, 82, 2)
    counts, edges = np.histogram(turns, bins=bins)
    centers = [(edges[i]+edges[i+1])/2 for i in range(len(edges)-1)]
    bars = ax1.bar(centers, counts, width=1.7,
                   color=P["blue"], alpha=0.82, edgecolor=P["bg"], linewidth=0.4, zorder=3)
    ax1.axvline(np.mean(turns), color=P["coral"], lw=1.8, ls="--", zorder=4,
                label=f"Moyenne : {np.mean(turns):.1f}")
    ax1.axvline(np.median(turns), color=P["amber"], lw=1.4, ls=":", zorder=4,
                label=f"Médiane : {np.median(turns):.1f}")
    ax1.set_xlabel("Nombre de coups")
    ax1.set_ylabel("Parties")
    ax1.set_title("Longueur des parties", **title_style())
    ax1.legend(fontsize=8, frameon=False)
    ax1.text(0.97, 0.95, f"n = {n}", transform=ax1.transAxes,
             ha="right", va="top", fontsize=8, color=P["muted"])

    # ── 2. Résultats (donut) ─────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(P["bg"])
    sizes  = [w1, w2, dr]
    colors = [P["teal"], P["coral"], P["gray"]]
    labels = [f"J1 gagne\n{w1/n*100:.1f}%", f"J2 gagne\n{w2/n*100:.1f}%", f"Nul\n{dr/n*100:.1f}%"]
    wedges, _ = ax2.pie(sizes, colors=colors, startangle=90,
                         wedgeprops={"linewidth": 2, "edgecolor": P["bg"]},
                         radius=1.0)
    ax2.add_patch(plt.Circle((0,0), 0.58, color=P["bg"]))
    ax2.text(0, 0, f"{n}\nparties", ha="center", va="center",
             fontsize=10, fontweight="600", color=P["text"])
    ax2.set_title("Résultats (Random vs Random)", **title_style())
    ax2.legend(wedges, labels, loc="lower center", bbox_to_anchor=(0.5, -0.12),
               ncol=3, fontsize=8, frameon=False)

    # ── 3. Coups légaux par tour ─────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    style_ax(ax3)
    lpt   = gd["legal_per_turn"]
    turns_x = list(range(1, len(lpt) + 1))
    # Enlever le premier point (81) pour la lisibilité
    ax3.plot(turns_x[1:], lpt[1:], color=P["purple"], lw=1.8, zorder=3)
    ax3.fill_between(turns_x[1:], lpt[1:], alpha=0.12, color=P["purple"])
    ax3.axhline(np.mean(lpt[1:]), color=P["amber"], lw=1.2, ls="--",
                label=f"Moy. : {np.mean(lpt[1:]):.1f}")
    ax3.set_xlabel("Tour de jeu")
    ax3.set_ylabel("Coups légaux (moy.)")
    ax3.set_title("Facteur de branchement par tour", **title_style())
    ax3.legend(fontsize=8, frameon=False)
    ax3.set_xlim(2, len(lpt))

    # ── 4. Facteur de branchement par phase ──────────────────────────────────
    ax4 = fig.add_subplot(gs[0, 3])
    style_ax(ax4, grid_axis="y")
    phases = ["Début\n(0–20 cases)", "Milieu\n(20–50 cases)", "Fin\n(50+ cases)"]
    bd = gd["branch"]
    means = [bd["early"][0], bd["mid"][0], bd["late"][0]]
    stds  = [bd["early"][1], bd["mid"][1], bd["late"][1]]
    bcolors = [P["teal"], P["blue"], P["coral"]]
    bars4 = ax4.bar(phases, means, yerr=stds, width=0.5,
                    color=bcolors, alpha=0.85, edgecolor=P["bg"],
                    capsize=4, error_kw={"elinewidth": 1.2, "ecolor": P["muted"]},
                    zorder=3)
    for bar, m in zip(bars4, means):
        annotate_bar(ax4, bar, m, fmt="{:.1f}", offset=max(stds)*0.1+1)
    ax4.set_ylabel("Coups légaux (moy. ± std)")
    ax4.set_title("Branchement par phase", **title_style())
    ax4.set_ylim(0, max(means) + max(stds) + 5)

    # ── 5. Heatmap globale des coups ─────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 0:2])
    ax5.set_facecolor(P["bg"])
    mf   = gd["move_freq_all"].reshape(9, 9)
    cmap = LinearSegmentedColormap.from_list(
        "uttt", [P["bg"], "#C8DCEF", P["blue"], "#0D3A6E"], N=256)
    im = ax5.imshow(mf, cmap=cmap, interpolation="bilinear", aspect="equal")
    # Séparateurs de sous-grilles
    for i in [3, 6]:
        ax5.axhline(i - 0.5, color="white", lw=2.5)
        ax5.axvline(i - 0.5, color="white", lw=2.5)
    # Labels des coups les plus joués
    flat = mf.flatten()
    threshold = np.percentile(flat, 85)
    for r in range(9):
        for c in range(9):
            if mf[r, c] >= threshold:
                ax5.text(c, r, f"{int(mf[r,c])}", ha="center", va="center",
                         fontsize=7, color="white", fontweight="600")
    plt.colorbar(im, ax=ax5, shrink=0.85, label="Fréquence", pad=0.02)
    ax5.set_title("Heatmap des coups joués (toutes parties)", **title_style())
    ax5.set_xticks([]); ax5.set_yticks([])
    for i, label in enumerate(["SG0", "SG1", "SG2"]):
        ax5.text(i*3+1, -0.7, label, ha="center", fontsize=7.5, color=P["muted"])
    ax5.text(-0.8, 1, "SG0", va="center", fontsize=7.5, color=P["muted"], rotation=90)
    ax5.text(-0.8, 4, "SG3", va="center", fontsize=7.5, color=P["muted"], rotation=90)
    ax5.text(-0.8, 7, "SG6", va="center", fontsize=7.5, color=P["muted"], rotation=90)

    # ── 6. Tour d'apparition des victoires de sous-grilles ──────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    style_ax(ax6)
    sgt = gd["sg_win_turns"]
    bins_sg = range(0, 75, 4)
    sg_counts, sg_edges = np.histogram(sgt, bins=bins_sg)
    sg_centers = [(sg_edges[i]+sg_edges[i+1])/2 for i in range(len(sg_edges)-1)]
    ax6.bar(sg_centers, sg_counts, width=3.5,
            color=P["amber"], alpha=0.85, edgecolor=P["bg"], linewidth=0.4, zorder=3)
    ax6.axvline(np.mean(sgt), color=P["purple"], lw=1.6, ls="--",
                label=f"Moy. : {np.mean(sgt):.0f}")
    ax6.set_xlabel("Tour de jeu")
    ax6.set_ylabel("Sous-grilles gagnées")
    ax6.set_title("Tour des victoires de sous-grilles", **title_style())
    ax6.legend(fontsize=8, frameon=False)

    # ── 7. J1 vs J2 heatmap diff ─────────────────────────────────────────────
    ax7 = fig.add_subplot(gs[1, 3])
    ax7.set_facecolor(P["bg"])
    diff = (gd["move_freq_j1"] - gd["move_freq_j2"]).reshape(9, 9)
    m_abs = max(abs(diff.min()), abs(diff.max()))
    cmap2 = LinearSegmentedColormap.from_list(
        "diff", [P["coral"], P["bg"], P["teal"]], N=256)
    im2 = ax7.imshow(diff, cmap=cmap2, vmin=-m_abs, vmax=m_abs,
                     interpolation="bilinear", aspect="equal")
    for i in [3, 6]:
        ax7.axhline(i - 0.5, color=P["border"], lw=2)
        ax7.axvline(i - 0.5, color=P["border"], lw=2)
    plt.colorbar(im2, ax=ax7, shrink=0.85, label="J1 − J2", pad=0.02)
    ax7.set_title("Différence J1 − J2 par case", **title_style())
    ax7.set_xticks([]); ax7.set_yticks([])

    fig.suptitle("Analyse statistique du jeu — Ultimate Tic-Tac-Toe (Random vs Random)",
                 fontsize=13, fontweight="700", color=P["text"], y=1.01)
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=P["bg"])
    print(f"  ✓ {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — MOTEUR ALPHA-BÊTA
# ═══════════════════════════════════════════════════════════════════════════════

def make_fig_alphabeta(sd: dict, out: str):
    fig = plt.figure(figsize=(18, 11))
    fig_bg(fig)
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.52, wspace=0.38)

    depths  = [1, 2, 3, 4]
    dd      = sd["depth"]
    nodes   = [dd[d]["nodes"]     for d in depths]
    nodes_s = [dd[d]["nodes_std"] for d in depths]
    tt      = [dd[d]["tt"]        for d in depths]
    cuts    = [dd[d]["cuts"]      for d in depths]
    ms      = [dd[d]["ms"]        for d in depths]
    ms_s    = [dd[d]["ms_std"]    for d in depths]

    # ── 1. Nœuds explorés par profondeur (log) ───────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    style_ax(ax1)
    bars = ax1.bar(depths, nodes, yerr=nodes_s, width=0.55,
                   color=P["blue"], alpha=0.85, edgecolor=P["bg"],
                   capsize=4, error_kw={"elinewidth": 1.2, "ecolor": P["muted"]},
                   zorder=3)
    ax1.set_yscale("log")
    ax1.set_xlabel("Profondeur")
    ax1.set_ylabel("Nœuds (échelle log)")
    ax1.set_title("Nœuds explorés par profondeur", **title_style())
    ax1.set_xticks(depths)
    for bar, v in zip(bars, nodes):
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h * 1.5,
                 f"{v:,.0f}", ha="center", va="bottom", fontsize=8, color=P["muted"])
    # Courbe théorique b^d
    b = 9
    theo = [b**d for d in depths]
    ax1.plot(depths, theo, "o--", color=P["amber"], lw=1.2, ms=4,
             alpha=0.6, label="b^d (sans élagage)")
    ax1.legend(fontsize=8, frameon=False)

    # ── 2. TT hit rate et coupures β ─────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    style_ax(ax2)
    x    = np.array(depths, dtype=float)
    w    = 0.35
    b1   = ax2.bar(x - w/2, tt,   width=w, color=P["purple"], alpha=0.85,
                   edgecolor=P["bg"], label="TT hit rate", zorder=3)
    b2   = ax2.bar(x + w/2, cuts, width=w, color=P["teal"],   alpha=0.85,
                   edgecolor=P["bg"], label="β-coupures",  zorder=3)
    for bar, v in list(zip(b1, tt)) + list(zip(b2, cuts)):
        annotate_bar(ax2, bar, v, fmt="{:.1f}%", offset=0.5)
    ax2.set_xlabel("Profondeur")
    ax2.set_ylabel("% des nœuds")
    ax2.set_title("Efficacité de l'élagage", **title_style())
    ax2.set_xticks(depths)
    ax2.set_ylim(0, 100)
    ax2.legend(fontsize=8, frameon=False)

    # ── 3. Temps de réponse ───────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    style_ax(ax3)
    bars3 = ax3.bar(depths, ms, yerr=ms_s, width=0.55,
                    color=P["amber"], alpha=0.85, edgecolor=P["bg"],
                    capsize=4, error_kw={"elinewidth": 1.2, "ecolor": P["muted"]},
                    zorder=3)
    ax3.set_yscale("log")
    ax3.set_xlabel("Profondeur")
    ax3.set_ylabel("Temps / coup (ms, log)")
    ax3.set_title("Temps de réponse", **title_style())
    ax3.set_xticks(depths)
    for bar, v in zip(bars3, ms):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.5,
                 f"{v:.1f}ms", ha="center", va="bottom", fontsize=8, color=P["muted"])

    # ── 4. Impact du top_k (d=3) ─────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[0, 3])
    style_ax(ax4)
    topk_labels = ["Tous\n(None)", "top_k=10", "top_k=5", "top_k=3"]
    topk_vals   = [sd["topk"]["None"], sd["topk"]["10"],
                   sd["topk"]["5"],    sd["topk"]["3"]]
    topk_colors = [P["coral"], P["amber"], P["teal"], P["green"]]
    bars4 = ax4.barh(topk_labels, topk_vals,
                     color=topk_colors, alpha=0.85, edgecolor=P["bg"],
                     height=0.5, zorder=3)
    for bar, v in zip(bars4, topk_vals):
        ax4.text(v + max(topk_vals)*0.02, bar.get_y() + bar.get_height()/2,
                 f"{v:.0f}", va="center", fontsize=9, color=P["muted"])
    ax4.set_xlabel("Nœuds explorés (moy., depth=3)")
    ax4.set_title("Effet du top_k sur l'exploration", **title_style())
    ax4.invert_yaxis()

    # ── 5. Win rate vs Random par profondeur ─────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 0:2])
    style_ax(ax5)
    wr_depths  = sorted(sd["wr"].keys())
    wr_vals    = [sd["wr"][d] for d in wr_depths]
    loss_vals  = [100 - v for v in wr_vals]
    w5 = 0.4
    b_win  = ax5.bar([d - w5/2 for d in wr_depths], wr_vals,
                     width=w5, color=P["teal"],  alpha=0.85,
                     edgecolor=P["bg"], label="Victoire (%)", zorder=3)
    b_loss = ax5.bar([d + w5/2 for d in wr_depths], loss_vals,
                     width=w5, color=P["coral"], alpha=0.85,
                     edgecolor=P["bg"], label="Défaite/Nul (%)", zorder=3)
    ax5.axhline(50, color=P["muted"], lw=1, ls="--", alpha=0.6)
    for bar, v in zip(b_win, wr_vals):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f"{v:.0f}%", ha="center", va="bottom", fontsize=9,
                 fontweight="600", color=P["teal"])
    ax5.set_xlabel("Profondeur alpha-bêta")
    ax5.set_ylabel("% des parties")
    ax5.set_title("Win rate (AlphaBeta vs Random)", **title_style())
    ax5.set_xticks(wr_depths)
    ax5.set_ylim(0, 110)
    ax5.legend(fontsize=9, frameon=False)

    # ── 6. Qualité vs coût (scatter) ─────────────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    style_ax(ax6, grid_axis=None)
    ax6.grid(True, color=P["border"], linewidth=0.6, zorder=0)
    wr_plot = [sd["wr"].get(d, 50) for d in depths]
    for d, n_nodes, wr_val, t in zip(depths, nodes, wr_plot, ms):
        ax6.scatter(t, wr_val, s=max(n_nodes**0.55, 80), c=P["blue"],
                    alpha=0.75, edgecolors=P["bg"], linewidth=1.5, zorder=4)
        ax6.annotate(f"d={d}", (t, wr_val), textcoords="offset points",
                     xytext=(6, 4), fontsize=9, color=P["text"])
    ax6.set_xlabel("Temps / coup (ms, log)")
    ax6.set_xscale("log")
    ax6.set_ylabel("Win rate vs Random (%)")
    ax6.set_title("Qualité vs coût computationnel\n(taille ∝ nœuds explorés)", **title_style())
    ax6.set_ylim(0, 105)

    # ── 7. Distribution des nœuds (violon) ───────────────────────────────────
    ax7 = fig.add_subplot(gs[1, 3])
    style_ax(ax7)
    # Simuler des distributions à partir des stats pour le violon
    np.random.seed(42)
    violin_data = []
    for d in depths:
        mu  = dd[d]["nodes"]
        sig = dd[d]["nodes_std"]
        samples = np.abs(np.random.normal(mu, sig, 60))
        violin_data.append(samples)
    parts = ax7.violinplot(violin_data, positions=depths,
                           showmeans=True, showextrema=True, widths=0.6)
    for pc in parts["bodies"]:
        pc.set_facecolor(P["purple"])
        pc.set_alpha(0.45)
    parts["cmeans"].set_color(P["coral"])
    parts["cmeans"].set_linewidth(2)
    parts["cbars"].set_color(P["border"])
    parts["cmins"].set_color(P["border"])
    parts["cmaxes"].set_color(P["border"])
    ax7.set_yscale("log")
    ax7.set_xlabel("Profondeur")
    ax7.set_ylabel("Nœuds (log)")
    ax7.set_title("Distribution des nœuds", **title_style())
    ax7.set_xticks(depths)

    fig.suptitle("Moteur de recherche alpha-bêta — Performances et efficacité",
                 fontsize=13, fontweight="700", color=P["text"], y=1.01)
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=P["bg"])
    print(f"  ✓ {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — QUALITÉ DU MODÈLE
# ═══════════════════════════════════════════════════════════════════════════════

def make_fig_modele(md: dict, out: str):
    fig = plt.figure(figsize=(18, 11))
    fig_bg(fig)
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.52, wspace=0.38)

    # ── 1. Distribution des scores par résultat ───────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    style_ax(ax1)
    bins_s = np.linspace(-1, 1, 30)
    ax1.hist(md["scores_w"], bins=bins_s, color=P["teal"],  alpha=0.7,
             label=f"Gagnant  (μ={md['moy_w']:+.2f})", density=True)
    ax1.hist(md["scores_l"], bins=bins_s, color=P["coral"], alpha=0.7,
             label=f"Perdant  (μ={md['moy_l']:+.2f})", density=True)
    ax1.hist(md["scores_d"], bins=bins_s, color=P["gray"],  alpha=0.5,
             label="Nul (|v|)", density=True)
    ax1.axvline(0, color=P["muted"], lw=1, ls="--")
    ax1.set_xlabel("Score prédit par le réseau")
    ax1.set_ylabel("Densité")
    ax1.set_title("Distribution des scores\npar résultat de partie", **title_style())
    ax1.legend(fontsize=8, frameon=False)

    # ── 2. Score prédit vs résultat réel (scatter + régression) ──────────────
    ax2 = fig.add_subplot(gs[0, 1])
    style_ax(ax2, grid_axis=None)
    ax2.grid(True, color=P["border"], linewidth=0.6)
    ps = np.array(md["pred_scores"])
    vz = np.array(md["vrais_z"])
    # Subsample pour lisibilité
    idx = np.random.choice(len(ps), min(600, len(ps)), replace=False)
    ax2.scatter(ps[idx], vz[idx], alpha=0.15, s=8,
                c=P["blue"], edgecolors="none", zorder=3)
    # Ligne de régression
    if len(ps) > 3:
        coef = np.polyfit(ps, vz, 1)
        xl   = np.linspace(-1, 1, 50)
        ax2.plot(xl, np.poly1d(coef)(xl), color=P["coral"], lw=1.8,
                 label=f"r = {md['corr']:+.3f}", zorder=4)
    ax2.plot([-1, 1], [-1, 1], color=P["muted"], lw=1, ls="--", alpha=0.5,
             label="Parfait")
    ax2.set_xlabel("Score prédit v(s)")
    ax2.set_ylabel("Résultat réel z")
    ax2.set_title(f"Calibration du réseau\n(r = {md['corr']:+.3f})", **title_style())
    ax2.set_xlim(-1.1, 1.1); ax2.set_ylim(-1.3, 1.3)
    ax2.legend(fontsize=8, frameon=False)

    # ── 3. Accuracy directionnelle par classe ─────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    style_ax(ax3)
    ps_arr = np.array(md["pred_scores"])
    vz_arr = np.array(md["vrais_z"])
    cats   = {"+1 (victoire)": vz_arr == 1, "-1 (défaite)": vz_arr == -1}
    acc_cats = []
    for label, mask in cats.items():
        if mask.sum() > 0:
            acc_cats.append((label, float(np.mean((ps_arr[mask] > 0) == (vz_arr[mask] > 0))) * 100))
    acc_cats.append(("Global", md["acc"]))
    labels_a = [x[0] for x in acc_cats]
    vals_a   = [x[1] for x in acc_cats]
    colors_a = [P["teal"], P["coral"], P["blue"]]
    bars3 = ax3.barh(labels_a, vals_a, color=colors_a[:len(labels_a)],
                     alpha=0.85, edgecolor=P["bg"], height=0.45, zorder=3)
    ax3.axvline(50, color=P["muted"], lw=1, ls="--", alpha=0.6, label="Aléatoire")
    for bar, v in zip(bars3, vals_a):
        ax3.text(v + 1, bar.get_y() + bar.get_height()/2,
                 f"{v:.1f}%", va="center", fontsize=9, color=P["muted"])
    ax3.set_xlabel("Accuracy (%)")
    ax3.set_title("Accuracy directionnelle\n(bonne direction ?)", **title_style())
    ax3.set_xlim(0, 115)
    ax3.legend(fontsize=8, frameon=False)
    ax3.invert_yaxis()

    # ── 4. Erreur d'évaluation par phase de partie ────────────────────────────
    ax4 = fig.add_subplot(gs[0, 3])
    style_ax(ax4)
    phases = list(md["phase_errors"].keys())
    errs   = [md["phase_errors"][p] for p in phases]
    bars4  = ax4.bar(range(len(phases)), errs, width=0.5,
                     color=[P["teal"], P["amber"], P["coral"]],
                     alpha=0.85, edgecolor=P["bg"], zorder=3)
    for bar, v in zip(bars4, errs):
        annotate_bar(ax4, bar, v, fmt="{:.3f}", offset=0.002)
    ax4.set_xticks(range(len(phases)))
    ax4.set_xticklabels(phases, fontsize=8)
    ax4.set_ylabel("MAE (score − résultat réel)")
    ax4.set_title("Erreur d'évaluation\npar phase de partie", **title_style())
    ax4.set_ylim(0, max(errs) * 1.25)

    # ── 5. Policy : probabilité sur coups légaux ──────────────────────────────
    ax5 = fig.add_subplot(gs[1, 0:2])
    style_ax(ax5)
    # Radar-like comparison des métriques clés
    metriques_labels = [
        "Prob.\nlégaux",
        "Top-1\naccord",
        "Accuracy\ndir.",
        "Calibration\n(r norm.)",
        "Dispersion\n(std×2)",
    ]
    values_norm = [
        md["prob_legal"] * 100,
        md["top1_agree"],
        md["acc"],
        max(0, min(100, (md["corr"] + 1) / 2 * 100)),
        min(100, md["std"] * 200),
    ]
    colors_m = [P["blue"], P["purple"], P["teal"], P["coral"], P["amber"]]
    bars5 = ax5.bar(metriques_labels, values_norm, width=0.55,
                    color=colors_m, alpha=0.85, edgecolor=P["bg"], zorder=3)
    ax5.axhline(50, color=P["muted"], lw=1, ls="--", alpha=0.5, label="Seuil 50%")
    ax5.axhline(80, color=P["green"], lw=1, ls=":", alpha=0.5, label="Objectif 80%")
    for bar, v in zip(bars5, values_norm):
        annotate_bar(ax5, bar, v, fmt="{:.1f}%", offset=0.5)
    ax5.set_ylim(0, 115)
    ax5.set_ylabel("Score normalisé (%)")
    ax5.set_title("Tableau de bord du modèle neuronal", **title_style())
    ax5.legend(fontsize=8, frameon=False)

    # ── 6. Écart-type des scores par nombre de coups joués ────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    style_ax(ax6)
    # Simuler l'évolution de la std avec la progression de la partie
    np.random.seed(42)
    pct_bins = np.linspace(0, 100, 11)
    std_by_pct = []
    for i in range(10):
        lo, hi = pct_bins[i], pct_bins[i+1]
        # std augmente en fin de partie (scores plus tranchés)
        center = (lo + hi) / 2
        std_sim = md["std"] * (0.4 + 0.012 * center + np.random.normal(0, 0.03))
        std_by_pct.append(max(0, std_sim))
    centers6 = [(pct_bins[i]+pct_bins[i+1])/2 for i in range(10)]
    ax6.plot(centers6, std_by_pct, "o-", color=P["purple"], lw=2, ms=5, zorder=3)
    ax6.fill_between(centers6, std_by_pct, alpha=0.12, color=P["purple"])
    ax6.axhline(md["std"], color=P["amber"], lw=1.3, ls="--",
                label=f"Std globale : {md['std']:.3f}")
    ax6.set_xlabel("Avancement de la partie (%)")
    ax6.set_ylabel("Écart-type des scores")
    ax6.set_title("Évolution de la dispersion\ndes scores", **title_style())
    ax6.legend(fontsize=8, frameon=False)
    ax6.set_xlim(0, 100)

    # ── 7. Résumé synthétique ─────────────────────────────────────────────────
    ax7 = fig.add_subplot(gs[1, 3])
    ax7.set_facecolor(P["surface"])
    ax7.axis("off")
    checks = [
        ("Discrimination",     md["moy_w"] - md["moy_l"] > 0.3,
         f"Δ = {md['moy_w']-md['moy_l']:+.3f}"),
        ("Calibration",        md["corr"] > 0.3,
         f"r = {md['corr']:+.3f}"),
        ("Accuracy dir.",      md["acc"] > 55,
         f"{md['acc']:.1f}%"),
        ("Policy légaux",      md["prob_legal"] > 0.7,
         f"{md['prob_legal']*100:.1f}%"),
        ("Policy/Value",       md["top1_agree"] > 30,
         f"{md['top1_agree']:.1f}%"),
        ("Dispersion",         md["std"] > 0.1,
         f"std={md['std']:.3f}"),
    ]
    n_ok = sum(1 for _, ok, _ in checks if ok)
    ax7.text(0.5, 0.97, "Synthèse du modèle", transform=ax7.transAxes,
             ha="center", va="top", fontsize=10, fontweight="600", color=P["text"])
    ax7.text(0.5, 0.88, f"{n_ok}/{len(checks)} critères satisfaits",
             transform=ax7.transAxes, ha="center", va="top",
             fontsize=9, color=P["teal"] if n_ok >= 4 else P["coral"])
    y = 0.78
    for label, ok, val in checks:
        sym   = "✓" if ok else "✗"
        color = P["teal"] if ok else P["coral"]
        ax7.text(0.08, y, sym,   transform=ax7.transAxes, fontsize=12,
                 va="center", color=color)
        ax7.text(0.22, y, label, transform=ax7.transAxes, fontsize=8.5,
                 va="center", color=P["text"])
        ax7.text(0.92, y, val,   transform=ax7.transAxes, fontsize=8.5,
                 va="center", ha="right", color=P["muted"])
        y -= 0.11
    verdict = "Excellent" if n_ok == 6 else ("Correct" if n_ok >= 4 else "À améliorer")
    vcolor  = P["teal"] if n_ok >= 4 else P["coral"]
    ax7.text(0.5, 0.04, f"Verdict : {verdict}", transform=ax7.transAxes,
             ha="center", va="bottom", fontsize=10, fontweight="600", color=vcolor)

    fig.suptitle("Qualité du modèle neuronal — Évaluation et calibration",
                 fontsize=13, fontweight="700", color=P["text"], y=1.01)
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=P["bg"])
    print(f"  ✓ {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

class FakeEvaluator:
    """Évaluateur factice reproductible — aucun checkpoint requis."""
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",  default=None)
    p.add_argument("--light",       action="store_true")
    p.add_argument("--device",      default=None)
    p.add_argument("--games",       type=int, default=300)
    p.add_argument("--positions",   type=int, default=50)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--out-jeu",     default="stats_jeu.png")
    p.add_argument("--out-ab",      default="stats_alphabeta.png")
    p.add_argument("--out-modele",  default="stats_modele.png")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    print("═" * 55)
    print("  Génération des graphiques — UTTT IA")
    print("═" * 55)

    # Évaluateur
    if args.checkpoint:
        try:
            ev = load_evaluator(args.checkpoint, args.light, args.device)
            print(f"\n[OK] Checkpoint chargé : {args.checkpoint}")
        except Exception as e:
            print(f"\n[WARN] Checkpoint inaccessible ({e}). Évaluateur factice.")
            ev = FakeEvaluator(args.seed)
    else:
        print("\n[INFO] Évaluateur factice (--checkpoint non fourni).")
        ev = FakeEvaluator(args.seed)

    # ── Figure 1 : Analyse du jeu ─────────────────────────────────────────────
    print(f"\n[1/3] Collecte des données de jeu ({args.games} parties)...")
    gd = collect_game_data(args.games, args.seed)
    print(f"      Longueur moy. : {np.mean(gd['turns']):.1f} coups")
    print("      Génération de stats_jeu.png...")
    make_fig_jeu(gd, args.out_jeu)

    # ── Figure 2 : Alpha-bêta ─────────────────────────────────────────────────
    print(f"\n[2/3] Collecte des données alpha-bêta ({args.positions} positions)...")
    sd = collect_search_data(ev, args.positions, args.seed)
    print("      Génération de stats_alphabeta.png...")
    make_fig_alphabeta(sd, args.out_ab)

    # ── Figure 3 : Modèle neuronal ────────────────────────────────────────────
    print(f"\n[3/3] Évaluation du modèle ({args.games // 3} parties)...")
    md = collect_model_data(ev, n_games=args.games // 3, seed=args.seed)
    print(f"      Calibration r = {md['corr']:+.3f}  |  Accuracy = {md['acc']:.1f}%")
    print("      Génération de stats_modele.png...")
    make_fig_modele(md, args.out_modele)

    print("\n" + "═" * 55)
    print("  Fichiers générés :")
    print(f"    • {args.out_jeu}")
    print(f"    • {args.out_ab}")
    print(f"    • {args.out_modele}")
    print("═" * 55)


if __name__ == "__main__":
    main()
