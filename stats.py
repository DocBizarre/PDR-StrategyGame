"""
stats.py — Module de statistiques pour Ultimate Tic-Tac-Toe IA
==============================================================
Fonctions autonomes et composables : chacune prend des données brutes
et retourne un dict prêt à être affiché ou exporté.

Groupes disponibles
───────────────────
  1. stats_parties()        — métriques sur un ensemble de parties
  2. stats_search()         — métriques du moteur alpha-bêta (nœuds, TT, coupures)
  3. stats_evaluateur()     — qualité du réseau neuronal (calibration, discrimination)
  4. stats_duels()          — comparaison de deux agents (win rate, Elo estimé)
  5. stats_temps()          — profilage des temps de réponse par profondeur
  6. conclusion()           — génère des conclusions textuelles à partir de métriques
  7. rapport_complet()      — lance tous les tests et retourne un dict consolidé

Usage minimal (sans checkpoint)
────────────────────────────────
  from stats import stats_parties, conclusion
  resultats = run_arena(...)          # liste de GameResult
  m = stats_parties(resultats)
  print(conclusion(m))

Usage complet (avec checkpoint)
────────────────────────────────
  from stats import rapport_complet
  from model import NeuralEvaluator
  ev     = NeuralEvaluator("models/best.pth")
  rapport = rapport_complet(ev, n_games=50, depth=3)
  rapport.print_all()
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from game       import UTTTState


# ══════════════════════════════════════════════════════════════════════════════
# TYPES DE DONNÉES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class StatResult:
    """
    Résultat d'un bloc de statistiques.
    Peut être imprimé, exporté en dict, ou agrégé avec d'autres.
    """
    titre:      str
    metriques:  Dict[str, Any]          = field(default_factory=dict)
    details:    List[Dict[str, Any]]    = field(default_factory=list)
    conclusions: List[str]              = field(default_factory=list)
    verdict:    str                     = ""      # "excellent" | "correct" | "faible"

    def print(self, large: bool = False) -> None:
        W = 62
        print(f"\n{'═'*W}")
        print(f"  {self.titre}")
        print(f"{'─'*W}")
        for k, v in self.metriques.items():
            if isinstance(v, float):
                print(f"  {k:<36} {v:>10.4f}")
            else:
                print(f"  {k:<36} {str(v):>10}")
        if self.details and large:
            print(f"{'─'*W}")
            for d in self.details:
                ok_sym = "✓" if d.get("ok", True) else "✗"
                print(f"  {ok_sym} {d.get('label',''):<40} {d.get('valeur','')}")
        if self.conclusions:
            print(f"{'─'*W}")
            for c in self.conclusions:
                print(f"  → {c}")
        if self.verdict:
            sym = {"excellent":"✓✓", "correct":"✓", "faible":"✗"}.get(self.verdict, "·")
            print(f"\n  Verdict : {sym} {self.verdict.upper()}")
        print(f"{'═'*W}")

    def to_dict(self) -> dict:
        return {
            "titre":      self.titre,
            "metriques":  self.metriques,
            "conclusions": self.conclusions,
            "verdict":    self.verdict,
        }


# ══════════════════════════════════════════════════════════════════════════════
# 1. STATISTIQUES SUR LES PARTIES
# ══════════════════════════════════════════════════════════════════════════════

def stats_parties(
    results: list,              # liste de GameResult (depuis arena.py)
    agent_a: str = "",          # nom de l'agent de référence
) -> StatResult:
    """
    Analyse un ensemble de GameResult produits par Arena.

    Métriques calculées
    ───────────────────
    - win_rate, draw_rate, loss_rate
    - win_rate_as_j1, win_rate_as_j2  (biais de premier joueur)
    - avg_turns, std_turns, min_turns, max_turns
    - avg_time_j1_ms, avg_time_j2_ms

    Paramètres
    ──────────
    results  : list[GameResult] — résultats de Arena.run()
    agent_a  : str              — nom de l'agent de référence (optionnel)
    """
    if not results:
        return StatResult("Statistiques des parties", verdict="faible",
                          conclusions=["Aucune partie fournie."])

    n = len(results)
    if not agent_a and hasattr(results[0], "name_j1"):
        agent_a = results[0].name_j1

    wins   = sum(1 for r in results if r.winner_name == agent_a)
    draws  = sum(1 for r in results if r.winner == 0)
    losses = n - wins - draws
    turns_list = [r.turns for r in results]

    # Win rate selon la couleur
    games_j1 = [r for r in results if r.name_j1 == agent_a]
    games_j2 = [r for r in results if r.name_j2 == agent_a]
    wr_j1 = sum(1 for r in games_j1 if r.winner == 1) / max(len(games_j1), 1) * 100
    wr_j2 = sum(1 for r in games_j2 if r.winner == 2) / max(len(games_j2), 1) * 100

    # Temps par partie
    times_a = [r.time_j1 if r.name_j1 == agent_a else r.time_j2 for r in results]

    m = {
        "Parties jouées":                n,
        "Win rate (%)":                  round(wins  / n * 100, 1),
        "Draw rate (%)":                 round(draws / n * 100, 1),
        "Loss rate (%)":                 round(losses / n * 100, 1),
        "Win rate jouant J1 (%)":        round(wr_j1, 1),
        "Win rate jouant J2 (%)":        round(wr_j2, 1),
        "Avantage J1 (points)":          round(wr_j1 - wr_j2, 1),
        "Coups / partie (moy.)":         round(float(np.mean(turns_list)), 1),
        "Coups / partie (std)":          round(float(np.std(turns_list)),  1),
        "Coups / partie (min)":          int(np.min(turns_list)),
        "Coups / partie (max)":          int(np.max(turns_list)),
        "Temps total agent (s)":         round(sum(times_a), 2),
        "Temps / partie agent (ms)":     round(np.mean(times_a) * 1000, 1),
    }

    # Conclusions automatiques
    conc = []
    wr = m["Win rate (%)"]
    if wr >= 80:
        conc.append(f"Win rate excellent ({wr}%) — l'agent domine nettement.")
    elif wr >= 55:
        conc.append(f"Win rate correct ({wr}%) — l'agent est meilleur que l'adversaire.")
    elif wr >= 45:
        conc.append(f"Win rate équilibré ({wr}%) — les agents sont de force similaire.")
    else:
        conc.append(f"Win rate faible ({wr}%) — l'agent est dominé.")

    av = m["Avantage J1 (points)"]
    if abs(av) > 10:
        conc.append(f"Biais premier joueur marqué (+{av:.1f} pts) — J1 a un avantage structurel.")
    else:
        conc.append(f"Biais premier joueur faible ({av:+.1f} pts) — le jeu est équilibré par couleur.")

    verdict = "excellent" if wr >= 75 else ("correct" if wr >= 50 else "faible")
    return StatResult(
        titre       = f"Statistiques des parties ({n} parties, agent={agent_a or '?'})",
        metriques   = m,
        conclusions = conc,
        verdict     = verdict,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 2. STATISTIQUES DU MOTEUR ALPHA-BÊTA
# ══════════════════════════════════════════════════════════════════════════════

def stats_search(
    evaluator,
    positions:      Optional[List[UTTTState]] = None,
    depths:         List[int]                 = (1, 2, 3, 4),
    n_positions:    int                       = 30,
    top_k:          Optional[int]             = None,
    seed:           int                       = 42,
) -> StatResult:
    """
    Mesure les performances du moteur alpha-bêta sur un ensemble de positions.

    Métriques calculées (par profondeur)
    ─────────────────────────────────────
    - nœuds explorés (moy., std)
    - TT hit rate
    - taux de coupures β
    - temps de réponse (ms)
    - facteur de branchement effectif

    Paramètres
    ──────────
    evaluator  : objet avec .evaluate() et .policy_logprobs()
    positions  : list[UTTTState] — si None, générées aléatoirement
    depths     : profondeurs à tester
    n_positions: nombre de positions si génération aléatoire
    top_k      : limite de largeur (None = tous les coups)
    seed       : reproductibilité
    """
    from search import _global_tt, zobrist_full, _iterative_deepening

    rng = random.Random(seed)

    # Génère des positions si non fournies
    if positions is None:
        positions = _generer_positions(n_positions, rng)

    positions = [p for p in positions if not p.is_terminal and len(p.legal_moves()) >= 2]
    if not positions:
        return StatResult("Statistiques de recherche", verdict="faible",
                          conclusions=["Aucune position valide fournie."])

    details_par_depth = {}
    for depth in depths:
        nodes_l, tt_l, cut_l, time_l = [], [], [], []

        for s in positions:
            evaluator.clear_cache()
            _global_tt.clear()
            root_hash = zobrist_full(s)
            st = {"nodes": 0, "evals": 0, "terminals": 0, "tt_hits": 0, "cutoffs": 0}
            t0 = time.perf_counter()
            _iterative_deepening(s, root_hash, depth, evaluator, _global_tt, top_k, st, verbose=False)
            elapsed = (time.perf_counter() - t0) * 1000

            nodes_l.append(st["nodes"])
            tt_l.append(st["tt_hits"] / max(st["nodes"], 1) * 100)
            cut_l.append(st["cutoffs"] / max(st["nodes"], 1) * 100)
            time_l.append(elapsed)

        details_par_depth[depth] = {
            "noeuds_moy":   float(np.mean(nodes_l)),
            "noeuds_std":   float(np.std(nodes_l)),
            "tt_hit_pct":   float(np.mean(tt_l)),
            "cuts_pct":     float(np.mean(cut_l)),
            "time_ms_moy":  float(np.mean(time_l)),
            "time_ms_std":  float(np.std(time_l)),
        }

    # Métriques principales (profondeur max)
    d_max   = max(depths)
    dm      = details_par_depth[d_max]
    d_first = details_par_depth[min(depths)]
    ebf = (dm["noeuds_moy"] / max(d_first["noeuds_moy"], 1)) ** (1 / max(d_max - min(depths), 1))

    metriques = {
        "Positions testées":             len(positions),
        "Profondeurs testées":           str(list(depths)),
        f"Nœuds moy. (depth={d_max})":  round(dm["noeuds_moy"], 0),
        f"Nœuds std (depth={d_max})":   round(dm["noeuds_std"], 0),
        f"TT hit rate (depth={d_max}) %": round(dm["tt_hit_pct"], 1),
        f"β-coupures (depth={d_max}) %": round(dm["cuts_pct"], 1),
        f"Temps moy. (depth={d_max}) ms": round(dm["time_ms_moy"], 1),
        "Facteur de branchement effectif": round(ebf, 2),
    }

    # Détails par profondeur (pour affichage large)
    details = []
    for d in sorted(depths):
        dd = details_par_depth[d]
        details.append({
            "label":  f"depth={d}",
            "valeur": (f"nœuds={dd['noeuds_moy']:.0f}  "
                       f"TT={dd['tt_hit_pct']:.1f}%  "
                       f"cuts={dd['cuts_pct']:.1f}%  "
                       f"{dd['time_ms_moy']:.1f}ms"),
            "ok": True,
            "_raw": dd,
        })

    # Conclusions
    conc = []
    tt = dm["tt_hit_pct"]
    cu = dm["cuts_pct"]
    if tt > 30:
        conc.append(f"TT hit rate élevé ({tt:.1f}%) — la table de transposition est efficace.")
    else:
        conc.append(f"TT hit rate faible ({tt:.1f}%) — peu de positions revisitées, normal en début de partie.")
    if cu > 50:
        conc.append(f"Taux de coupures excellent ({cu:.1f}%) — le move ordering neuronal est efficace.")
    elif cu > 25:
        conc.append(f"Taux de coupures correct ({cu:.1f}%).")
    else:
        conc.append(f"Taux de coupures faible ({cu:.1f}%) — le move ordering peut être amélioré.")
    conc.append(f"Facteur de branchement effectif : {ebf:.2f} (idéal < racine_b pour alpha-bêta parfait).")

    verdict = "excellent" if tt > 30 and cu > 50 else ("correct" if cu > 20 else "faible")
    return StatResult(
        titre       = f"Statistiques de recherche alpha-bêta (top_k={top_k or 'tous'})",
        metriques   = metriques,
        details     = details,
        conclusions = conc,
        verdict     = verdict,
        # On stocke le détail brut pour accès programmatique
    )


def stats_search_par_depth(
    evaluator,
    depths:      List[int]       = (1, 2, 3, 4),
    n_positions: int             = 30,
    top_k:       Optional[int]   = None,
    seed:        int             = 42,
) -> Dict[int, dict]:
    """
    Version légère de stats_search() qui retourne directement un dict
    {depth → {noeuds, tt, cuts, time_ms}} pour alimenter des graphiques.
    """
    r = stats_search(evaluator, depths=depths, n_positions=n_positions,
                     top_k=top_k, seed=seed)
    # Extraire les détails bruts depuis les details
    out = {}
    for d in r.details:
        depth_key = int(d["label"].replace("depth=", ""))
        out[depth_key] = d["_raw"]
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 3. STATISTIQUES DE L'ÉVALUATEUR NEURONAL
# ══════════════════════════════════════════════════════════════════════════════

def stats_evaluateur(
    evaluator,
    n_games:  int = 40,
    n_states: int = 50,
    seed:     int = 42,
) -> StatResult:
    """
    Évalue la qualité du réseau neuronal (value head + policy head).

    Métriques calculées
    ───────────────────
    - séparation gagnant/perdant/nul (discrimination)
    - corrélation score ↔ résultat réel (calibration)
    - accuracy directionnelle (prédit-on la bonne direction ?)
    - symétrie joueur (v(J1) ≈ -v(J2) pour même position)
    - probabilité sur coups légaux (policy head)
    - accord top-1 policy / value (cohérence)
    - std des scores (dispersion)

    Paramètres
    ──────────
    evaluator : objet avec .evaluate() et .policy_logprobs()
    n_games   : parties aléatoires pour les tests de calibration
    n_states  : états pour le test policy
    seed      : graine de reproductibilité
    """
    rng = random.Random(seed)
    np.random.seed(seed)

    # ── A. Discrimination (séparation gagnant / perdant / nul) ──────────────
    scores_w, scores_l, scores_d = [], [], []
    for _ in range(n_games):
        s_hist, winner = _jouer_partie_aleatoire(rng)
        if len(s_hist) < 4:
            continue
        for s in s_hist[-5:-1]:   # 4 états proches de la fin
            v = float(evaluator.evaluate(s))
            if winner == 0:
                scores_d.append(abs(v))
            elif winner == s.player:
                scores_w.append(v)
            else:
                scores_l.append(v)

    moy_w = float(np.mean(scores_w)) if scores_w else 0.0
    moy_l = float(np.mean(scores_l)) if scores_l else 0.0
    moy_d = float(np.mean(scores_d)) if scores_d else 0.0
    sep   = moy_w - moy_l

    # ── B. Calibration (corrélation score ↔ résultat) ───────────────────────
    pred_scores, vrais_z = [], []
    all_scores = []
    for _ in range(n_games):
        s_hist, winner = _jouer_partie_aleatoire(rng)
        for s in s_hist[:-1]:
            v  = float(evaluator.evaluate(s))
            z  = 0.0 if winner == 0 else (1.0 if winner == s.player else -1.0)
            pred_scores.append(v)
            vrais_z.append(z)
            all_scores.append(v)

    corr = float(np.corrcoef(pred_scores, vrais_z)[0, 1]) if len(pred_scores) > 2 else 0.0
    mse  = float(np.mean([(p - z)**2 for p, z in zip(pred_scores, vrais_z)]))
    std  = float(np.std(all_scores))

    # ── C. Accuracy directionnelle ───────────────────────────────────────────
    correct_dir = []
    for _ in range(n_games):
        s_hist, winner = _jouer_partie_aleatoire(rng)
        for s in s_hist[:-1]:
            v = float(evaluator.evaluate(s))
            z = 0.0 if winner == 0 else (1.0 if winner == s.player else -1.0)
            if z != 0.0:
                correct_dir.append((v > 0) == (z > 0))
    acc = float(np.mean(correct_dir)) * 100 if correct_dir else 0.0

    # ── D. Symétrie joueur ───────────────────────────────────────────────────
    sym_errs = []
    for _ in range(20):
        s = UTTTState.initial()
        for __ in range(rng.randint(3, 15)):
            if s.is_terminal: break
            s = s.apply_move(rng.choice(s.legal_moves()))
        if s.is_terminal: continue
        # Même position, joueur alterné
        import copy
        s2       = copy.copy(s)
        s2.board = s2.board.copy()
        s2.meta_board = s2.meta_board.copy()
        s2.player = 3 - s.player
        v1 = float(evaluator.evaluate(s))
        v2 = float(evaluator.evaluate(s2))
        sym_errs.append(abs(v1 + v2))   # idéal → 0 (v(J2) = -v(J1))
    sym_err = float(np.mean(sym_errs)) if sym_errs else 1.0

    # ── E. Policy head ───────────────────────────────────────────────────────
    prob_legal_list, top1_agree_list = [], []
    s = UTTTState.initial()
    for _ in range(n_states):
        if s.is_terminal:
            s = UTTTState.initial()
        legal = s.legal_moves()
        if not legal:
            break
        lp    = evaluator.policy_logprobs(s)
        probs = np.exp(lp)
        prob_legal_list.append(float(probs[legal].sum()))

        top_policy = int(np.argmax(lp))
        child_vals = {m: -float(evaluator.evaluate(s.apply_move(m))) for m in legal}
        best_val_move = max(child_vals, key=lambda m: child_vals[m])
        top1_agree_list.append(top_policy == best_val_move)

        s = s.apply_move(rng.choice(legal))

    prob_legal = float(np.mean(prob_legal_list)) if prob_legal_list else 0.0
    top1_agree = float(np.mean(top1_agree_list)) * 100 if top1_agree_list else 0.0

    # ── Agrégation ──────────────────────────────────────────────────────────
    metriques = {
        "Score moyen gagnant":            round(moy_w,      4),
        "Score moyen perdant":            round(moy_l,      4),
        "Score moyen nul (|v|)":          round(moy_d,      4),
        "Séparation gagnant-perdant":     round(sep,        4),
        "Corrélation score ↔ résultat":   round(corr,       4),
        "MSE score ↔ résultat":           round(mse,        4),
        "Std des scores (dispersion)":    round(std,        4),
        "Accuracy directionnelle (%)":    round(acc,        1),
        "Erreur symétrie joueur (|v1+v2|)": round(sym_err,  4),
        "Prob. sur coups légaux":         round(prob_legal, 4),
        "Accord top-1 policy/value (%)":  round(top1_agree, 1),
    }

    # Détails
    details = [
        {"label": "Discrimination", "ok": sep > 0.5,
         "valeur": f"séparation={sep:.4f} (idéal > 0.5)"},
        {"label": "Calibration",    "ok": corr > 0.4,
         "valeur": f"corrélation={corr:.4f} (idéal > 0.4)"},
        {"label": "Accuracy",       "ok": acc > 60,
         "valeur": f"{acc:.1f}% (idéal > 60%)"},
        {"label": "Symétrie joueur","ok": sym_err < 0.3,
         "valeur": f"erreur={sym_err:.4f} (idéal < 0.3)"},
        {"label": "Policy légaux",  "ok": prob_legal > 0.85,
         "valeur": f"{prob_legal:.4f} (idéal > 0.85)"},
        {"label": "Policy/Value accord","ok": top1_agree > 40,
         "valeur": f"{top1_agree:.1f}% (idéal > 40%)"},
    ]

    # Conclusions
    conc = []
    if sep > 1.0:
        conc.append(f"Discrimination excellente (séparation={sep:.2f}) — le réseau reconnaît les positions gagnantes.")
    elif sep > 0.5:
        conc.append(f"Discrimination correcte (séparation={sep:.2f}).")
    else:
        conc.append(f"Discrimination faible (séparation={sep:.2f}) — le réseau ne distingue pas bien gagnant/perdant.")

    if corr > 0.6:
        conc.append(f"Calibration excellente (r={corr:.2f}) — les scores prédisent bien les résultats.")
    elif corr > 0.35:
        conc.append(f"Calibration acceptable (r={corr:.2f}).")
    else:
        conc.append(f"Calibration faible (r={corr:.2f}) — continuer l'entraînement.")

    if sym_err < 0.15:
        conc.append("Symétrie parfaite — le réseau distingue correctement J1 et J2.")
    elif sym_err < 0.4:
        conc.append(f"Symétrie partielle (err={sym_err:.2f}) — le réseau perçoit partiellement le joueur courant.")
    else:
        conc.append(f"Symétrie absente (err={sym_err:.2f}) — le réseau ignore le joueur courant (canal 19 non utilisé).")

    if prob_legal > 0.9:
        conc.append("Policy head concentrée sur les coups légaux — apprentissage réussi.")
    else:
        conc.append(f"Policy dispersée sur coups illégaux ({1-prob_legal:.1%}) — entraînement insuffisant.")

    n_ok    = sum(1 for d in details if d["ok"])
    verdict = "excellent" if n_ok >= 5 else ("correct" if n_ok >= 3 else "faible")
    return StatResult(
        titre       = f"Qualité de l'évaluateur neuronal ({n_games} parties, {n_states} états)",
        metriques   = metriques,
        details     = details,
        conclusions = conc,
        verdict     = verdict,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 4. STATISTIQUES DE DUEL (Elo + détail par couleur)
# ══════════════════════════════════════════════════════════════════════════════

def stats_duels(
    agent_a,
    agent_b,
    n_games:   int  = 30,
    alternate: bool = True,
    seed:      int  = 42,
) -> StatResult:
    """
    Confronte deux agents et produit des métriques de duel détaillées.

    Métriques calculées
    ───────────────────
    - win_rate, draw_rate, loss_rate
    - win_rate_as_j1, win_rate_as_j2 (avantage couleur)
    - Elo estimé (différentiel)
    - durée moyenne par coup (ms)
    - score moyen retourné par l'agent

    Paramètres
    ──────────
    agent_a / agent_b : agents avec .choose_move(state) → (move, score)
    n_games  : nombre de parties
    alternate : alterner les couleurs
    seed     : reproductibilité
    """
    from arena import Arena
    random.seed(seed)

    t_a, t_b = [], []
    scores_a = []

    # Wrappeur de chronométrage
    class _Timed:
        def __init__(self, ag):
            self.ag = ag
            self.name = ag.name
            self.times = []
            self.scores = []
        def choose_move(self, state):
            t0 = time.perf_counter()
            mv, sc = self.ag.choose_move(state)
            self.times.append(time.perf_counter() - t0)
            self.scores.append(sc)
            return mv, sc

    wa = _Timed(agent_a)
    wb = _Timed(agent_b)

    report = Arena(wa, wb).run(n_games=n_games, alternate=alternate)

    n    = report.n
    wins = report.wins(agent_a.name)
    drws = report.draws()
    loss = report.wins(agent_b.name)

    wr_j1_a = sum(1 for r in report._results
                  if r.name_j1 == agent_a.name and r.winner == 1)
    wr_j2_a = sum(1 for r in report._results
                  if r.name_j2 == agent_a.name and r.winner == 2)
    n_j1_a  = sum(1 for r in report._results if r.name_j1 == agent_a.name)
    n_j2_a  = sum(1 for r in report._results if r.name_j2 == agent_a.name)

    wr_pct  = wins / n * 100
    elo_diff = _elo_diff(wr_pct / 100)

    metriques = {
        "Parties jouées":               n,
        f"Win rate {agent_a.name} (%)": round(wr_pct, 1),
        "Draw rate (%)":                round(drws / n * 100, 1),
        f"Loss rate {agent_a.name} (%)" : round(loss / n * 100, 1),
        f"Win rate en J1 (%)":          round(wr_j1_a / max(n_j1_a, 1) * 100, 1),
        f"Win rate en J2 (%)":          round(wr_j2_a / max(n_j2_a, 1) * 100, 1),
        "Elo différentiel estimé":      round(elo_diff, 0),
        f"Temps / coup {agent_a.name} (ms)": round(np.mean(wa.times) * 1000, 1),
        f"Temps / coup {agent_b.name} (ms)": round(np.mean(wb.times) * 1000, 1),
        f"Score moyen retourné {agent_a.name}": round(float(np.mean(wa.scores)), 3) if wa.scores else 0.0,
    }

    conc = []
    if wr_pct >= 80:
        conc.append(f"{agent_a.name} est nettement supérieur (+{elo_diff:.0f} Elo estimé).")
    elif wr_pct >= 60:
        conc.append(f"{agent_a.name} est meilleur (+{elo_diff:.0f} Elo estimé).")
    elif wr_pct >= 45:
        conc.append(f"Résultat équilibré — différence Elo estimée : {elo_diff:+.0f}.")
    else:
        conc.append(f"{agent_a.name} est dominé par {agent_b.name} ({elo_diff:+.0f} Elo).")

    speed_ratio = np.mean(wb.times) / max(np.mean(wa.times), 1e-9)
    if speed_ratio > 5:
        conc.append(f"{agent_a.name} est {speed_ratio:.0f}× plus lent que {agent_b.name}.")
    elif speed_ratio < 0.2:
        conc.append(f"{agent_a.name} est {1/speed_ratio:.0f}× plus rapide que {agent_b.name}.")

    verdict = "excellent" if wr_pct >= 70 else ("correct" if wr_pct >= 45 else "faible")
    return StatResult(
        titre       = f"Duel : {agent_a.name} vs {agent_b.name}",
        metriques   = metriques,
        conclusions = conc,
        verdict     = verdict,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 5. PROFILAGE DES TEMPS
# ══════════════════════════════════════════════════════════════════════════════

def stats_temps(
    agent,
    n_coups:      int       = 50,
    seed:         int       = 42,
    percentiles:  List[int] = (50, 90, 95, 99),
) -> StatResult:
    """
    Profile les temps de réponse de l'agent sur des positions réelles.

    Métriques calculées
    ───────────────────
    - min, max, moy, std (ms)
    - percentiles P50, P90, P95, P99
    - coups dépassant un seuil (latence critique)
    - répartition début / milieu / fin de partie

    Paramètres
    ──────────
    agent    : objet avec .choose_move(state) → (move, score)
    n_coups  : nombre de coups à mesurer
    seed     : reproductibilité
    percentiles : percentiles à calculer
    """
    rng = random.Random(seed)
    times_ms = []
    times_par_phase = {"debut": [], "milieu": [], "fin": []}

    state = UTTTState.initial()
    mesures = 0

    while mesures < n_coups:
        if state.is_terminal:
            state = UTTTState.initial()

        legal = state.legal_moves()
        if not legal:
            state = UTTTState.initial()
            continue

        # Phase de partie
        nb_joues = int((state.board != 0).sum())  # cases remplies
        phase = "debut" if nb_joues < 20 else ("fin" if nb_joues > 55 else "milieu")

        t0 = time.perf_counter()
        move, _ = agent.choose_move(state)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        times_ms.append(elapsed_ms)
        times_par_phase[phase].append(elapsed_ms)
        mesures += 1

        # Avance la partie
        state = state.apply_move(move)

    t = np.array(times_ms)
    seuil_critique = float(np.percentile(t, 95))

    metriques = {
        "Mesures effectuées":              len(times_ms),
        "Temps min (ms)":                  round(float(t.min()),  2),
        "Temps moy. (ms)":                 round(float(t.mean()), 2),
        "Temps max (ms)":                  round(float(t.max()),  2),
        "Écart-type (ms)":                 round(float(t.std()),  2),
    }
    for p in percentiles:
        metriques[f"P{p} (ms)"] = round(float(np.percentile(t, p)), 2)
    metriques["Seuil 95e percentile (ms)"] = round(seuil_critique, 2)

    for phase, vals in times_par_phase.items():
        if vals:
            metriques[f"Moy. {phase} (ms)"] = round(float(np.mean(vals)), 2)

    conc = []
    moy = metriques["Temps moy. (ms)"]
    p99 = metriques.get("P99 (ms)", metriques.get(f"P{max(percentiles)} (ms)", moy))
    if moy < 10:
        conc.append(f"Temps de réponse très rapide ({moy:.1f}ms moy.) — adapté au jeu en temps réel.")
    elif moy < 100:
        conc.append(f"Temps de réponse acceptable ({moy:.1f}ms moy.).")
    else:
        conc.append(f"Temps de réponse élevé ({moy:.1f}ms moy.) — profondeur peut-être trop grande pour le temps réel.")

    ratio = float(t.std()) / max(float(t.mean()), 1e-9)
    if ratio > 2:
        conc.append(f"Haute variabilité (CV={ratio:.1f}) — le temps dépend fortement de la position.")
    else:
        conc.append(f"Temps stable (CV={ratio:.2f}) — peu de variance selon la position.")

    verdict = "excellent" if moy < 50 else ("correct" if moy < 500 else "faible")
    return StatResult(
        titre       = f"Profilage des temps — {agent.name}",
        metriques   = metriques,
        conclusions = conc,
        verdict     = verdict,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 6. CONCLUSION GLOBALE TEXTUELLE
# ══════════════════════════════════════════════════════════════════════════════

def conclusion(
    *stat_results: StatResult,
    titre_presentation: str = "Résultats de l'IA UTTT",
) -> str:
    """
    Agrège plusieurs StatResult et génère un texte de conclusion
    structuré, prêt pour une présentation.

    Usage
    ─────
    texte = conclusion(
        stats_parties(results),
        stats_evaluateur(ev),
        stats_search(ev),
        titre_presentation="Présentation — Projet IA UTTT",
    )
    print(texte)

    Retourne
    ────────
    str — texte multi-lignes avec sections, bullets et verdict global
    """
    W    = 64
    sep  = "─" * W
    dsep = "═" * W

    lignes = [
        dsep,
        f"  {titre_presentation}",
        f"  Généré le : {time.strftime('%d/%m/%Y %H:%M')}",
        dsep,
    ]

    verdicts = []
    for sr in stat_results:
        verdicts.append(sr.verdict)
        lignes += [
            "",
            f"  ▌ {sr.titre}",
            sep,
        ]
        # 3 métriques clés
        items = list(sr.metriques.items())
        for k, v in items[:6]:
            if isinstance(v, float):
                lignes.append(f"    • {k:<38} {v:.4g}")
            else:
                lignes.append(f"    • {k:<38} {v}")
        if sr.conclusions:
            lignes.append("")
            for c in sr.conclusions:
                lignes.append(f"    → {c}")
        lignes.append(f"    Verdict : {sr.verdict.upper()}")

    # Score global
    scores = {"excellent": 2, "correct": 1, "faible": 0}
    total  = sum(scores.get(v, 1) for v in verdicts)
    max_sc = len(verdicts) * 2 if verdicts else 1
    pct    = total / max_sc * 100 if max_sc else 0

    lignes += [
        "",
        dsep,
        f"  VERDICT GLOBAL ({total}/{max_sc} points, {pct:.0f}%)",
        sep,
    ]

    if pct >= 75:
        lignes += [
            "  ✓✓ Excellent — l'IA est opérationnelle et performante.",
            "     Le modèle peut être utilisé dans une démo ou compétition.",
    ]
    elif pct >= 50:
        lignes += [
            "  ✓  Correct — l'IA est fonctionnelle mais perfectible.",
            "     Continuer l'entraînement pour améliorer les métriques faibles.",
    ]
    else:
        lignes += [
            "  ✗  Insuffisant — des améliorations significatives sont nécessaires.",
            "     Vérifier l'architecture, les données, et la boucle d'entraînement.",
    ]

    # Recommandations ciblées
    faibles = [sr for sr in stat_results if sr.verdict == "faible"]
    if faibles:
        lignes += ["", "  Axes d'amélioration prioritaires :"]
        for sr in faibles:
            lignes.append(f"    • {sr.titre.split('(')[0].strip()}")

    lignes.append(dsep)
    return "\n".join(lignes)


# ══════════════════════════════════════════════════════════════════════════════
# 7. RAPPORT COMPLET
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Rapport:
    """Contient tous les StatResult et génère les affichages consolidés."""
    blocs:  List[StatResult]  = field(default_factory=list)
    titre:  str               = "Rapport UTTT IA"

    def print_all(self, large: bool = True) -> None:
        for b in self.blocs:
            b.print(large=large)

    def print_conclusion(self) -> None:
        print(conclusion(*self.blocs, titre_presentation=self.titre))

    def to_dict(self) -> dict:
        return {
            "titre": self.titre,
            "blocs": [b.to_dict() for b in self.blocs],
        }

    def save_json(self, path: str) -> None:
        import json
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"[Rapport] Sauvegardé → {path}")


def rapport_complet(
    evaluator,
    agent_a         = None,     # agent de référence (si None : AlphaBeta depth=3)
    agent_b         = None,     # agent adversaire (si None : Random)
    n_games:    int = 30,
    depth:      int = 3,
    n_states:   int = 40,
    seed:       int = 42,
    titre:      str = "Rapport complet — IA UTTT",
) -> Rapport:
    """
    Lance tous les blocs de statistiques et retourne un Rapport.

    Utilisation
    ───────────
    from stats import rapport_complet
    from model import NeuralEvaluator

    ev      = NeuralEvaluator("models/best.pth")
    rapport = rapport_complet(ev, n_games=50, depth=3)
    rapport.print_all()
    rapport.print_conclusion()
    rapport.save_json("results/rapport.json")
    """
    from bot_alphabeta import AlphaBetaAgent
    from bot_random    import RandomAgent
    from arena         import Arena

    if agent_a is None:
        agent_a = AlphaBetaAgent(evaluator, depth=depth)
    if agent_b is None:
        agent_b = RandomAgent()

    random.seed(seed)
    np.random.seed(seed)

    print(f"\n[Rapport] {titre}")
    print("═" * 55)

    blocs = []

    # 1. Duel principal
    print("\n[1/5] Duel principal...")
    blocs.append(stats_duels(agent_a, agent_b, n_games=n_games, seed=seed))

    # 2. Parties (depuis le duel)
    print("\n[2/5] Statistiques des parties...")
    report = Arena(agent_a, agent_b).run(n_games=n_games, alternate=True)
    blocs.append(stats_parties(report._results, agent_a=agent_a.name))

    # 3. Évaluateur
    print("\n[3/5] Qualité de l'évaluateur...")
    blocs.append(stats_evaluateur(evaluator, n_games=n_games, n_states=n_states, seed=seed))

    # 4. Moteur de recherche
    print("\n[4/5] Moteur alpha-bêta...")
    blocs.append(stats_search(evaluator, depths=list(range(1, depth + 1)),
                               n_positions=20, seed=seed))

    # 5. Temps de réponse
    print("\n[5/5] Profilage des temps...")
    blocs.append(stats_temps(agent_a, n_coups=40, seed=seed))

    print("\n[Rapport] Terminé.\n")
    return Rapport(blocs=blocs, titre=titre)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS INTERNES
# ══════════════════════════════════════════════════════════════════════════════

def _jouer_partie_aleatoire(rng: random.Random):
    """Retourne (liste d'états, gagnant). Ne dépend pas de torch."""
    states = []
    s = UTTTState.initial()
    while not s.is_terminal:
        states.append(s)
        s = s.apply_move(rng.choice(s.legal_moves()))
    states.append(s)
    return states, s.winner


def _generer_positions(n: int, rng: random.Random) -> List[UTTTState]:
    """Génère n positions diversifiées (début, milieu, fin)."""
    positions = []
    for _ in range(n * 3):
        s = UTTTState.initial()
        nb = rng.randint(3, 40)
        for __ in range(nb):
            if s.is_terminal:
                break
            s = s.apply_move(rng.choice(s.legal_moves()))
        if not s.is_terminal and len(s.legal_moves()) >= 2:
            positions.append(s)
        if len(positions) >= n:
            break
    return positions


def _elo_diff(win_rate: float) -> float:
    """Estime la différence Elo depuis le win rate (formule Elo standard)."""
    wr = max(0.001, min(0.999, win_rate))
    return -400 * math.log10(1.0 / wr - 1.0)
