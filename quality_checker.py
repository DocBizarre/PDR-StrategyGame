"""
quality_checker.py — Vérification qualité des scores du réseau
===============================================================
Teste si le réseau attribue des scores cohérents avec la réalité :
  +1 pour les coups gagnants
  -1 pour les coups perdants
   0 pour les coups nuls

Usage
─────
  python quality_checker.py
  python quality_checker.py --checkpoint models/alphazero/best.pth
  python quality_checker.py --games 50 --depth 2
"""

from __future__ import annotations

import argparse
import random
import time
from typing import List, Tuple

import numpy as np

from game import UTTTState


def _load_evaluator(checkpoint_path: str, device: str = None):
    """
    Charge automatiquement le bon évaluateur selon le checkpoint.
    Détecte si c'est un UTTTNet (NeuralEvaluator) ou UTTTNetLight (LightEvaluator)
    en inspectant les clés du state_dict.
    """
    import torch
    sd   = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    keys = list(sd.keys())

    # UTTTNetLight a des clés "p_conv.*" ou "res_blocks.0.net.*"
    is_light = any("p_conv" in k or ".net.0." in k for k in keys)

    if is_light:
        from bot_mcts import LightEvaluator
        filters = sd["stem.0.weight"].shape[0]
        blocks  = max(int(k.split(".")[1]) for k in keys if k.startswith("res_blocks.")) + 1
        return LightEvaluator(
            checkpoint     = checkpoint_path,
            device         = device,
            num_filters    = filters,
            num_res_blocks = blocks,
        )
    else:
        from model import NeuralEvaluator
        return NeuralEvaluator(checkpoint_path, device=device)


# ═════════════════════════════════════════════════════════════════════════════
# CONSTRUCTEURS DE POSITIONS TEST
# ═════════════════════════════════════════════════════════════════════════════

def _build_near_win(player: int) -> UTTTState:
    """
    Position où player a gagné SG0 et SG1 et a 2 pions dans SG2.
    Construite via mutation directe avec _winner recalculé.
    """
    from game import check_winner_small
    s        = UTTTState.initial()
    other    = 3 - player
    s.board      = s.board.copy()
    s.meta_board = s.meta_board.copy()

    # player gagne SG0 (ligne du haut)
    for cell in [0, 1, 2]:
        s.board[cell] = player
    s.meta_board[0] = player

    # player gagne SG1 (ligne du haut)
    for cell in [9, 10, 11]:
        s.board[cell] = player
    s.meta_board[1] = player

    # player a 2 pions dans SG2 (sur le point de gagner)
    s.board[18] = player
    s.board[19] = player

    # adversaire a 2 pions épars dans SG3 (pas de menace)
    s.board[27] = other
    s.board[28] = other

    s.active_idx = -1
    s.player     = player
    s._winner    = check_winner_small(s.meta_board)
    return s


def _build_near_loss(player: int) -> UTTTState:
    """Position où l'adversaire (3-player) est sur le point de gagner."""
    from game import check_winner_small
    s        = _build_near_win(3 - player)
    s.player = player
    s._winner = check_winner_small(s.meta_board)
    return s


def _play_random_game() -> Tuple[List[UTTTState], int]:
    """Joue une partie aléatoire complète. Retourne (états, gagnant)."""
    states = []
    s = UTTTState.initial()
    while not s.is_terminal:
        states.append(s)
        s = s.apply_move(random.choice(s.legal_moves()))
    states.append(s)
    return states, s.winner


# ═════════════════════════════════════════════════════════════════════════════
# TESTS DE QUALITÉ
# ═════════════════════════════════════════════════════════════════════════════

def test_positions_construites(ev) -> dict:
    """
    Test 1 — Positions manuelles.
    Le réseau doit scorer +1 si le joueur courant est en position gagnante,
    -1 s'il est en position perdante.
    """
    results = {"correct": 0, "total": 0, "details": []}

    for player in [1, 2]:
        # Position quasi-gagnante : joueur courant va gagner → score > 0
        s_win = _build_near_win(player)
        v_win = ev.evaluate(s_win)
        ok_win = v_win > 0.0
        results["correct"] += ok_win
        results["total"]   += 1
        results["details"].append({
            "label":    f"J{player} va gagner (J{player} joue)",
            "score":    round(float(v_win), 4),
            "attendu":  "> 0  (idéal → +1.0)",
            "ok":       ok_win,
        })

        # Position quasi-gagnante : adversaire joue → score doit être négatif
        from game import check_winner_small
        s_win2 = _build_near_win(player)
        s_win2.player = 3 - player
        s_win2._winner = check_winner_small(s_win2.meta_board)
        v_win2 = ev.evaluate(s_win2)
        ok_win2 = v_win2 < 0.0
        results["correct"] += ok_win2
        results["total"]   += 1
        results["details"].append({
            "label":    f"J{player} va gagner (J{3-player} joue)",
            "score":    round(float(v_win2), 4),
            "attendu":  "< 0  (idéal → -1.0)",
            "ok":       ok_win2,
        })

        # Position quasi-perdante : joueur courant va perdre → score < 0
        s_lose = _build_near_loss(player)
        v_lose = ev.evaluate(s_lose)
        ok_lose = v_lose < 0.0
        results["correct"] += ok_lose
        results["total"]   += 1
        results["details"].append({
            "label":    f"J{player} va perdre (J{player} joue)",
            "score":    round(float(v_lose), 4),
            "attendu":  "< 0  (idéal → -1.0)",
            "ok":       ok_lose,
        })

    results["accuracy"] = results["correct"] / results["total"] * 100
    return results


def test_fin_de_partie(ev, n_games: int = 30) -> dict:
    """
    Test 2 — États terminaux et quasi-terminaux.
    Les 3 derniers coups avant la fin doivent avoir des scores
    de plus en plus proches de +1 (gagnant) ou -1 (perdant).
    """
    scores_gagnant = []
    scores_perdant = []
    scores_nul     = []
    monotonie_ok   = 0
    n_parties      = 0

    for _ in range(n_games):
        states, winner = _play_random_game()
        if winner == 0 or len(states) < 4:
            # Pour les nuls, on mesure juste que le score est proche de 0
            if winner == 0 and len(states) >= 2:
                for s in states[-3:-1]:
                    scores_nul.append(abs(ev.evaluate(s)))
            continue

        n_parties += 1

        # Scores sur les 5 derniers états non terminaux
        last_states = states[-6:-1]  # exclut l'état terminal
        last_scores = []

        for s in last_states:
            v = ev.evaluate(s)
            # Depuis la perspective du gagnant final
            score_from_winner = v if s.player == winner else -v
            last_scores.append(score_from_winner)

            if s.player == winner:
                scores_gagnant.append(v)
            else:
                scores_perdant.append(v)

        # Monotonie : les scores du gagnant doivent augmenter en fin de partie
        if len(last_scores) >= 2:
            trend = last_scores[-1] - last_scores[0]
            if trend > 0:
                monotonie_ok += 1

    return {
        "score_gagnant_mean": round(float(np.mean(scores_gagnant)), 4) if scores_gagnant else 0.0,
        "score_perdant_mean": round(float(np.mean(scores_perdant)), 4) if scores_perdant else 0.0,
        "score_nul_mean":     round(float(np.mean(scores_nul)),     4) if scores_nul     else 0.0,
        "separation":         round(float(np.mean(scores_gagnant) - np.mean(scores_perdant)), 4)
                              if scores_gagnant and scores_perdant else 0.0,
        "monotonie_pct":      round(monotonie_ok / n_parties * 100, 1) if n_parties else 0.0,
        "n_parties":          n_parties,
        # Idéal : gagnant → +1, perdant → -1, nul → 0, separation → 2.0
    }


def test_accuracy_directionnelle(ev, n_games: int = 40) -> dict:
    """
    Test 3 — Accuracy directionnelle.
    Pour chaque état non terminal :
      score > 0 → le réseau prédit que le joueur courant va gagner
    On compare avec l'issue réelle.
    Par phase : début / milieu / fin de partie.
    """
    correct_total = 0
    total         = 0
    by_phase      = {
        "début":  [0, 0],
        "milieu": [0, 0],
        "fin":    [0, 0],
    }

    for _ in range(n_games):
        states, winner = _play_random_game()
        if winner == 0: continue
        n = len(states) - 1  # exclut le terminal

        for t, s in enumerate(states[:-1]):
            v    = ev.evaluate(s)
            pred = s.player if v > 0 else (3 - s.player)
            ok   = (pred == winner)

            correct_total += ok
            total         += 1

            pct = t / n
            if pct < 0.33:   phase = "début"
            elif pct < 0.66: phase = "milieu"
            else:            phase = "fin"

            by_phase[phase][0] += ok
            by_phase[phase][1] += 1

    def acc(p):
        c, t = by_phase[p]
        return round(c / t * 100, 1) if t else 0.0

    return {
        "accuracy_globale": round(correct_total / total * 100, 1) if total else 0.0,
        "accuracy_debut":   acc("début"),
        "accuracy_milieu":  acc("milieu"),
        "accuracy_fin":     acc("fin"),
        "n_etats":          total,
        # Idéal : accuracy_globale > 65%, accuracy_fin > 80%
    }


def test_calibration(ev, n_games: int = 40) -> dict:
    """
    Test 4 — Calibration.
    Corrélation entre le score prédit et le résultat réel (+1/-1/0).
    MSE entre score prédit et résultat final.
    """
    predicted = []
    actual    = []

    for _ in range(n_games):
        states, winner = _play_random_game()
        for s in states[:-1]:
            v = ev.evaluate(s)
            z = 0.0 if winner == 0 else (1.0 if winner == s.player else -1.0)
            predicted.append(v)
            actual.append(z)

    p = np.array(predicted, dtype=np.float32)
    a = np.array(actual,    dtype=np.float32)

    corr = float(np.corrcoef(p, a)[0, 1]) if np.std(p) > 1e-6 else 0.0
    mse  = float(np.mean((p - a) ** 2))

    return {
        "correlation": round(corr, 4),
        "mse":         round(mse, 4),
        "score_std":   round(float(np.std(p)), 4),
        "n_etats":     len(predicted),
        # Idéal : correlation > 0.4, mse < 0.5, score_std > 0.15
    }


def test_policy_head(ev, n_states: int = 30) -> dict:
    """
    Test 5 — Policy head.
    - Le coup le plus probable est-il souvent le meilleur selon value ?
    - La probabilité totale sur les coups légaux est-elle proche de 1 ?
    """
    prob_legal_list     = []
    top1_matches_value  = []

    s = UTTTState.initial()
    for _ in range(n_states):
        if s.is_terminal:
            s = UTTTState.initial()
        legal = s.legal_moves()
        if not legal: break

        lp    = ev.policy_logprobs(s)
        probs = np.exp(lp)

        # Prob totale sur coups légaux
        prob_legal_list.append(float(probs[legal].sum()))

        # Meilleur coup selon policy
        top_policy_move = legal[int(np.argmax(probs[legal]))]

        # Meilleur coup selon value à depth=1
        child_vals = {m: -ev.evaluate(s.apply_move(m)) for m in legal}
        best_value_move = max(child_vals, key=lambda m: child_vals[m])

        top1_matches_value.append(top_policy_move == best_value_move)

        # Avance d'un coup aléatoire
        s = s.apply_move(random.choice(legal))

    return {
        "prob_legal_mean":    round(float(np.mean(prob_legal_list)),    4) if prob_legal_list   else 0.0,
        "top1_match_value":   round(float(np.mean(top1_matches_value)) * 100, 1) if top1_matches_value else 0.0,
        "n_etats":            len(prob_legal_list),
        # Idéal : prob_legal_mean > 0.9, top1_match_value > 50%
    }


# ═════════════════════════════════════════════════════════════════════════════
# RAPPORT
# ═════════════════════════════════════════════════════════════════════════════

def run_quality_check(
    ev,
    n_games:  int = 30,
    n_states: int = 30,
    verbose:  bool = True,
) -> dict:
    """
    Lance tous les tests et retourne un dict de métriques.
    Peut être appelé depuis un autre module.
    """
    t0 = time.time()

    r1 = test_positions_construites(ev)
    r2 = test_fin_de_partie(ev,            n_games=n_games)
    r3 = test_accuracy_directionnelle(ev,  n_games=n_games)
    r4 = test_calibration(ev,              n_games=n_games)
    r5 = test_policy_head(ev,              n_states=n_states)

    elapsed = time.time() - t0

    results = {
        "positions":   r1,
        "fin_partie":  r2,
        "accuracy":    r3,
        "calibration": r4,
        "policy":      r5,
        "elapsed":     round(elapsed, 1),
        "timestamp":   time.strftime("%H:%M:%S"),
    }

    if verbose:
        _print_report(results)

    return results


def _print_report(r: dict) -> None:
    W = 58

    def sep(t):
        print(f"\n{'─'*W}\n  {t}\n{'─'*W}")

    def ok(b):
        return "✓" if b else "✗"

    print(f"\n{'═'*W}")
    print(f"  RAPPORT QUALITÉ DES SCORES  ({r['timestamp']})")
    print(f"{'═'*W}")

    # ── Test 1 ────────────────────────────────────────────────────────────
    sep(f"1. Positions construites  ({r['positions']['accuracy']:.0f}% correctes)")
    for d in r["positions"]["details"]:
        print(f"  {ok(d['ok'])} {d['label']:<40} score={d['score']:+.4f}  (attendu {d['attendu']})")

    # ── Test 2 ────────────────────────────────────────────────────────────
    p2 = r["fin_partie"]
    sep("2. Fin de partie")
    print(f"  Score moyen gagnant : {p2['score_gagnant_mean']:+.4f}  (idéal → +1.0)")
    print(f"  Score moyen perdant : {p2['score_perdant_mean']:+.4f}  (idéal → -1.0)")
    print(f"  Score moyen nul     : {p2['score_nul_mean']:+.4f}  (idéal →  0.0)")
    print(f"  Séparation          : {p2['separation']:+.4f}  (idéal →  2.0)")
    print(f"  Monotonie fin/début : {p2['monotonie_pct']:.1f}%  (idéal > 60%)")
    sep2 = p2["separation"]
    print(f"  {ok(sep2 > 0.5)} Séparation {'correcte' if sep2 > 0.5 else 'trop faible — le réseau ne discrimine pas'}")

    # ── Test 3 ────────────────────────────────────────────────────────────
    p3 = r["accuracy"]
    sep("3. Accuracy directionnelle")
    print(f"  Globale : {p3['accuracy_globale']:.1f}%  (idéal > 65%)")
    print(f"  Début   : {p3['accuracy_debut']:.1f}%")
    print(f"  Milieu  : {p3['accuracy_milieu']:.1f}%")
    print(f"  Fin     : {p3['accuracy_fin']:.1f}%  (le plus important)")
    print(f"  {ok(p3['accuracy_globale'] > 65)} {'Bonne prédiction' if p3['accuracy_globale'] > 65 else 'Prédiction faible'}")

    # ── Test 4 ────────────────────────────────────────────────────────────
    p4 = r["calibration"]
    sep("4. Calibration (corrélation score ↔ résultat réel)")
    print(f"  Corrélation : {p4['correlation']:+.4f}  (idéal > 0.4)")
    print(f"  MSE         : {p4['mse']:.4f}  (idéal < 0.5)")
    print(f"  Std scores  : {p4['score_std']:.4f}  (idéal > 0.15)")
    print(f"  {ok(p4['correlation'] > 0.4)} {'Bonne calibration' if p4['correlation'] > 0.4 else 'Calibration faible'}")

    # ── Test 5 ────────────────────────────────────────────────────────────
    p5 = r["policy"]
    sep("5. Policy head")
    print(f"  Prob. sur coups légaux : {p5['prob_legal_mean']:.4f}  (idéal → 1.0)")
    print(f"  Top-1 concorde value   : {p5['top1_match_value']:.1f}%  (idéal > 50%)")
    print(f"  {ok(p5['prob_legal_mean'] > 0.8)} {'Policy concentrée' if p5['prob_legal_mean'] > 0.8 else 'Policy dispersée sur coups illégaux'}")

    # ── Score global ──────────────────────────────────────────────────────
    scores = [
        r["positions"]["accuracy"] >= 66,
        p2["separation"] > 0.5,
        p3["accuracy_globale"] > 60,
        p4["correlation"] > 0.3,
        p5["prob_legal_mean"] > 0.8,
    ]
    n_ok = sum(scores)
    sep(f"VERDICT  ({n_ok}/5 métriques satisfaites)")

    if n_ok == 5:
        print("  ✓ Excellent — le réseau score correctement les positions.")
    elif n_ok >= 3:
        print("  ~ Acceptable — quelques faiblesses, continuer l'entraînement.")
    else:
        print("  ✗ Insuffisant — les scores ne reflètent pas la réalité.")
        print("    → Augmenter les itérations ou vérifier le réseau.")

    print(f"\n  Temps d'exécution : {r['elapsed']:.1f}s")
    print(f"{'═'*W}\n")


# ═════════════════════════════════════════════════════════════════════════════
# MODE CONTINU (parallèle à l'entraînement)
# ═════════════════════════════════════════════════════════════════════════════

def run_continuous(
    checkpoint_path: str,
    output_json:     str,
    interval:        int,
    n_games:         int,
    device:          str = None,
) -> None:
    """
    Tourne en boucle infinie :
      1. Charge le checkpoint le plus récent
      2. Lance run_quality_check()
      3. Écrit les résultats dans output_json
      4. Attend interval secondes
    """
    import json, os

    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
    cycle = 0

    print(f"[QualityChecker] Démarré — checkpoint : {checkpoint_path}")
    print(f"[QualityChecker] Sortie JSON : {output_json}")
    print(f"[QualityChecker] Intervalle  : {interval}s\n")

    while True:
        cycle += 1
        print(f"[QualityChecker] Cycle {cycle} — chargement du checkpoint…")

        try:
            ev = _load_evaluator(checkpoint_path, device=device)
        except Exception as e:
            print(f"[QualityChecker] ✗ Erreur chargement : {e} — nouvelle tentative dans {interval}s")
            time.sleep(interval)
            continue

        results = run_quality_check(ev, n_games=n_games, verbose=True)
        results["cycle"] = cycle

        with open(output_json, "w") as f:
            json.dump(results, f, indent=2)

        print(f"[QualityChecker] Résultats écrits → {output_json}")
        print(f"[QualityChecker] Prochain cycle dans {interval}s…\n")
        time.sleep(interval)


# ═════════════════════════════════════════════════════════════════════════════
# POINT D'ENTRÉE
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vérification qualité des scores du réseau")
    parser.add_argument("--checkpoint", default="models/best_uttt_model.pth",
                        help="Checkpoint à évaluer")
    parser.add_argument("--games",    type=int, default=30,
                        help="Parties par test (défaut : 30)")
    parser.add_argument("--states",   type=int, default=30,
                        help="États pour le test policy (défaut : 30)")
    parser.add_argument("--device",   default=None)
    parser.add_argument("--seed",     type=int, default=42)

    # Mode continu
    parser.add_argument("--continuous", action="store_true",
                        help="Tourne en boucle, recharge le checkpoint à chaque cycle")
    parser.add_argument("--interval",   type=int, default=120,
                        help="Secondes entre deux cycles (défaut : 120)")
    parser.add_argument("--output",     default="models/alphazero/quality.json",
                        help="Fichier JSON de sortie (mode continu)")

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.continuous:
        run_continuous(
            checkpoint_path = args.checkpoint,
            output_json     = args.output,
            interval        = args.interval,
            n_games         = args.games,
            device          = args.device,
        )
    else:
        ev = _load_evaluator(args.checkpoint, device=args.device)
        run_quality_check(ev, n_games=args.games, n_states=args.states, verbose=True)
