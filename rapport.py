"""
rapport.py — Script principal de génération du rapport de présentation
=======================================================================
Lance toutes les fonctions de stats.py, affiche les résultats et
génère un fichier texte de conclusion prêt pour une présentation.

Usage
─────
  # Sans checkpoint (évaluateur aléatoire — pour tester la structure)
  python rapport.py --no-eval

  # Avec checkpoint complet (UTTTNet 10×256)
  python rapport.py --checkpoint models/best_uttt_model.pth

  # Avec checkpoint AlphaZero léger (UTTTNetLight 4×128)
  python rapport.py --checkpoint models/alphazero/best.pth --light

  # Paramètres personnalisés
  python rapport.py --checkpoint models/best.pth --games 50 --depth 4 --seed 7

Sorties
───────
  - Affichage console (métriques + conclusions)
  - rapport_presentation.txt  (texte de conclusion structuré)
  - rapport_data.json         (données brutes exportées)
"""

import argparse
import random
import sys

import numpy as np

sys.path.insert(0, ".")

from game  import UTTTState
from stats import (
    StatResult,
    Rapport,
    stats_parties,
    stats_search,
    stats_evaluateur,
    stats_duels,
    stats_temps,
    conclusion,
    rapport_complet,
)


# ─────────────────────────────────────────────────────────────────────────────
# ÉVALUATEUR ALÉATOIRE (pour tests sans checkpoint)
# ─────────────────────────────────────────────────────────────────────────────

class RandomEvaluator:
    """Évaluateur factice — scores aléatoires reproductibles."""
    name = "RandomEvaluator"

    def __init__(self, seed: int = 0):
        self._rng = np.random.RandomState(seed)

    def evaluate(self, state) -> float:
        h = abs(hash(state.board.tobytes())) % 1000
        return float(h) / 500.0 - 1.0

    def policy_logprobs(self, state) -> np.ndarray:
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


# ─────────────────────────────────────────────────────────────────────────────
# CHARGEMENT DU CHECKPOINT
# ─────────────────────────────────────────────────────────────────────────────

def load_evaluator(checkpoint: str, light: bool = False, device: str = None):
    """
    Charge NeuralEvaluator ou LightEvaluator selon les paramètres.
    Détecte automatiquement le type si --light n'est pas spécifié.
    """
    import torch
    sd   = torch.load(checkpoint, map_location="cpu", weights_only=True)
    keys = list(sd.keys())

    is_light = light or any("p_conv" in k or ".net.0." in k for k in keys)

    if is_light:
        from bot_mcts import LightEvaluator
        filters = sd["stem.0.weight"].shape[0]
        blocks  = max(int(k.split(".")[1]) for k in keys if k.startswith("res_blocks.")) + 1
        print(f"[Checkpoint] UTTTNetLight détecté ({filters} filtres × {blocks} blocs)")
        return LightEvaluator(checkpoint, device=device,
                              num_filters=filters, num_res_blocks=blocks)
    else:
        from model import NeuralEvaluator
        filters = sd["stem.0.weight"].shape[0]
        blocks  = max(int(k.split(".")[1]) for k in keys if k.startswith("res_blocks.")) + 1
        print(f"[Checkpoint] UTTTNet détecté ({filters} filtres × {blocks} blocs)")
        return NeuralEvaluator(checkpoint, device=device,
                               num_filters=filters, num_res_blocks=blocks)


# ─────────────────────────────────────────────────────────────────────────────
# RAPPORT PAR SECTIONS (alternative à rapport_complet)
# ─────────────────────────────────────────────────────────────────────────────

def run_rapport_sections(ev, args) -> Rapport:
    """
    Lance les sections de rapport séparément pour plus de contrôle.
    Utile pour personnaliser quels blocs inclure.
    """
    from bot_alphabeta import AlphaBetaAgent
    from bot_random    import RandomAgent
    from arena         import Arena

    random.seed(args.seed)
    np.random.seed(args.seed)

    agent_ab  = AlphaBetaAgent(ev, depth=args.depth)
    agent_rnd = RandomAgent()

    blocs = []

    # ── Section 1 : Vue d'ensemble du duel ───────────────────────────────────
    if not args.skip_duel:
        print(f"\n{'─'*55}")
        print(f"  Section 1 : Duel AlphaBeta(d={args.depth}) vs Random")
        print(f"{'─'*55}")
        blocs.append(stats_duels(agent_ab, agent_rnd,
                                 n_games=args.games, seed=args.seed))
        blocs[-1].print(large=True)

    # ── Section 2 : Statistiques des parties ─────────────────────────────────
    if not args.skip_parties:
        print(f"\n{'─'*55}")
        print(f"  Section 2 : Statistiques des parties")
        print(f"{'─'*55}")
        report = Arena(AlphaBetaAgent(ev, depth=args.depth),
                       RandomAgent()).run(n_games=args.games, alternate=True)
        s_parties = stats_parties(report._results, agent_a=agent_ab.name)
        blocs.append(s_parties)
        s_parties.print(large=True)

    # ── Section 3 : Qualité de l'évaluateur ──────────────────────────────────
    if not args.skip_eval:
        print(f"\n{'─'*55}")
        print(f"  Section 3 : Qualité de l'évaluateur neuronal")
        print(f"{'─'*55}")
        s_ev = stats_evaluateur(ev, n_games=args.games,
                                n_states=args.states, seed=args.seed)
        blocs.append(s_ev)
        s_ev.print(large=True)

    # ── Section 4 : Moteur de recherche ──────────────────────────────────────
    if not args.skip_search:
        print(f"\n{'─'*55}")
        print(f"  Section 4 : Moteur alpha-bêta")
        print(f"{'─'*55}")
        s_search = stats_search(
            ev,
            depths      = list(range(1, args.depth + 1)),
            n_positions = max(args.states, 20),
            seed        = args.seed,
        )
        blocs.append(s_search)
        s_search.print(large=True)

    # ── Section 5 : Temps de réponse ─────────────────────────────────────────
    if not args.skip_temps:
        print(f"\n{'─'*55}")
        print(f"  Section 5 : Profilage des temps de réponse")
        print(f"{'─'*55}")
        s_temps = stats_temps(agent_ab, n_coups=args.states, seed=args.seed)
        blocs.append(s_temps)
        s_temps.print(large=True)

    # ── Comparaison de profondeurs ────────────────────────────────────────────
    if args.compare_depths:
        print(f"\n{'─'*55}")
        print(f"  Section extra : Comparaison de profondeurs")
        print(f"{'─'*55}")
        for d in range(1, args.depth + 1):
            ag_d   = AlphaBetaAgent(ev, depth=d)
            ag_rnd = RandomAgent()
            rep_d  = Arena(ag_d, ag_rnd).run(n_games=max(args.games // 2, 10), alternate=True)
            s_d    = stats_parties(rep_d._results, agent_a=ag_d.name)
            s_d.titre = f"AlphaBeta depth={d} vs Random"
            blocs.append(s_d)
            print(f"  depth={d}  WR={s_d.metriques['Win rate (%)']:.1f}%  "
                  f"coups={s_d.metriques['Coups / partie (moy.)']:.1f}")

    return Rapport(blocs=blocs,
                   titre=f"Rapport IA UTTT — AlphaBeta(d={args.depth})")


# ─────────────────────────────────────────────────────────────────────────────
# SAUVEGARDE DU RAPPORT TEXTE
# ─────────────────────────────────────────────────────────────────────────────

def sauvegarder_rapport(rapport: Rapport, path_txt: str, path_json: str) -> None:
    conc = conclusion(*rapport.blocs, titre_presentation=rapport.titre)

    with open(path_txt, "w", encoding="utf-8") as f:
        f.write(conc)
    print(f"\n✓ Rapport texte   → {path_txt}")

    rapport.save_json(path_json)
    print(f"✓ Données JSON    → {path_json}")


# ─────────────────────────────────────────────────────────────────────────────
# AFFICHAGE FINAL POUR PRÉSENTATION
# ─────────────────────────────────────────────────────────────────────────────

def print_resume_presentation(rapport: Rapport) -> None:
    """Affiche un résumé condensé, idéal pour une diapositive."""
    W = 64
    print(f"\n{'═'*W}")
    print(f"  RÉSUMÉ POUR PRÉSENTATION")
    print(f"{'═'*W}")

    verdict_symbols = {"excellent": "✓✓", "correct": "✓ ", "faible": "✗ "}
    for b in rapport.blocs:
        sym = verdict_symbols.get(b.verdict, "· ")
        titre_court = b.titre.split("(")[0].strip()[:45]
        print(f"  {sym}  {titre_court:<45}  [{b.verdict.upper()}]")

    print(f"{'─'*W}")
    scores = {"excellent": 2, "correct": 1, "faible": 0}
    total  = sum(scores.get(b.verdict, 1) for b in rapport.blocs)
    max_sc = len(rapport.blocs) * 2
    pct    = total / max_sc * 100 if max_sc else 0
    print(f"  Score global : {total}/{max_sc}  ({pct:.0f}%)")
    print(f"{'═'*W}\n")

    # Points clés extraits
    print("  Points clés à mentionner en présentation :")
    print(f"{'─'*W}")
    for b in rapport.blocs:
        for c in b.conclusions[:1]:   # 1 conclusion par section
            print(f"  • {c}")
    print(f"{'═'*W}\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Génère le rapport complet de l'IA UTTT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python rapport.py --no-eval                    # sans checkpoint
  python rapport.py --checkpoint models/best.pth
  python rapport.py --checkpoint models/best.pth --depth 4 --games 50
  python rapport.py --checkpoint models/best.pth --compare-depths
        """,
    )
    p.add_argument("--checkpoint",      default=None,
                   help="Chemin vers le .pth (optionnel)")
    p.add_argument("--light",           action="store_true",
                   help="Forcer LightEvaluator (UTTTNetLight)")
    p.add_argument("--no-eval",         action="store_true",
                   help="Utiliser un évaluateur aléatoire (sans torch)")
    p.add_argument("--device",          default=None,
                   help="cpu | cuda (auto-détecté si absent)")
    p.add_argument("--depth",           type=int, default=3,
                   help="Profondeur alpha-bêta (défaut: 3)")
    p.add_argument("--games",           type=int, default=30,
                   help="Parties par test (défaut: 30)")
    p.add_argument("--states",          type=int, default=40,
                   help="États pour les tests policy/temps (défaut: 40)")
    p.add_argument("--seed",            type=int, default=42)
    p.add_argument("--out-txt",         default="rapport_presentation.txt")
    p.add_argument("--out-json",        default="rapport_data.json")

    # Options de sélection des sections
    p.add_argument("--skip-duel",       action="store_true")
    p.add_argument("--skip-parties",    action="store_true")
    p.add_argument("--skip-eval",       action="store_true")
    p.add_argument("--skip-search",     action="store_true")
    p.add_argument("--skip-temps",      action="store_true")
    p.add_argument("--compare-depths",  action="store_true",
                   help="Ajoute une comparaison de profondeurs 1..depth")

    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    # ── Évaluateur ────────────────────────────────────────────────────────────
    if args.no_eval or args.checkpoint is None:
        print("[INFO] Évaluateur aléatoire utilisé (--no-eval ou pas de checkpoint).")
        ev = RandomEvaluator(seed=args.seed)
    else:
        print(f"[INFO] Chargement du checkpoint : {args.checkpoint}")
        ev = load_evaluator(args.checkpoint, light=args.light, device=args.device)

    # ── Rapport ───────────────────────────────────────────────────────────────
    rapport = run_rapport_sections(ev, args)

    # ── Conclusion ────────────────────────────────────────────────────────────
    print("\n" + "═" * 64)
    print("  CONCLUSION GLOBALE")
    print("═" * 64)
    print(conclusion(*rapport.blocs, titre_presentation=rapport.titre))

    # ── Résumé présentation ───────────────────────────────────────────────────
    print_resume_presentation(rapport)

    # ── Sauvegarde ────────────────────────────────────────────────────────────
    sauvegarder_rapport(rapport, args.out_txt, args.out_json)


if __name__ == "__main__":
    main()
