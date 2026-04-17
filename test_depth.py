"""
test_depth.py — Vérifie que les scores alpha-bêta varient avec la profondeur
=============================================================================
Usage :
  python test_depth.py
  python test_depth.py --checkpoint models/best_uttt_model.pth
"""

import argparse
import random
import numpy as np
from game          import UTTTState
from model         import NeuralEvaluator
from search        import best_move

def sep(t): print(f"\n{'─'*55}\n  {t}\n{'─'*55}")

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default="models/best_uttt_model.pth")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

random.seed(args.seed)
ev = NeuralEvaluator(args.checkpoint)

# ── État initial ──────────────────────────────────────────────────────────────
sep("Test 1 — État initial (plateau vide)")
state = UTTTState.initial()
legal = state.legal_moves()
print(f"  Coups légaux : {len(legal)}")

results = {}
for depth in [1, 2, 3, 4]:
    move, score = best_move(state, ev, depth=depth, top_k=None, verbose=False)
    results[depth] = (move, score)
    sg, cell = move // 9, move % 9
    print(f"  depth={depth}  →  move={move:2d} (SG={sg}, cell={cell})  score={score:+.4f}")

# Vérification : les scores doivent varier
scores_list = [r[1] for r in results.values()]
moves_list  = [r[0] for r in results.values()]
all_same_score = len(set(f"{s:.4f}" for s in scores_list)) == 1
all_same_move  = len(set(moves_list)) == 1

print(f"\n  Scores identiques à toutes profondeurs : {'OUI ← PROBLÈME' if all_same_score else 'NON ← OK'}")
print(f"  Même coup à toutes profondeurs         : {'OUI (peut être normal)' if all_same_move else 'NON ← OK'}")

# ── État milieu de partie ─────────────────────────────────────────────────────
sep("Test 2 — Milieu de partie (10 coups joués aléatoirement)")
state2 = UTTTState.initial()
for _ in range(10):
    state2 = state2.apply_move(random.choice(state2.legal_moves()))

legal2 = state2.legal_moves()
print(f"  Coups légaux : {len(legal2)}  |  Joueur : J{state2.player}")

results2 = {}
for depth in [1, 2, 3, 4]:
    move, score = best_move(state2, ev, depth=depth, top_k=None, verbose=False)
    results2[depth] = (move, score)
    sg, cell = move // 9, move % 9
    print(f"  depth={depth}  →  move={move:2d} (SG={sg}, cell={cell})  score={score:+.4f}")

scores2 = [r[1] for r in results2.values()]
moves2  = [r[0] for r in results2.values()]
all_same_score2 = len(set(f"{s:.4f}" for s in scores2)) == 1
all_same_move2  = len(set(moves2)) == 1
print(f"\n  Scores identiques : {'OUI ← PROBLÈME' if all_same_score2 else 'NON ← OK'}")
print(f"  Même coup         : {'OUI (peut être normal)' if all_same_move2 else 'NON ← OK'}")

# ── Scores par coup à différentes profondeurs ─────────────────────────────────
sep("Test 3 — Distribution des scores sur tous les coups légaux (depth 1 vs 3)")
for depth in [1, 3]:
    scores_per_move = {}
    for m in legal2:
        child = state2.apply_move(m)
        v = ev.evaluate(child)
        scores_per_move[m] = -v if child.player != state2.player else v

    vals = list(scores_per_move.values())
    top3 = sorted(scores_per_move.items(), key=lambda x: x[1], reverse=True)[:3]
    print(f"\n  depth={depth}  min={min(vals):+.4f}  max={max(vals):+.4f}  "
          f"std={np.std(vals):.4f}")
    print(f"  Top 3 : " + "  ".join(f"move={m}({v:+.3f})" for m, v in top3))

# ── Verdict ───────────────────────────────────────────────────────────────────
sep("VERDICT")
prob = all_same_score and all_same_score2
if prob:
    print("  ✗ Les scores ne changent PAS avec la profondeur.")
    print("    → Le problème vient de search.py ou NeuralEvaluator.")
    print("    → L'UI n'est pas en cause.")
else:
    print("  ✓ Les scores changent bien avec la profondeur.")
    print("    → Le problème vient de l'UI (l'agent n'est pas rechargé).")
    print("    → Vérifier _reload_agents() dans DebugTab.")
print()
