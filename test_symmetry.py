"""
test_symmetry.py — Vérifie si le réseau distingue les deux joueurs
===================================================================
Si le réseau ignore le canal 19 (qui joue), v(état, J1) ≈ v(état, J2).
Si le réseau l'utilise correctement, v(état, J1) ≈ -v(état, J2).

Usage :
  python test_symmetry.py --checkpoint models/alphazero/iter_0003.pth
  python test_symmetry.py --checkpoint models/best_uttt_model.pth
"""

import argparse
import random
import numpy as np
from game import UTTTState, check_winner_small

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--device", default=None)
args = parser.parse_args()

# Charge le bon évaluateur
import torch
sd   = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
keys = list(sd.keys())
is_light = any("p_conv" in k or ".net.0." in k for k in keys)

if is_light:
    from bot_mcts import LightEvaluator
    filters = sd["stem.0.weight"].shape[0]
    blocks  = max(int(k.split(".")[1]) for k in keys if k.startswith("res_blocks.")) + 1
    ev = LightEvaluator(args.checkpoint, device=args.device,
                        num_filters=filters, num_res_blocks=blocks)
else:
    from model import NeuralEvaluator
    ev = NeuralEvaluator(args.checkpoint, device=args.device)

print(f"\n{'═'*55}")
print(f"  TEST SYMÉTRIE JOUEUR")
print(f"  Checkpoint : {args.checkpoint}")
print(f"{'═'*55}")

# ── Test 1 : même position, joueur alterné ────────────────────────────────────
print("\n  [1] Même état, joueur alterné — v doit changer de signe")
print(f"  {'Position':<30} {'J1 joue':>10} {'J2 joue':>10} {'Somme':>8} {'OK?':>6}")
print(f"  {'─'*30} {'─'*10} {'─'*10} {'─'*8} {'─'*6}")

random.seed(42)
sommes = []
for label, moves in [
    ("Initial (vide)",         []),
    ("Après 5 coups",          [40, 4, 36, 0, 9]),
    ("Après 10 coups",         [40, 4, 36, 0, 9, 81//2, 3, 27, 6, 54]),
    ("J1 a 1 SG gagnée",       None),
]:
    if moves is None:
        # Construire une position où J1 a gagné SG0
        s = UTTTState.initial()
        s.board      = s.board.copy()
        s.meta_board = s.meta_board.copy()
        for cell in [0,1,2]: s.board[cell] = 1
        s.meta_board[0] = 1
        s.board[27] = 2; s.board[28] = 2
        s.active_idx = -1
        s.player = 1
        s._winner = check_winner_small(s.meta_board)
    else:
        s = UTTTState.initial()
        for m in moves:
            if not s.is_terminal and m in s.legal_moves():
                s = s.apply_move(m)

    # Évaluer avec J1 qui joue
    s1 = UTTTState.__new__(UTTTState)
    s1.board      = s.board.copy()
    s1.meta_board = s.meta_board.copy()
    s1.active_idx = s.active_idx
    s1.player     = 1
    s1._winner    = check_winner_small(s1.meta_board)
    v1 = ev.evaluate(s1)

    # Évaluer avec J2 qui joue (même position)
    s2 = UTTTState.__new__(UTTTState)
    s2.board      = s.board.copy()
    s2.meta_board = s.meta_board.copy()
    s2.active_idx = s.active_idx
    s2.player     = 2
    s2._winner    = check_winner_small(s2.meta_board)
    v2 = ev.evaluate(s2)

    somme = v1 + v2
    sommes.append(abs(somme))
    ok = "✓" if abs(somme) < 0.2 else "✗"
    print(f"  {label:<30} {v1:>+10.4f} {v2:>+10.4f} {somme:>+8.4f} {ok:>6}")

mean_somme = np.mean(sommes)
print(f"\n  Somme moyenne |v1+v2| : {mean_somme:.4f}  (idéal → 0.0)")
if mean_somme < 0.1:
    print("  ✓ Le réseau est symétrique — il distingue bien les deux joueurs.")
elif mean_somme < 0.5:
    print("  ~ Symétrie partielle — le réseau perçoit partiellement qui joue.")
else:
    print("  ✗ Le réseau IGNORE le joueur courant — canal 19 non utilisé.")
    print("    → La loss AlphaZero doit mieux encoder z par joueur.")

# ── Test 2 : vérifier que le canal 19 change bien le tenseur ─────────────────
print(f"\n{'─'*55}")
print("  [2] Canal 19 dans le tenseur d'entrée")
from model import state_to_tensor

s_j1 = UTTTState.initial(); s_j1.player = 1
s_j2 = UTTTState.initial(); s_j2.player = 2

t1 = state_to_tensor(s_j1.to_string())
t2 = state_to_tensor(s_j2.to_string())

print(f"  Canal 19 quand J1 joue : valeur unique = {np.unique(t1[19])}")
print(f"  Canal 19 quand J2 joue : valeur unique = {np.unique(t2[19])}")
diff = np.abs(t1[19] - t2[19]).sum()
print(f"  Différence totale canal 19 : {diff:.1f}  (doit être > 0)")
print(f"  {'✓ Canal 19 encode bien le joueur' if diff > 0 else '✗ Canal 19 identique — bug encodage'}")

# ── Test 3 : gradient du canal 19 ────────────────────────────────────────────
print(f"\n{'─'*55}")
print("  [3] Sensibilité du réseau au canal 19")
import torch

t1_tensor = torch.from_numpy(state_to_tensor(s_j1.to_string())).unsqueeze(0)
t2_tensor = torch.from_numpy(state_to_tensor(s_j2.to_string())).unsqueeze(0)

with torch.no_grad():
    lp1, v1_t = ev.model(t1_tensor.to(ev.device))
    lp2, v2_t = ev.model(t2_tensor.to(ev.device))

diff_v = abs(float(v1_t.item()) - float(v2_t.item()))
print(f"  v(état initial, J1 joue) = {float(v1_t.item()):+.4f}")
print(f"  v(état initial, J2 joue) = {float(v2_t.item()):+.4f}")
print(f"  Différence               = {diff_v:.4f}  (doit être > 0.05)")
print(f"  {'✓ Le réseau réagit au joueur' if diff_v > 0.05 else '✗ Réseau insensible au joueur courant'}")

print(f"\n{'═'*55}\n")
