"""
arena.py — Moteur de tournoi générique pour Ultimate Tic-Tac-Toe
=================================================================
Ne contient aucune logique de bot. Les agents sont des modules séparés
que l'on branche via import ou via la CLI.

Architecture
────────────
  arena.py          ← ce fichier (moteur pur)
  bot_alphabeta.py  ← AlphaBetaAgent
  bot_mcts.py       ← MCTSAgent + LightEvaluator
  bot_random.py     ← RandomAgent (baseline)

Utilisation programmatique
──────────────────────────
  from arena         import Arena, StatsReport
  from bot_alphabeta import AlphaBetaAgent
  from bot_mcts      import MCTSAgent, LightEvaluator
  from bot_random    import RandomAgent
  from model         import NeuralEvaluator

  ev1  = NeuralEvaluator("models/best.pth")
  ev2  = LightEvaluator("models/light.pth")

  report = Arena(
      AlphaBetaAgent(ev1, depth=3),
      MCTSAgent(ev2, simulations=200),
  ).run_verbose(n_games=10, title="BATTLE")
  report.print_summary()

Utilisation CLI
───────────────
  python arena.py benchmark --games 20 --depth 3
  python arena.py battle    --games 10 --simulations 200
  python arena.py eval      --games 20 --depth 3
"""

from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from game import UTTTState


# ═════════════════════════════════════════════════════════════════════════════
# AGENT — interface de base à implémenter dans chaque bot_*.py
# ═════════════════════════════════════════════════════════════════════════════

class Agent:
    """
    Interface commune à tout agent jouant à UTTT.

    Pour créer un nouvel agent :
      1. Hériter de Agent (depuis arena import Agent)
      2. Définir self.name  (utilisé dans les rapports)
      3. Implémenter choose_move(state) → (move_index, score)
    """
    name: str = "Agent"

    def choose_move(self, state: UTTTState) -> Tuple[int, float]:
        """
        Retourne (move_index, score).
          move_index : entier 0–80
          score      : estimation de la valeur du coup, ou 0.0
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement choose_move()")

    def __str__(self)  -> str: return self.name
    def __repr__(self) -> str: return f"{self.__class__.__name__}(name={self.name!r})"


# ═════════════════════════════════════════════════════════════════════════════
# GAME RESULT
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class GameResult:
    """Résultat d'une partie. name_j1 / name_j2 correspondent aux agents."""
    winner:  int     # 0 = nul, 1 = J1, 2 = J2
    turns:   int
    name_j1: str
    name_j2: str
    time_j1: float = 0.0
    time_j2: float = 0.0

    @property
    def winner_name(self) -> str:
        if self.winner == 1: return self.name_j1
        if self.winner == 2: return self.name_j2
        return "draw"

    @property
    def total_time(self) -> float:
        return self.time_j1 + self.time_j2


# ═════════════════════════════════════════════════════════════════════════════
# RUNNER DE PARTIES
# ═════════════════════════════════════════════════════════════════════════════

def play_game(
    agent_j1: Agent,
    agent_j2: Agent,
    verbose: bool = False,
) -> GameResult:
    """Joue une partie complète. J1 = X (commence), J2 = O."""
    state  = UTTTState.initial()
    agents = {1: agent_j1, 2: agent_j2}
    times  = {1: 0.0,      2: 0.0}
    SYMS   = {1: "X",      2: "O"}
    turn   = 0

    while not state.is_terminal:
        turn  += 1
        player = state.player
        ag     = agents[player]

        t0 = time.time()
        move, score = ag.choose_move(state)
        times[player] += time.time() - t0

        if verbose:
            print(f"  Tour {turn:3d} | J{player}({SYMS[player]}) [{ag.name}] "
                  f"→ index={move} (SG={move//9}, cell={move%9})  score={score:+.3f}")

        state = state.apply_move(move)

    return GameResult(
        winner  = state.winner,
        turns   = turn,
        name_j1 = agent_j1.name,
        name_j2 = agent_j2.name,
        time_j1 = times[1],
        time_j2 = times[2],
    )


# ═════════════════════════════════════════════════════════════════════════════
# STATS REPORT
# ═════════════════════════════════════════════════════════════════════════════

class StatsReport:
    """
    Agrège les GameResult d'une série de parties et produit les rapports.

    Convention : agent_a est toujours "l'agent de référence" dans les stats,
    quelle que soit la couleur qu'il a jouée (J1 ou J2).
    """
    W = 64

    def __init__(self, name_a: str, name_b: str):
        self.name_a   = name_a
        self.name_b   = name_b
        self._results: list[GameResult] = []

    # ── Alimentation ──────────────────────────────────────────────────────

    def add(self, r: GameResult) -> None:
        self._results.append(r)

    # ── Métriques ─────────────────────────────────────────────────────────

    @property
    def n(self) -> int:
        return len(self._results)

    def wins(self, name: str) -> int:
        return sum(1 for r in self._results if r.winner_name == name)

    def draws(self) -> int:
        return sum(1 for r in self._results if r.winner == 0)

    def win_rate(self, name: str) -> float:
        return self.wins(name) / self.n if self.n else 0.0

    def avg_time(self, name: str) -> float:
        ts = [r.time_j1 if r.name_j1 == name else r.time_j2 for r in self._results]
        return sum(ts) / len(ts) if ts else 0.0

    def avg_turns(self) -> float:
        return sum(r.turns for r in self._results) / self.n if self.n else 0.0

    def wins_as(self, name: str, player_num: int) -> Tuple[int, int]:
        """(victoires, total) quand name joue en J{player_num}."""
        sub = [r for r in self._results
               if (r.name_j1 == name and player_num == 1)
               or (r.name_j2 == name and player_num == 2)]
        return sum(1 for r in sub if r.winner_name == name), len(sub)

    def as_dict(self) -> dict:
        """Métriques principales — utile pour les pipelines automatisés."""
        return {
            "n":          self.n,
            "wins_a":     self.wins(self.name_a),
            "wins_b":     self.wins(self.name_b),
            "draws":      self.draws(),
            "win_rate_a": self.win_rate(self.name_a),
            "win_rate_b": self.win_rate(self.name_b),
            "avg_turns":  self.avg_turns(),
        }

    # ── Affichage ─────────────────────────────────────────────────────────

    def print_header(self, title: str, **params) -> None:
        print(f"\n{'═' * self.W}")
        print(f"  {title}")
        print(f"  A : {self.name_a}")
        print(f"  B : {self.name_b}")
        for k, v in params.items():
            print(f"  {k:<14}: {v}")
        print(f"{'═' * self.W}")

    def print_live(self, i: int, n_total: int, last: GameResult) -> None:
        wa  = self.wins(self.name_a)
        wb  = self.wins(self.name_b)
        dr  = self.draws()
        tag = {self.name_a: "A ✓", self.name_b: "B ✓", "draw": "=  "}.get(
            last.winner_name, "=  "
        )
        print(f"  Partie {i:3d}/{n_total}  [{tag}]  "
              f"A={wa:3d}  B={wb:3d}  D={dr:3d}  "
              f"WR_A={wa/i*100:5.1f}%  ({last.total_time:.1f}s)")

    def print_summary(self) -> None:
        n  = self.n
        wa = self.wins(self.name_a)
        wb = self.wins(self.name_b)
        dr = self.draws()
        wa1, na1 = self.wins_as(self.name_a, 1)
        wa2, na2 = self.wins_as(self.name_a, 2)

        print(f"\n{'─' * self.W}")
        print(f"  RÉSULTATS FINAUX  ({n} parties)")
        print(f"{'─' * self.W}")
        print(f"  {self.name_a:<32} {wa:3d} V  ({wa/n*100:5.1f}%)")
        print(f"  {self.name_b:<32} {wb:3d} V  ({wb/n*100:5.1f}%)")
        print(f"  {'Nuls':<32} {dr:3d}    ({dr/n*100:5.1f}%)")
        print(f"{'─' * self.W}")
        if na1: print(f"  A en J1 : {wa1}/{na1}  ({wa1/na1*100:.0f}%)")
        if na2: print(f"  A en J2 : {wa2}/{na2}  ({wa2/na2*100:.0f}%)")
        print(f"{'─' * self.W}")
        print(f"  Temps moy. A : {self.avg_time(self.name_a):.3f}s/partie")
        print(f"  Temps moy. B : {self.avg_time(self.name_b):.3f}s/partie")
        print(f"  Tours moy.   : {self.avg_turns():.1f}")
        print(f"{'═' * self.W}")
        margin = max(2, int(n * 0.12))
        print()
        if wa > wb + margin:   print(f"  🏆  {self.name_a} domine.")
        elif wb > wa + margin: print(f"  🏆  {self.name_b} domine.")
        else:                  print("  ⚖   Résultat serré — agents équilibrés.")
        print()


# ═════════════════════════════════════════════════════════════════════════════
# ARENA
# ═════════════════════════════════════════════════════════════════════════════

class Arena:
    """
    Orchestre N parties entre deux agents et retourne un StatsReport.

    Méthodes
    ────────
    run(n_games, alternate)              → StatsReport  (silencieux)
    run_verbose(n_games, alternate, ...) → StatsReport  (avec progression live)
    """

    def __init__(self, agent_a: Agent, agent_b: Agent, verbose: bool = False):
        self.agent_a = agent_a
        self.agent_b = agent_b
        self.verbose = verbose   # affiche chaque coup individuel si True

    def _play_one(self, i: int, alternate: bool) -> Tuple[GameResult, bool]:
        """Joue une partie. Retourne (result normalisé, a_était_j1)."""
        a_is_j1 = (not alternate) or (i % 2 == 1)

        if a_is_j1:
            raw = play_game(self.agent_a, self.agent_b, verbose=self.verbose)
        else:
            raw = play_game(self.agent_b, self.agent_a, verbose=self.verbose)

        # Normalise : name_j1 = agent_a pour que wins_as() soit cohérent
        result = GameResult(
            winner  = raw.winner if a_is_j1 else (3 - raw.winner if raw.winner else 0),
            turns   = raw.turns,
            name_j1 = self.agent_a.name,
            name_j2 = self.agent_b.name,
            time_j1 = raw.time_j1 if a_is_j1 else raw.time_j2,
            time_j2 = raw.time_j2 if a_is_j1 else raw.time_j1,
        )
        return result, a_is_j1

    def run(self, n_games: int = 10, alternate: bool = True) -> StatsReport:
        """Lance n_games parties en silence. Retourne le StatsReport."""
        report = StatsReport(self.agent_a.name, self.agent_b.name)
        for i in range(1, n_games + 1):
            result, _ = self._play_one(i, alternate)
            report.add(result)
        return report

    def run_verbose(
        self,
        n_games: int   = 10,
        alternate: bool = True,
        title: str     = "TOURNOI",
        **header_params,
    ) -> StatsReport:
        """Lance n_games parties avec progression live et en-tête."""
        report = StatsReport(self.agent_a.name, self.agent_b.name)
        report.print_header(title, parties=n_games, **header_params)

        for i in range(1, n_games + 1):
            result, _ = self._play_one(i, alternate)
            report.add(result)
            report.print_live(i, n_games, result)

        return report


# ═════════════════════════════════════════════════════════════════════════════
# CLI — point d'entrée autonome
# ═════════════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="arena",
        description="Moteur de tournoi UTTT  —  modes : benchmark | battle | eval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python arena.py benchmark --games 20 --depth 3
  python arena.py battle    --games 10 --simulations 200
  python arena.py eval      --depth 3
        """,
    )
    sub = root.add_subparsers(dest="mode", required=True)

    def common(p):
        p.add_argument("--checkpoint1", default="models/best_uttt_model.pth")
        p.add_argument("--device",  default=None)
        p.add_argument("--seed",    type=int, default=42)
        p.add_argument("--verbose", action="store_true")
        return p

    pb = common(sub.add_parser("benchmark", help="AlphaBeta vs Random"))
    pb.add_argument("--games", type=int, default=20)
    pb.add_argument("--depth", type=int, default=3)
    pb.add_argument("--top-k", type=int, default=None, dest="top_k")

    pt = common(sub.add_parser("battle", help="AlphaBeta vs MCTS"))
    pt.add_argument("--checkpoint2", default=None)
    pt.add_argument("--games",       type=int,   default=10)
    pt.add_argument("--depth",       type=int,   default=3)
    pt.add_argument("--top-k",       type=int,   default=None, dest="top_k")
    pt.add_argument("--simulations", type=int,   default=200)
    pt.add_argument("--c-puct",      type=float, default=1.5,  dest="c_puct")
    pt.add_argument("--temperature", type=float, default=0.1)

    pe = common(sub.add_parser("eval", help="Tests qualité du modèle"))
    pe.add_argument("--games", type=int, default=20)
    pe.add_argument("--depth", type=int, default=3)

    return root


def _cli_benchmark(args) -> None:
    from model         import NeuralEvaluator
    from bot_alphabeta import AlphaBetaAgent
    from bot_random    import RandomAgent

    ev  = NeuralEvaluator(args.checkpoint1, device=args.device)
    rep = Arena(
        AlphaBetaAgent(ev, depth=args.depth, top_k=args.top_k),
        RandomAgent(),
        verbose=args.verbose,
    ).run_verbose(n_games=args.games, title="BENCHMARK — AlphaBeta vs Random",
                  depth=args.depth, top_k=args.top_k)
    rep.print_summary()


def _cli_battle(args) -> None:
    from model         import NeuralEvaluator
    from bot_alphabeta import AlphaBetaAgent
    from bot_mcts      import MCTSAgent, LightEvaluator

    ev1 = NeuralEvaluator(args.checkpoint1, device=args.device)
    ev2 = LightEvaluator(args.checkpoint2,  device=args.device)
    rep = Arena(
        AlphaBetaAgent(ev1, depth=args.depth, top_k=args.top_k),
        MCTSAgent(ev2, simulations=args.simulations,
                  c_puct=args.c_puct, temperature=args.temperature),
        verbose=args.verbose,
    ).run_verbose(n_games=args.games, title="BATTLE — AlphaBeta vs MCTS",
                  depth=args.depth, simulations=args.simulations)
    rep.print_summary()


def _cli_eval(args) -> None:
    from model import NeuralEvaluator
    ev = NeuralEvaluator(args.checkpoint1, device=args.device)
    _run_eval(ev, depth=args.depth, n_games=args.games, seed=args.seed)


def _run_eval(ev, depth: int = 3, n_games: int = 20, seed: int = 42) -> None:
    """Tests qualité — appelable depuis CLI ou depuis un script externe."""
    from bot_alphabeta import AlphaBetaAgent
    from bot_random    import RandomAgent

    random.seed(seed); np.random.seed(seed)

    def sep(t=""): print(f"\n{'─'*58}\n  {t}\n{'─'*58}" if t else f"\n{'─'*58}")

    print("═" * 58)
    print("  TESTS QUALITÉ DU MODÈLE")
    print("═" * 58)

    sep("1. Architecture")
    total = sum(p.numel() for p in ev.model.parameters())
    print(f"  Paramètres : {total:,} (~{total/1e6:.1f}M)  |  Device : {ev.device}")

    sep("2. Symétrie joueur")
    s  = UTTTState.initial()
    v1 = ev.evaluate(s)
    s2 = UTTTState.initial(); s2.player = 2
    v2 = ev.evaluate(s2)
    sym_ok = abs(v1 + v2) < 0.2
    print(f"  V(J1 joue)={v1:+.4f}  V(J2 joue)={v2:+.4f}  somme={v1+v2:+.4f}")
    print(f"  {'✓ Symétrique' if sym_ok else '✗ Asymétrique — modèle biaisé'}")

    sep("3. Dispersion (50 états aléatoires)")
    vals, s = [], UTTTState.initial()
    for _ in range(50):
        if s.is_terminal: break
        s = s.apply_move(random.choice(s.legal_moves()))
        vals.append(ev.evaluate(s))
    v = np.array(vals)
    print(f"  Min={v.min():+.4f}  Max={v.max():+.4f}  Std={v.std():.4f}")
    print(f"  {'✓ Bonne dispersion' if v.std()>=0.15 else ('~ Faible' if v.std()>=0.05 else '✗ Quasi-constant')}")

    sep("4. Policy head")
    s  = UTTTState.initial()
    lp = ev.policy_logprobs(s)
    pr = np.exp(lp)
    legal = s.legal_moves()
    illeg = [i for i in range(81) if i not in legal]
    pl = pr[legal].sum()
    print(f"  Prob. légaux={pl:.4f}  {'✓' if pl>0.5 else '✗'}  |  illégaux={pr[illeg].sum():.4f}")
    for rank, idx in enumerate(np.argsort(lp)[::-1][:5], 1):
        print(f"    {rank}. idx={idx:2d}  prob={pr[idx]:.4f}  "
              f"{'légal  ' if idx in legal else 'ILLEGAL'}")

    sep(f"5. Mini-benchmark vs Random ({n_games} parties, depth={depth})")
    bot  = AlphaBetaAgent(ev, depth=depth)
    rand = RandomAgent()
    rep  = Arena(bot, rand).run_verbose(n_games=n_games,
                                        title=f"Mini-benchmark (depth={depth})")
    wr = rep.win_rate(bot.name) * 100
    print(f"  Win rate : {wr:.1f}%  "
          f"{'✓ Excellent' if wr>=80 else ('~ Correct' if wr>=60 else '✗ Insuffisant')}")

    sep("VERDICT")
    issues = []
    if not sym_ok:      issues.append("symétrie douteuse")
    if v.std() < 0.05:  issues.append("discrimination nulle")
    if pl < 0.5:        issues.append("policy dispersée")
    if wr < 60:         issues.append(f"win rate faible ({wr:.0f}%)")
    print(f"  {'⚠  ' + ', '.join(issues) if issues else '✓ Modèle sain.'}")
    if issues: print("     Ré-entraîner ou augmenter la profondeur.")
    print()


def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed)
    {"benchmark": _cli_benchmark,
     "battle":    _cli_battle,
     "eval":      _cli_eval}[args.mode](args)


if __name__ == "__main__":
    main()
