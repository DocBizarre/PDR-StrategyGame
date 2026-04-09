"""
ui.py — Interface graphique unifiée pour Ultimate Tic-Tac-Toe
==============================================================
3 onglets :
  1. Jouer        — Humain vs AlphaBeta ou MCTS
  2. Match        — Bot vs Bot avec stats en live
  3. Entraînement — Courbes loss/win-rate du self-play AlphaZero

Usage :
  python ui.py
"""

from __future__ import annotations

import os
import queue
import threading
import time
import tkinter as tk
from tkinter import font as tkfont, ttk, filedialog, messagebox
from typing import Optional

# ── Palette ──────────────────────────────────────────────────────────────────
C_BG         = "#F8F9FA"
C_BOARD      = "#FFFFFF"
C_CELL_LEGAL = "#D1FAE5"
C_CELL_HOVER = "#6EE7B7"
C_ACTIVE_SUB = "#FEF9C3"
C_INACTIVE   = "#E5E7EB"
C_P1_SUB     = "#DBEAFE"
C_P2_SUB     = "#FEE2E2"
C_DRAW_SUB   = "#E5E7EB"
C_P1         = "#1D4ED8"
C_P2         = "#B91C1C"
C_SEP_THICK  = "#6B7280"
C_SEP_THIN   = "#D1D5DB"
C_TEXT       = "#111827"
C_MUTED      = "#6B7280"
C_THINKING   = "#D97706"
C_BTN        = "#374151"
C_BTN_RED    = "#991B1B"
C_BTN_GREEN  = "#065F46"
C_SCORE_POS  = "#065F46"
C_SCORE_NEG  = "#991B1B"
C_CHART_LOSS = "#1D4ED8"
C_CHART_WR   = "#065F46"

# ── Géométrie plateau ─────────────────────────────────────────────────────────
CELL   = 48
THIN   = 1
THICK  = 5
PAD    = 12
STRIDE = 3 * CELL + 2 * THIN + THICK
SUB_W  = 3 * CELL + 2 * THIN
BOARD  = 3 * STRIDE - THICK + 2 * PAD


# ═════════════════════════════════════════════════════════════════════════════
# HELPERS PLATEAU
# ═════════════════════════════════════════════════════════════════════════════

def _sub_origin(sub):
    sr, sc = divmod(sub, 3)
    return PAD + sc * STRIDE, PAD + sr * STRIDE

def _cell_rect(move):
    sub, cell = divmod(move, 9)
    sr, sc    = divmod(sub, 3)
    cr, cc    = divmod(cell, 3)
    x1 = PAD + sc * STRIDE + cc * (CELL + THIN)
    y1 = PAD + sr * STRIDE + cr * (CELL + THIN)
    return x1, y1, x1 + CELL, y1 + CELL

def _sub_center(sub):
    x0, y0 = _sub_origin(sub)
    return x0 + SUB_W // 2, y0 + SUB_W // 2

def _find_win_line(meta):
    from game import WIN_LINES
    for a, b, c in WIN_LINES:
        v = meta[a]
        if v in (1, 2) and v == meta[b] == meta[c]:
            return (a, b, c), v
    return None, 0


# ═════════════════════════════════════════════════════════════════════════════
# WIDGET PLATEAU (réutilisé dans les deux onglets de jeu)
# ═════════════════════════════════════════════════════════════════════════════

class BoardCanvas(tk.Canvas):
    """Canvas du plateau UTTT. Indépendant de la logique de jeu."""

    def __init__(self, parent, on_click_move=None, **kwargs):
        super().__init__(parent, width=BOARD, height=BOARD,
                         bg=C_BOARD, highlightthickness=1,
                         highlightbackground=C_SEP_THICK, **kwargs)
        self._on_click_move = on_click_move
        self._state     = None
        self._legal_set = set()
        self._hover     = None
        self._interactive = True

        self.bind("<Motion>",   self._on_hover)
        self.bind("<Leave>",    self._on_leave)
        self.bind("<Button-1>", self._on_click)

        self._font_piece = tkfont.Font(family="Segoe UI", size=20, weight="bold")
        self._font_big   = tkfont.Font(family="Segoe UI", size=48, weight="bold")

    def update_state(self, state, legal_set=None, interactive=True):
        self._state       = state
        self._legal_set   = legal_set if legal_set is not None else set()
        self._interactive = interactive
        self._hover       = None
        self.redraw()

    def redraw(self):
        if self._state is None:
            return
        self.delete("all")
        state = self._state

        # Fonds des sous-grilles
        for sub in range(9):
            x0, y0 = _sub_origin(sub)
            m = state.meta_board[sub]
            if   m == 1: bg = C_P1_SUB
            elif m == 2: bg = C_P2_SUB
            elif m == 3: bg = C_DRAW_SUB
            elif state.active_idx in (-1, sub): bg = C_ACTIVE_SUB
            else: bg = C_INACTIVE
            self.create_rectangle(x0, y0, x0 + SUB_W, y0 + SUB_W, fill=bg, outline="")

        # Cases
        for move in range(81):
            x1, y1, x2, y2 = _cell_rect(move)
            piece = state.board[move]
            if piece == 0 and move in self._legal_set:
                fill = C_CELL_HOVER if move == self._hover else C_CELL_LEGAL
            else:
                fill = ""
            self.create_rectangle(x1, y1, x2, y2, fill=fill or "", outline=C_SEP_THIN, width=1)
            if piece:
                self.create_text((x1+x2)//2, (y1+y2)//2,
                                 text="X" if piece == 1 else "O",
                                 font=self._font_piece,
                                 fill=C_P1 if piece == 1 else C_P2)

        # Séparateurs épais
        total = 3 * STRIDE - THICK
        for i in (1, 2):
            xv = PAD + i * STRIDE - THICK
            yh = PAD + i * STRIDE - THICK
            self.create_rectangle(xv, PAD, xv + THICK, PAD + total, fill=C_SEP_THICK, outline="")
            self.create_rectangle(PAD, yh, PAD + total, yh + THICK, fill=C_SEP_THICK, outline="")

        # Grands symboles sur sous-grilles terminées
        for sub in range(9):
            m = state.meta_board[sub]
            if m in (1, 2):
                cx, cy = _sub_center(sub)
                sym    = "X" if m == 1 else "O"
                col    = C_P1 if m == 1 else C_P2
                self.create_text(cx+2, cy+2, text=sym, font=self._font_big, fill="white")
                self.create_text(cx,   cy,   text=sym, font=self._font_big, fill=col)

        # Ligne de victoire
        line, winner = _find_win_line(state.meta_board)
        if line:
            col = C_P1 if winner == 1 else C_P2
            pts = [_sub_center(s) for s in line]
            self.create_line(pts[0][0], pts[0][1], pts[2][0], pts[2][1],
                             fill=col, width=6, capstyle=tk.ROUND)

    def _move_at(self, x, y):
        for move in self._legal_set:
            x1, y1, x2, y2 = _cell_rect(move)
            if x1 <= x <= x2 and y1 <= y <= y2:
                return move
        return None

    def _on_hover(self, event):
        if not self._interactive: return
        move = self._move_at(event.x, event.y)
        if move != self._hover:
            self._hover = move
            self.config(cursor="hand2" if move is not None else "")
            self.redraw()

    def _on_leave(self, event):
        if self._hover is not None:
            self._hover = None
            self.config(cursor="")
            self.redraw()

    def _on_click(self, event):
        if not self._interactive or not self._on_click_move: return
        move = self._move_at(event.x, event.y)
        if move is not None:
            self._hover = None
            self._on_click_move(move)


# ═════════════════════════════════════════════════════════════════════════════
# ONGLET 1 — JOUER (Humain vs IA)
# ═════════════════════════════════════════════════════════════════════════════

class PlayTab(tk.Frame):

    def __init__(self, parent):
        super().__init__(parent, bg=C_BG)
        self._state     = None
        self._legal_set = set()
        self._thinking  = False
        self._agent     = None
        self._human_player = 1
        self._q         = queue.Queue()

        self._build()
        self._poll()

    def _build(self):
        f_h = tkfont.Font(family="Segoe UI", size=12, weight="bold")
        f_s = tkfont.Font(family="Segoe UI", size=9)
        f_l = tkfont.Font(family="Segoe UI", size=9)

        # Config
        cfg = tk.Frame(self, bg=C_BG)
        cfg.pack(fill=tk.X, padx=12, pady=(10, 4))

        tk.Label(cfg, text="Bot :", bg=C_BG, fg=C_TEXT, font=f_s).pack(side=tk.LEFT)
        self._bot_var = tk.StringVar(value="AlphaBeta")
        for v in ("AlphaBeta", "MCTS"):
            tk.Radiobutton(cfg, text=v, variable=self._bot_var, value=v,
                           bg=C_BG, fg=C_TEXT, font=f_s,
                           command=self._on_cfg_change).pack(side=tk.LEFT, padx=4)

        tk.Label(cfg, text="  Vous :", bg=C_BG, fg=C_TEXT, font=f_s).pack(side=tk.LEFT)
        self._side_var = tk.StringVar(value="X (J1)")
        tk.OptionMenu(cfg, self._side_var, "X (J1)", "O (J2)",
                      command=lambda _: self._on_cfg_change()).pack(side=tk.LEFT, padx=4)

        tk.Label(cfg, text="  Profondeur/Sims :", bg=C_BG, fg=C_TEXT, font=f_s).pack(side=tk.LEFT)
        self._strength = tk.Spinbox(cfg, from_=1, to=500, width=5, font=f_s)
        self._strength.delete(0, tk.END); self._strength.insert(0, "3")
        self._strength.pack(side=tk.LEFT, padx=4)

        tk.Label(cfg, text="  Checkpoint :", bg=C_BG, fg=C_TEXT, font=f_s).pack(side=tk.LEFT)
        self._ckpt_var = tk.StringVar(value="models/best_uttt_model.pth")
        tk.Entry(cfg, textvariable=self._ckpt_var, width=28, font=f_s).pack(side=tk.LEFT, padx=2)
        tk.Button(cfg, text="…", font=f_s, command=self._browse_ckpt,
                  bg=C_BTN, fg="white", relief=tk.FLAT, padx=4).pack(side=tk.LEFT)

        # Header
        hdr = tk.Frame(self, bg=C_BG)
        hdr.pack(fill=tk.X, padx=12, pady=(2, 2))
        self._lbl_status = tk.Label(hdr, text="", font=f_h, bg=C_BG, fg=C_TEXT)
        self._lbl_status.pack(side=tk.LEFT)
        self._lbl_score  = tk.Label(hdr, text="", font=f_s, bg=C_BG, fg=C_MUTED)
        self._lbl_score.pack(side=tk.LEFT, padx=12)
        tk.Button(hdr, text="Nouvelle partie", font=f_s,
                  bg=C_BTN, fg="white", relief=tk.FLAT, padx=10, pady=3,
                  command=self._new_game).pack(side=tk.RIGHT)

        # Plateau
        self._board = BoardCanvas(self, on_click_move=self._human_move)
        self._board.pack(padx=12, pady=4)

        # Légende
        leg = tk.Frame(self, bg=C_BG)
        leg.pack(pady=(0, 8))
        for col, txt in [(C_P1, "X Joueur 1"), (C_P2, "O Joueur 2"),
                         (C_CELL_LEGAL, " Case jouable"), (C_ACTIVE_SUB, " SG active")]:
            tk.Canvas(leg, width=12, height=12, bg=col,
                      highlightthickness=1, highlightbackground=C_SEP_THICK).pack(side=tk.LEFT, padx=(8,2))
            tk.Label(leg, text=txt, font=f_s, bg=C_BG, fg=C_MUTED).pack(side=tk.LEFT, padx=(0,4))

        self._new_game()

    def _browse_ckpt(self):
        p = filedialog.askopenfilename(filetypes=[("PyTorch", "*.pth"), ("All", "*.*")])
        if p: self._ckpt_var.set(p)

    def _on_cfg_change(self):
        self._new_game()

    def _load_agent(self):
        bot  = self._bot_var.get()
        ckpt = self._ckpt_var.get()
        try:
            val  = int(self._strength.get())
        except ValueError:
            val = 3
        try:
            if bot == "AlphaBeta":
                from model         import NeuralEvaluator
                from bot_alphabeta import AlphaBetaAgent
                ev = NeuralEvaluator(ckpt)
                return AlphaBetaAgent(ev, depth=val)
            else:
                from bot_mcts import LightEvaluator, MCTSAgent
                ev = LightEvaluator(ckpt if os.path.exists(ckpt) else None)
                return MCTSAgent(ev, simulations=val)
        except Exception as e:
            messagebox.showerror("Erreur chargement", str(e))
            return None

    def _new_game(self):
        from game import UTTTState
        self._thinking = False
        self._agent    = None
        self._human_player = 1 if "J1" in self._side_var.get() else 2
        self._state    = UTTTState.initial()
        self._legal_set = set(self._state.legal_moves())
        self._board.update_state(self._state, self._legal_set, interactive=True)
        self._lbl_status.config(text="Chargement…", fg=C_MUTED)
        self._lbl_score.config(text="")

        def load():
            ag = self._load_agent()
            self._q.put(("agent_ready", ag))
        threading.Thread(target=load, daemon=True).start()

    def _human_move(self, move):
        if self._thinking or self._state is None or self._state.is_terminal: return
        if self._state.player != self._human_player: return
        self._apply_move(move)

    def _apply_move(self, move):
        self._state     = self._state.apply_move(move)
        self._legal_set = set(self._state.legal_moves()) if not self._state.is_terminal else set()
        interactive     = (not self._state.is_terminal) and (self._state.player == self._human_player)
        self._board.update_state(self._state, self._legal_set, interactive=interactive)
        self._update_status()
        if not self._state.is_terminal and self._state.player != self._human_player:
            self._thinking = True
            self._update_status()
            ag = self._agent
            state = self._state
            def run():
                move, score = ag.choose_move(state)
                self._q.put(("ai_move", move, score))
            threading.Thread(target=run, daemon=True).start()

    def _update_status(self):
        if self._state is None: return
        if self._thinking:
            self._lbl_status.config(text="IA réfléchit…", fg=C_THINKING); return
        if self._state.is_terminal:
            w = self._state.winner
            if w == 0:   self._lbl_status.config(text="Partie nulle.", fg=C_MUTED)
            elif w == self._human_player: self._lbl_status.config(text="Vous avez gagné !", fg=C_P1)
            else:        self._lbl_status.config(text="L'IA a gagné.", fg=C_P2)
            return
        cp  = self._state.player
        col = C_P1 if cp == 1 else C_P2
        who = "Votre tour" if cp == self._human_player else "Tour de l'IA"
        self._lbl_status.config(text=f"{who}  ({'X' if cp==1 else 'O'})", fg=col)

    def _poll(self):
        try:
            while True:
                msg = self._q.get_nowait()
                if msg[0] == "agent_ready":
                    self._agent = msg[1]
                    if self._agent is None: return
                    self._update_status()
                    if self._state and not self._state.is_terminal and self._state.player != self._human_player:
                        self._thinking = True
                        self._update_status()
                        ag, st = self._agent, self._state
                        def run():
                            mv, sc = ag.choose_move(st)
                            self._q.put(("ai_move", mv, sc))
                        threading.Thread(target=run, daemon=True).start()
                elif msg[0] == "ai_move":
                    self._thinking = False
                    _, move, score = msg
                    sign = "+" if score >= 0 else ""
                    col  = C_SCORE_POS if score >= 0 else C_SCORE_NEG
                    self._lbl_score.config(text=f"Éval : {sign}{score:.3f}", fg=col)
                    self._apply_move(move)
        except queue.Empty:
            pass
        self.after(50, self._poll)


# ═════════════════════════════════════════════════════════════════════════════
# ONGLET 2 — MATCH (Bot vs Bot)
# ═════════════════════════════════════════════════════════════════════════════

class MatchTab(tk.Frame):

    def __init__(self, parent):
        super().__init__(parent, bg=C_BG)
        self._running   = False
        self._q         = queue.Queue()
        self._results   = {"A": 0, "B": 0, "D": 0}
        self._total     = 0
        self._build()
        self._poll()

    def _build(self):
        f_s = tkfont.Font(family="Segoe UI", size=9)
        f_h = tkfont.Font(family="Segoe UI", size=11, weight="bold")
        f_m = tkfont.Font(family="Segoe UI", size=10)

        # ── Config agents ──────────────────────────────────────────────────
        for idx, label in enumerate(["Agent A", "Agent B"]):
            grp = tk.LabelFrame(self, text=label, bg=C_BG, fg=C_TEXT, font=f_s, padx=8, pady=4)
            grp.pack(fill=tk.X, padx=12, pady=(8 if idx == 0 else 2, 0))

            row = tk.Frame(grp, bg=C_BG); row.pack(fill=tk.X)
            tk.Label(row, text="Type :", bg=C_BG, fg=C_TEXT, font=f_s).pack(side=tk.LEFT)
            var_type = tk.StringVar(value="AlphaBeta")
            for v in ("AlphaBeta", "MCTS", "Random"):
                tk.Radiobutton(row, text=v, variable=var_type, value=v,
                               bg=C_BG, font=f_s).pack(side=tk.LEFT, padx=3)

            row2 = tk.Frame(grp, bg=C_BG); row2.pack(fill=tk.X, pady=(2,0))
            tk.Label(row2, text="Checkpoint :", bg=C_BG, fg=C_TEXT, font=f_s).pack(side=tk.LEFT)
            var_ckpt = tk.StringVar(value="models/best_uttt_model.pth" if idx == 0 else "models/alphazero/best.pth")
            tk.Entry(row2, textvariable=var_ckpt, width=32, font=f_s).pack(side=tk.LEFT, padx=2)
            tk.Button(row2, text="…", font=f_s, bg=C_BTN, fg="white", relief=tk.FLAT, padx=4,
                      command=lambda v=var_ckpt: self._browse(v)).pack(side=tk.LEFT)
            tk.Label(row2, text="  Force :", bg=C_BG, fg=C_TEXT, font=f_s).pack(side=tk.LEFT)
            var_str = tk.Spinbox(row2, from_=1, to=500, width=5, font=f_s)
            var_str.delete(0, tk.END); var_str.insert(0, "3" if idx == 0 else "100")
            var_str.pack(side=tk.LEFT, padx=4)

            if idx == 0:
                self._a_type, self._a_ckpt, self._a_str = var_type, var_ckpt, var_str
            else:
                self._b_type, self._b_ckpt, self._b_str = var_type, var_ckpt, var_str

        # ── Contrôles ──────────────────────────────────────────────────────
        ctrl = tk.Frame(self, bg=C_BG); ctrl.pack(fill=tk.X, padx=12, pady=6)
        tk.Label(ctrl, text="Parties :", bg=C_BG, fg=C_TEXT, font=f_s).pack(side=tk.LEFT)
        self._n_games = tk.Spinbox(ctrl, from_=2, to=200, width=5, font=f_s)
        self._n_games.delete(0, tk.END); self._n_games.insert(0, "10")
        self._n_games.pack(side=tk.LEFT, padx=4)

        self._btn_start = tk.Button(ctrl, text="▶  Lancer", font=f_s,
                                    bg=C_BTN_GREEN, fg="white", relief=tk.FLAT,
                                    padx=12, pady=4, command=self._start)
        self._btn_start.pack(side=tk.LEFT, padx=8)
        self._btn_stop = tk.Button(ctrl, text="■  Stop", font=f_s,
                                   bg=C_BTN_RED, fg="white", relief=tk.FLAT,
                                   padx=12, pady=4, command=self._stop, state=tk.DISABLED)
        self._btn_stop.pack(side=tk.LEFT)

        # ── Stats live ─────────────────────────────────────────────────────
        stats = tk.Frame(self, bg=C_BG); stats.pack(fill=tk.X, padx=12, pady=4)
        self._lbl_progress = tk.Label(stats, text="", font=f_h, bg=C_BG, fg=C_TEXT)
        self._lbl_progress.pack(side=tk.LEFT)

        # Barre de progression
        self._pbar = ttk.Progressbar(self, orient=tk.HORIZONTAL, mode="determinate")
        self._pbar.pack(fill=tk.X, padx=12, pady=2)

        # Scores
        sc_frame = tk.Frame(self, bg=C_BG); sc_frame.pack(fill=tk.X, padx=12, pady=4)
        self._lbl_a = tk.Label(sc_frame, text="A : 0", font=f_h, bg=C_BG, fg=C_P1, width=14)
        self._lbl_a.pack(side=tk.LEFT)
        self._lbl_d = tk.Label(sc_frame, text="Nuls : 0", font=f_h, bg=C_BG, fg=C_MUTED, width=12)
        self._lbl_d.pack(side=tk.LEFT)
        self._lbl_b = tk.Label(sc_frame, text="B : 0", font=f_h, bg=C_BG, fg=C_P2, width=14)
        self._lbl_b.pack(side=tk.LEFT)

        # Plateau en direct
        self._board = BoardCanvas(self)
        self._board.pack(padx=12, pady=4)

        # Log
        self._log = tk.Text(self, height=5, font=tkfont.Font(family="Consolas", size=8),
                            bg="#1F2937", fg="#D1D5DB", relief=tk.FLAT, state=tk.DISABLED)
        self._log.pack(fill=tk.X, padx=12, pady=(0, 8))

    def _browse(self, var):
        p = filedialog.askopenfilename(filetypes=[("PyTorch", "*.pth"), ("All", "*.*")])
        if p: var.set(p)

    def _make_agent(self, type_var, ckpt_var, str_var):
        t    = type_var.get()
        ckpt = ckpt_var.get()
        try: val = int(str_var.get())
        except ValueError: val = 3
        if t == "Random":
            from bot_random import RandomAgent
            return RandomAgent()
        elif t == "AlphaBeta":
            from model         import NeuralEvaluator
            from bot_alphabeta import AlphaBetaAgent
            return AlphaBetaAgent(NeuralEvaluator(ckpt), depth=val)
        else:
            from bot_mcts import LightEvaluator, MCTSAgent
            ev = LightEvaluator(ckpt if os.path.exists(ckpt) else None)
            return MCTSAgent(ev, simulations=val)

    def _log_line(self, text):
        self._log.config(state=tk.NORMAL)
        self._log.insert(tk.END, text + "\n")
        self._log.see(tk.END)
        self._log.config(state=tk.DISABLED)

    def _start(self):
        try:
            n = int(self._n_games.get())
        except ValueError:
            return
        self._results = {"A": 0, "B": 0, "D": 0}
        self._total   = n
        self._running = True
        self._btn_start.config(state=tk.DISABLED)
        self._btn_stop.config(state=tk.NORMAL)
        self._pbar["maximum"] = n; self._pbar["value"] = 0
        self._log.config(state=tk.NORMAL); self._log.delete("1.0", tk.END); self._log.config(state=tk.DISABLED)

        def run():
            try:
                ag_a = self._make_agent(self._a_type, self._a_ckpt, self._a_str)
                ag_b = self._make_agent(self._b_type, self._b_ckpt, self._b_str)
            except Exception as e:
                self._q.put(("error", str(e))); return

            from game import UTTTState
            import time

            for i in range(1, n + 1):
                if not self._running: break
                a_is_j1 = (i % 2 == 1)
                j1, j2  = (ag_a, ag_b) if a_is_j1 else (ag_b, ag_a)
                state   = UTTTState.initial()
                turn    = 0
                t0      = time.time()

                while not state.is_terminal:
                    if not self._running: break
                    turn += 1
                    ag    = j1 if state.player == 1 else j2
                    move, _ = ag.choose_move(state)
                    state   = state.apply_move(move)
                    self._q.put(("board", state))

                if not self._running: break
                w = state.winner
                if w == 0:   res = "D"
                elif (w == 1 and a_is_j1) or (w == 2 and not a_is_j1): res = "A"
                else:        res = "B"

                elapsed = time.time() - t0
                self._q.put(("result", i, res, elapsed, ag_a.name, ag_b.name))

            self._q.put(("done",))

        threading.Thread(target=run, daemon=True).start()

    def _stop(self):
        self._running = False

    def _poll(self):
        try:
            while True:
                msg = self._q.get_nowait()
                if msg[0] == "board":
                    state = msg[1]
                    self._board.update_state(state, set(), interactive=False)
                elif msg[0] == "result":
                    _, i, res, elapsed, na, nb = msg
                    self._results[res] += 1
                    a, b, d = self._results["A"], self._results["B"], self._results["D"]
                    wr = a / i * 100
                    self._lbl_a.config(text=f"A : {a}  ({wr:.0f}%)")
                    self._lbl_b.config(text=f"B : {b}  ({b/i*100:.0f}%)")
                    self._lbl_d.config(text=f"Nuls : {d}")
                    self._lbl_progress.config(text=f"Partie {i}/{self._total}  [{res}]  {elapsed:.1f}s")
                    self._pbar["value"] = i
                    self._log_line(f"  #{i:3d}  {res}  A={a:3d} B={b:3d} D={d:3d}  WR_A={wr:5.1f}%  ({elapsed:.1f}s)")
                elif msg[0] == "done":
                    self._running = False
                    self._btn_start.config(state=tk.NORMAL)
                    self._btn_stop.config(state=tk.DISABLED)
                    a, b, d = self._results["A"], self._results["B"], self._results["D"]
                    n = a + b + d
                    self._log_line(f"\n  FINAL : A={a} ({a/n*100:.1f}%)  B={b} ({b/n*100:.1f}%)  Nuls={d}")
                elif msg[0] == "error":
                    messagebox.showerror("Erreur", msg[1])
                    self._running = False
                    self._btn_start.config(state=tk.NORMAL)
                    self._btn_stop.config(state=tk.DISABLED)
        except queue.Empty:
            pass
        self.after(30, self._poll)


# ═════════════════════════════════════════════════════════════════════════════
# ONGLET 3 — ENTRAÎNEMENT (courbes live)
# ═════════════════════════════════════════════════════════════════════════════

class TrainTab(tk.Frame):

    def __init__(self, parent):
        super().__init__(parent, bg=C_BG)
        self._running  = False
        self._q        = queue.Queue()
        self._loss_h   = []   # historique loss total
        self._lossv_h  = []
        self._lossp_h  = []
        self._wr_h     = []   # (iter, win_rate)
        self._build()
        self._poll()

    def _build(self):
        f_s = tkfont.Font(family="Segoe UI", size=9)
        f_h = tkfont.Font(family="Segoe UI", size=10, weight="bold")
        f_c = tkfont.Font(family="Consolas", size=8)

        # Config
        cfg = tk.LabelFrame(self, text="Paramètres", bg=C_BG, fg=C_TEXT, font=f_s, padx=8, pady=6)
        cfg.pack(fill=tk.X, padx=12, pady=(10, 4))

        row1 = tk.Frame(cfg, bg=C_BG); row1.pack(fill=tk.X)
        params = [
            ("Simulations",  "100",  4),
            ("Parties/iter", "25",   4),
            ("Steps/iter",   "100",  4),
            ("Batch",        "256",  5),
            ("Itérations",   "30",   4),
        ]
        self._p = {}
        for label, default, w in params:
            tk.Label(row1, text=f"{label} :", bg=C_BG, fg=C_TEXT, font=f_s).pack(side=tk.LEFT, padx=(8,1))
            sp = tk.Spinbox(row1, from_=1, to=10000, width=w, font=f_s)
            sp.delete(0, tk.END); sp.insert(0, default)
            sp.pack(side=tk.LEFT, padx=(0, 4))
            self._p[label] = sp

        row2 = tk.Frame(cfg, bg=C_BG); row2.pack(fill=tk.X, pady=(4, 0))
        tk.Label(row2, text="Checkpoint :", bg=C_BG, fg=C_TEXT, font=f_s).pack(side=tk.LEFT, padx=(8,1))
        self._ckpt_var = tk.StringVar(value="models/alphazero/best.pth")
        tk.Entry(row2, textvariable=self._ckpt_var, width=34, font=f_s).pack(side=tk.LEFT, padx=2)
        tk.Button(row2, text="…", font=f_s, bg=C_BTN, fg="white", relief=tk.FLAT, padx=4,
                  command=self._browse_ckpt).pack(side=tk.LEFT)
        tk.Label(row2, text="  Dossier sortie :", bg=C_BG, fg=C_TEXT, font=f_s).pack(side=tk.LEFT, padx=(8,1))
        self._out_var = tk.StringVar(value="models/alphazero")
        tk.Entry(row2, textvariable=self._out_var, width=20, font=f_s).pack(side=tk.LEFT, padx=2)

        # Boutons
        ctrl = tk.Frame(self, bg=C_BG); ctrl.pack(fill=tk.X, padx=12, pady=4)
        self._btn_start = tk.Button(ctrl, text="▶  Lancer l'entraînement", font=f_s,
                                    bg=C_BTN_GREEN, fg="white", relief=tk.FLAT,
                                    padx=12, pady=5, command=self._start)
        self._btn_start.pack(side=tk.LEFT)
        self._btn_stop = tk.Button(ctrl, text="■  Stop", font=f_s,
                                   bg=C_BTN_RED, fg="white", relief=tk.FLAT,
                                   padx=12, pady=5, command=self._stop, state=tk.DISABLED)
        self._btn_stop.pack(side=tk.LEFT, padx=8)
        self._lbl_eta = tk.Label(ctrl, text="", font=f_s, bg=C_BG, fg=C_MUTED)
        self._lbl_eta.pack(side=tk.LEFT)

        # Graphiques (canvas tkinter, pas matplotlib)
        charts = tk.Frame(self, bg=C_BG); charts.pack(fill=tk.BOTH, expand=True, padx=12)

        self._canvas_loss = tk.Canvas(charts, bg="#1F2937", height=160, highlightthickness=0)
        self._canvas_loss.pack(fill=tk.X, pady=(4, 2))
        self._canvas_wr   = tk.Canvas(charts, bg="#1F2937", height=120, highlightthickness=0)
        self._canvas_wr.pack(fill=tk.X, pady=(2, 4))

        # Log
        self._log = tk.Text(self, height=6, font=f_c,
                            bg="#1F2937", fg="#D1D5DB", relief=tk.FLAT, state=tk.DISABLED)
        self._log.pack(fill=tk.X, padx=12, pady=(0, 8))

    def _browse_ckpt(self):
        p = filedialog.askopenfilename(filetypes=[("PyTorch", "*.pth"), ("All", "*.*")])
        if p: self._ckpt_var.set(p)

    def _log_line(self, text):
        self._log.config(state=tk.NORMAL)
        self._log.insert(tk.END, text + "\n")
        self._log.see(tk.END)
        self._log.config(state=tk.DISABLED)

    def _iv(self, key, default=1):
        try: return int(self._p[key].get())
        except: return default

    def _start(self):
        self._running = True
        self._loss_h.clear(); self._lossv_h.clear(); self._lossp_h.clear(); self._wr_h.clear()
        self._btn_start.config(state=tk.DISABLED)
        self._btn_stop.config(state=tk.NORMAL)
        self._log.config(state=tk.NORMAL); self._log.delete("1.0", tk.END); self._log.config(state=tk.DISABLED)
        self._draw_loss(); self._draw_wr()

        sims      = self._iv("Simulations",  100)
        gpi       = self._iv("Parties/iter", 25)
        steps     = self._iv("Steps/iter",   100)
        batch     = self._iv("Batch",        256)
        iters     = self._iv("Itérations",   30)
        ckpt      = self._ckpt_var.get()
        out_dir   = self._out_var.get()
        q         = self._q

        def run():
            import random, numpy as np, torch
            from bot_mcts      import LightEvaluator, MCTSEngine
            from replay_buffer import ReplayBuffer
            from trainer       import Trainer
            from arena         import Arena
            from bot_mcts      import MCTSAgent
            from self_play     import play_self_play_game

            resume = ckpt if os.path.exists(ckpt) else None
            current  = LightEvaluator(resume)
            champion = LightEvaluator()
            champion.clone_weights_from(current)
            buffer   = ReplayBuffer(max_size=50_000)
            trainer  = Trainer(current, buffer, total_steps=iters * steps)
            engine   = MCTSEngine(current, simulations=sims)
            t_start  = time.time()

            for it in range(1, iters + 1):
                if not self._running: break
                q.put(("log", f"── Itération {it}/{iters} ──"))

                # Self-play
                n_ex = 0
                for _ in range(gpi):
                    if not self._running: break
                    exs = play_self_play_game(engine)
                    buffer.push_game(exs)
                    n_ex += len(exs)
                q.put(("log", f"  Self-play : {n_ex} ex  buffer={len(buffer)}"))

                # Entraînement
                if buffer.is_ready:
                    summary = trainer.train_epoch(n_steps=steps, batch_size=batch, log_every=0)
                    q.put(("loss", it, summary["loss"], summary["loss_v"], summary["loss_p"]))
                    q.put(("log", f"  Train : loss={summary['loss']:.4f}  v={summary['loss_v']:.4f}  p={summary['loss_p']:.4f}"))

                # Éval tous les 3 iter
                if it % 3 == 0 and buffer.is_ready:
                    chal = MCTSAgent(current, simulations=50, temperature=0.1)
                    chmp = MCTSAgent(champion, simulations=50, temperature=0.1)
                    rep  = Arena(chal, chmp).run(n_games=10)
                    wr   = rep.win_rate(chal.name)
                    q.put(("wr", it, wr))
                    q.put(("log", f"  Eval WR={wr*100:.1f}%"))
                    if wr >= 0.55:
                        champion.clone_weights_from(current)
                        path = os.path.join(out_dir, "best.pth")
                        current.save(path)
                        q.put(("log", f"  ✓ Nouveau champion sauvegardé → {path}"))

                # Checkpoint
                os.makedirs(out_dir, exist_ok=True)
                current.save(os.path.join(out_dir, f"iter_{it:04d}.pth"))

                elapsed = time.time() - t_start
                eta     = elapsed / it * (iters - it)
                q.put(("eta", elapsed, eta, it, iters))

            q.put(("done",))

        threading.Thread(target=run, daemon=True).start()

    def _stop(self):
        self._running = False

    def _draw_loss(self):
        c = self._canvas_loss
        c.delete("all")
        w, h = c.winfo_width() or 600, c.winfo_height() or 160
        c.create_text(8, 8, text="Loss (total / valeur / politique)", anchor="nw",
                      fill="#9CA3AF", font=("Segoe UI", 8))
        if len(self._loss_h) < 2: return
        pad = 30
        mx  = max(self._loss_h) or 1
        def px(i): return pad + (w - 2*pad) * i / (len(self._loss_h) - 1)
        def py(v): return h - pad - (h - 2*pad) * min(v / mx, 1.0)
        for data, col, lbl in [
            (self._loss_h,  C_CHART_LOSS, "total"),
            (self._lossv_h, "#F59E0B",    "valeur"),
            (self._lossp_h, "#EF4444",    "policy"),
        ]:
            pts = [(px(i), py(v)) for i, v in enumerate(data)]
            for j in range(1, len(pts)):
                c.create_line(pts[j-1][0], pts[j-1][1], pts[j][0], pts[j][1],
                              fill=col, width=2)

    def _draw_wr(self):
        c = self._canvas_wr
        c.delete("all")
        w, h = c.winfo_width() or 600, c.winfo_height() or 120
        c.create_text(8, 8, text="Win rate challenger vs champion", anchor="nw",
                      fill="#9CA3AF", font=("Segoe UI", 8))
        if len(self._wr_h) < 2: return
        pad = 30
        iters = [x[0] for x in self._wr_h]
        wrs   = [x[1] for x in self._wr_h]
        mi, ma = min(iters), max(iters)
        def px(i): return pad + (w - 2*pad) * (i - mi) / max(ma - mi, 1)
        def py(v): return h - pad - (h - 2*pad) * v
        # Ligne 55%
        y55 = py(0.55)
        c.create_line(pad, y55, w - pad, y55, fill="#6B7280", dash=(4, 2), width=1)
        c.create_text(w - pad - 2, y55 - 4, text="55%", anchor="e", fill="#6B7280", font=("Segoe UI", 7))
        # Courbe WR
        pts = [(px(iters[i]), py(wrs[i])) for i in range(len(wrs))]
        for j in range(1, len(pts)):
            c.create_line(pts[j-1][0], pts[j-1][1], pts[j][0], pts[j][1],
                          fill=C_CHART_WR, width=2)

    def _poll(self):
        try:
            while True:
                msg = self._q.get_nowait()
                if msg[0] == "loss":
                    _, it, l, lv, lp = msg
                    self._loss_h.append(l); self._lossv_h.append(lv); self._lossp_h.append(lp)
                    self._draw_loss()
                elif msg[0] == "wr":
                    _, it, wr = msg
                    self._wr_h.append((it, wr))
                    self._draw_wr()
                elif msg[0] == "log":
                    self._log_line(msg[1])
                elif msg[0] == "eta":
                    _, elapsed, eta, it, total = msg
                    self._lbl_eta.config(
                        text=f"Itération {it}/{total}  |  écoulé {elapsed/60:.1f}min  |  ETA {eta/60:.1f}min"
                    )
                elif msg[0] == "done":
                    self._running = False
                    self._btn_start.config(state=tk.NORMAL)
                    self._btn_stop.config(state=tk.DISABLED)
                    self._log_line("✓ Entraînement terminé.")
        except queue.Empty:
            pass
        self.after(100, self._poll)


# ═════════════════════════════════════════════════════════════════════════════
# ONGLET 4 — DEBUG (heatmap des scores + bot vs bot pas à pas)
# ═════════════════════════════════════════════════════════════════════════════

def _lerp_color(t: float, lo=(219,234,254), hi=(185,28,28)) -> str:
    """Interpole entre bleu clair (lo, t=0) et rouge (hi, t=1)."""
    r = int(lo[0] + (hi[0]-lo[0]) * t)
    g = int(lo[1] + (hi[1]-lo[1]) * t)
    b = int(lo[2] + (hi[2]-lo[2]) * t)
    return f"#{r:02x}{g:02x}{b:02x}"


class DebugTab(tk.Frame):
    """
    Onglet Debug — deux sous-modes :
      • Heatmap  : une position quelconque, affiche le score du bot pour chaque coup légal
      • Bot vs Bot pas à pas : avance coup par coup, affiche les scores à chaque tour
    """

    def __init__(self, parent):
        super().__init__(parent, bg=C_BG)
        self._state      = None
        self._legal_set  = set()
        self._scores     = {}      # move → float
        self._agent      = None
        self._agent2     = None    # bot B (mode bot vs bot)
        self._mode       = "heatmap"
        self._auto_play  = False
        self._q          = queue.Queue()
        self._history    = []      # liste de (state, scores_dict, move_joué)

        self._build()
        self._poll()
        self._new_game()

    # ── Construction ──────────────────────────────────────────────────────

    def _build(self):
        f_s = tkfont.Font(family="Segoe UI", size=9)
        f_h = tkfont.Font(family="Segoe UI", size=11, weight="bold")
        f_c = tkfont.Font(family="Consolas", size=8)

        # ── Mode ──────────────────────────────────────────────────────────
        top = tk.Frame(self, bg=C_BG); top.pack(fill=tk.X, padx=12, pady=(10,4))
        self._mode_var = tk.StringVar(value="heatmap")
        tk.Radiobutton(top, text="Heatmap (1 bot)", variable=self._mode_var,
                       value="heatmap", bg=C_BG, font=f_s,
                       command=self._on_mode).pack(side=tk.LEFT)
        tk.Radiobutton(top, text="Bot vs Bot pas à pas", variable=self._mode_var,
                       value="botvsbot", bg=C_BG, font=f_s,
                       command=self._on_mode).pack(side=tk.LEFT, padx=(12,0))

        # ── Config Bot A ───────────────────────────────────────────────────
        self._grp_a = tk.LabelFrame(self, text="Bot A", bg=C_BG, fg=C_TEXT,
                                     font=f_s, padx=8, pady=4)
        self._grp_a.pack(fill=tk.X, padx=12, pady=(4,0))
        self._a_type = tk.StringVar(value="AlphaBeta")
        for v in ("AlphaBeta", "MCTS", "Random"):
            tk.Radiobutton(self._grp_a, text=v, variable=self._a_type,
                           value=v, bg=C_BG, font=f_s).pack(side=tk.LEFT, padx=3)
        tk.Label(self._grp_a, text="  Ckpt :", bg=C_BG, font=f_s).pack(side=tk.LEFT)
        self._a_ckpt = tk.StringVar(value="models/best_uttt_model.pth")
        tk.Entry(self._grp_a, textvariable=self._a_ckpt, width=28, font=f_s).pack(side=tk.LEFT, padx=2)
        tk.Button(self._grp_a, text="…", font=f_s, bg=C_BTN, fg="white",
                  relief=tk.FLAT, padx=4,
                  command=lambda: self._browse(self._a_ckpt)).pack(side=tk.LEFT)
        tk.Label(self._grp_a, text="  Force :", bg=C_BG, font=f_s).pack(side=tk.LEFT)
        self._a_str = tk.Spinbox(self._grp_a, from_=1, to=500, width=5, font=f_s)
        self._a_str.delete(0,tk.END); self._a_str.insert(0,"3")
        self._a_str.pack(side=tk.LEFT, padx=4)

        # ── Config Bot B (mode botvsbot seulement) ─────────────────────────
        self._grp_b = tk.LabelFrame(self, text="Bot B", bg=C_BG, fg=C_TEXT,
                                     font=f_s, padx=8, pady=4)
        self._grp_b.pack(fill=tk.X, padx=12, pady=(2,0))
        self._b_type = tk.StringVar(value="MCTS")
        for v in ("AlphaBeta", "MCTS", "Random"):
            tk.Radiobutton(self._grp_b, text=v, variable=self._b_type,
                           value=v, bg=C_BG, font=f_s).pack(side=tk.LEFT, padx=3)
        tk.Label(self._grp_b, text="  Ckpt :", bg=C_BG, font=f_s).pack(side=tk.LEFT)
        self._b_ckpt = tk.StringVar(value="models/alphazero/best.pth")
        tk.Entry(self._grp_b, textvariable=self._b_ckpt, width=28, font=f_s).pack(side=tk.LEFT, padx=2)
        tk.Button(self._grp_b, text="…", font=f_s, bg=C_BTN, fg="white",
                  relief=tk.FLAT, padx=4,
                  command=lambda: self._browse(self._b_ckpt)).pack(side=tk.LEFT)
        tk.Label(self._grp_b, text="  Force :", bg=C_BG, font=f_s).pack(side=tk.LEFT)
        self._b_str = tk.Spinbox(self._grp_b, from_=1, to=500, width=5, font=f_s)
        self._b_str.delete(0,tk.END); self._b_str.insert(0,"100")
        self._b_str.pack(side=tk.LEFT, padx=4)

        # ── Contrôles ──────────────────────────────────────────────────────
        ctrl = tk.Frame(self, bg=C_BG); ctrl.pack(fill=tk.X, padx=12, pady=6)
        tk.Button(ctrl, text="⟳  Nouvelle partie", font=f_s,
                  bg=C_BTN, fg="white", relief=tk.FLAT, padx=10, pady=4,
                  command=self._new_game).pack(side=tk.LEFT)
        tk.Button(ctrl, text="◀  Précédent", font=f_s,
                  bg=C_BTN, fg="white", relief=tk.FLAT, padx=10, pady=4,
                  command=self._prev_move).pack(side=tk.LEFT, padx=6)
        self._btn_next = tk.Button(ctrl, text="Calculer ▶", font=f_s,
                                   bg=C_BTN_GREEN, fg="white", relief=tk.FLAT,
                                   padx=10, pady=4, command=self._next_move)
        self._btn_next.pack(side=tk.LEFT)
        self._btn_auto = tk.Button(ctrl, text="▶▶  Auto", font=f_s,
                                   bg="#1D4ED8", fg="white", relief=tk.FLAT,
                                   padx=10, pady=4, command=self._toggle_auto)
        self._btn_auto.pack(side=tk.LEFT, padx=6)
        self._lbl_turn = tk.Label(ctrl, text="", font=f_h, bg=C_BG, fg=C_TEXT)
        self._lbl_turn.pack(side=tk.LEFT, padx=12)

        # ── Légende heatmap ────────────────────────────────────────────────
        leg = tk.Frame(self, bg=C_BG); leg.pack(fill=tk.X, padx=12)
        tk.Label(leg, text="Score :", bg=C_BG, fg=C_MUTED, font=f_s).pack(side=tk.LEFT)
        for t, lbl in [(0.0,"−1.0 (perdant)"), (0.25,"−0.5"), (0.5,"0.0"),
                       (0.75,"+0.5"), (1.0,"+1.0 (gagnant)")]:
            col = _lerp_color(t)
            tk.Canvas(leg, width=14, height=14, bg=col,
                      highlightthickness=1, highlightbackground=C_SEP_THICK).pack(side=tk.LEFT, padx=(6,1))
            tk.Label(leg, text=lbl, font=f_s, bg=C_BG, fg=C_MUTED).pack(side=tk.LEFT, padx=(0,4))

        # ── Plateau + panneau latéral ──────────────────────────────────────
        body = tk.Frame(self, bg=C_BG); body.pack(fill=tk.BOTH, expand=True, padx=12, pady=4)

        self._board = BoardCanvas(body, on_click_move=None)
        self._board.pack(side=tk.LEFT)

        # Panneau droit : top 10 coups
        right = tk.Frame(body, bg=C_BG); right.pack(side=tk.LEFT, fill=tk.BOTH,
                                                      expand=True, padx=(12,0))
        tk.Label(right, text="Top coups", font=f_h, bg=C_BG, fg=C_TEXT).pack(anchor="w")
        self._moves_text = tk.Text(right, width=28, font=f_c,
                                   bg="#1F2937", fg="#D1D5DB", relief=tk.FLAT,
                                   state=tk.DISABLED)
        self._moves_text.pack(fill=tk.BOTH, expand=True)

        tk.Label(right, text="Log coups joués", font=f_h, bg=C_BG,
                 fg=C_TEXT).pack(anchor="w", pady=(8,0))
        self._log = tk.Text(right, width=28, font=f_c,
                            bg="#1F2937", fg="#9CA3AF", relief=tk.FLAT,
                            state=tk.DISABLED)
        self._log.pack(fill=tk.BOTH, expand=True)

        self._on_mode()

    # ── Helpers ───────────────────────────────────────────────────────────

    def _browse(self, var):
        p = filedialog.askopenfilename(filetypes=[("PyTorch","*.pth"),("All","*.*")])
        if p: var.set(p)

    def _on_mode(self):
        self._mode = self._mode_var.get()
        state = tk.NORMAL if self._mode == "botvsbot" else tk.DISABLED
        for w in self._grp_b.winfo_children():
            try: w.config(state=state)
            except: pass
        self._new_game()

    def _make_agent(self, type_var, ckpt_var, str_var):
        t    = type_var.get()
        ckpt = ckpt_var.get()
        try: val = int(str_var.get())
        except ValueError: val = 3
        if t == "Random":
            from bot_random import RandomAgent
            return RandomAgent()
        elif t == "AlphaBeta":
            from model         import NeuralEvaluator
            from bot_alphabeta import AlphaBetaAgent
            ev = NeuralEvaluator(ckpt)
            return AlphaBetaAgent(ev, depth=val)
        else:
            from bot_mcts import LightEvaluator, MCTSAgent
            ev = LightEvaluator(ckpt if os.path.exists(ckpt) else None)
            return MCTSAgent(ev, simulations=val)

    def _log_line(self, text, widget=None):
        w = widget or self._log
        w.config(state=tk.NORMAL)
        w.insert(tk.END, text + "\n")
        w.see(tk.END)
        w.config(state=tk.DISABLED)

    def _score_to_t(self, score: float) -> float:
        """Normalise un score [-1, 1] → [0, 1] pour la heatmap."""
        return max(0.0, min(1.0, (score + 1.0) / 2.0))

    # ── Heatmap sur le plateau ────────────────────────────────────────────

    def _redraw_debug(self):
        """Redessine le plateau avec la heatmap des scores superposée."""
        if self._state is None: return

        # Dessin de base
        self._board.update_state(self._state, set(), interactive=False)

        scores  = self._scores
        legal   = list(self._legal_set)
        if not scores or not legal: return

        vals = list(scores.values())
        mn, mx = min(vals), max(vals)
        span   = mx - mn if mx > mn else 1.0

        SYMS = {1: "X", 2: "O"}

        for move in legal:
            if move not in scores: continue
            raw   = scores[move]
            t     = (raw - mn) / span          # relatif dans la plage observée
            color = _lerp_color(t)
            x1, y1, x2, y2 = _cell_rect(move)

            # Fond coloré semi-transparent simulé
            self._board.create_rectangle(x1+1, y1+1, x2-1, y2-1,
                                          fill=color, outline="", stipple="gray50")
            self._board.create_rectangle(x1+1, y1+1, x2-1, y2-1,
                                          fill=color, outline="")

            # Score affiché
            lbl = f"{raw:+.2f}"
            self._board.create_text((x1+x2)//2, (y1+y2)//2,
                                     text=lbl,
                                     font=tkfont.Font(family="Segoe UI", size=7, weight="bold"),
                                     fill="#111827")

        # Re-dessiner les pièces par-dessus
        fp = tkfont.Font(family="Segoe UI", size=16, weight="bold")
        for i in range(81):
            p = self._state.board[i]
            if p:
                x1,y1,x2,y2 = _cell_rect(i)
                self._board.create_text((x1+x2)//2, (y1+y2)//2,
                                         text=SYMS[p], font=fp,
                                         fill=C_P1 if p==1 else C_P2)

    def _update_moves_panel(self):
        """Affiche le top 10 coups triés par score dans le panneau droit."""
        self._moves_text.config(state=tk.NORMAL)
        self._moves_text.delete("1.0", tk.END)

        if not self._scores:
            self._moves_text.config(state=tk.DISABLED)
            return

        sorted_moves = sorted(self._scores.items(), key=lambda x: x[1], reverse=True)
        player = self._state.player if self._state else 1
        who    = "J1 (X)" if player == 1 else "J2 (O)"
        self._moves_text.insert(tk.END, f"Tour {who}\n{'─'*26}\n")
        self._moves_text.insert(tk.END, f"  {'Coup':>4}  {'SG':>3}  {'Cell':>4}  {'Score':>7}\n")
        self._moves_text.insert(tk.END, f"{'─'*26}\n")

        for rank, (move, score) in enumerate(sorted_moves[:15], 1):
            sg   = move // 9
            cell = move % 9
            bar  = "█" * int(abs(score) * 8)
            sign = "+" if score >= 0 else ""
            line = f"  {rank:2d}.  {move:2d}   SG{sg}  c{cell}  {sign}{score:.3f} {bar}\n"
            self._moves_text.insert(tk.END, line)

        self._moves_text.config(state=tk.DISABLED)

    # ── Logique de jeu ────────────────────────────────────────────────────

    def _new_game(self):
        from game import UTTTState
        self._auto_play = False
        self._btn_auto.config(text="▶▶  Auto", bg="#1D4ED8")
        self._history.clear()
        self._scores.clear()
        self._state     = UTTTState.initial()
        self._legal_set = set(self._state.legal_moves())
        self._board.update_state(self._state, self._legal_set, interactive=False)
        self._lbl_turn.config(text="Tour 0 — appuie sur Calculer", fg=C_MUTED)

        self._moves_text.config(state=tk.NORMAL); self._moves_text.delete("1.0",tk.END)
        self._moves_text.config(state=tk.DISABLED)
        self._log.config(state=tk.NORMAL); self._log.delete("1.0",tk.END)
        self._log.config(state=tk.DISABLED)

        # Charger les agents en background
        def load():
            try:
                ag_a = self._make_agent(self._a_type, self._a_ckpt, self._a_str)
                ag_b = self._make_agent(self._b_type, self._b_ckpt, self._b_str) \
                       if self._mode == "botvsbot" else None
                self._q.put(("agents", ag_a, ag_b))
            except Exception as e:
                self._q.put(("error", str(e)))
        threading.Thread(target=load, daemon=True).start()

    def _current_agent(self):
        """Retourne l'agent qui doit jouer dans l'état courant."""
        if self._mode == "heatmap" or self._agent2 is None:
            return self._agent
        return self._agent if self._state.player == 1 else self._agent2

    def _next_move(self):
        """Calcule les scores du bot courant et joue le meilleur coup."""
        if self._state is None or self._state.is_terminal: return
        if self._agent is None: return

        ag = self._current_agent()
        state = self._state
        self._lbl_turn.config(text="Calcul en cours…", fg=C_THINKING)
        self._btn_next.config(state=tk.DISABLED)

        def compute():
            try:
                scores = {}
                legal  = state.legal_moves()

                # Récupère les scores pour chaque coup légal
                if hasattr(ag, 'engine'):
                    # MCTSAgent → on utilise search_with_policy pour avoir la distribution
                    import numpy as np
                    move, pi = ag.engine.search_with_policy(state, temperature=0.5)
                    for m in legal:
                        scores[m] = float(pi[m])
                    best_move = move
                    best_score = float(pi[move])
                elif hasattr(ag, 'evaluator'):
                    # AlphaBetaAgent → on évalue chaque coup à depth=1
                    from search import best_move as ab_best
                    best_move, best_score = ab_best(state, ag.evaluator,
                                                     depth=ag.depth, top_k=ag.top_k,
                                                     verbose=False)
                    # Score pour chaque coup légal à depth=1
                    for m in legal:
                        child = state.apply_move(m)
                        v = ag.evaluator.evaluate(child)
                        # Depuis perspective du joueur courant
                        scores[m] = -v if child.player != state.player else v
                else:
                    # RandomAgent → tous les coups équivalents
                    import random
                    best_move = random.choice(legal)
                    best_score = 0.0
                    for m in legal:
                        scores[m] = 0.0

                self._q.put(("scores", state, scores, best_move, best_score, ag.name))
            except Exception as e:
                self._q.put(("error", str(e)))

        threading.Thread(target=compute, daemon=True).start()

    def _prev_move(self):
        """Revient à l'état précédent."""
        if not self._history: return
        self._state, self._scores = self._history.pop()
        self._legal_set = set(self._state.legal_moves())
        self._redraw_debug()
        self._update_moves_panel()
        turn = len(self._history)
        self._lbl_turn.config(text=f"Tour {turn} — retour arrière", fg=C_MUTED)

    def _toggle_auto(self):
        self._auto_play = not self._auto_play
        if self._auto_play:
            self._btn_auto.config(text="⏸  Pause", bg=C_BTN_RED)
            self._schedule_auto()
        else:
            self._btn_auto.config(text="▶▶  Auto", bg="#1D4ED8")

    def _schedule_auto(self):
        if self._auto_play and not (self._state and self._state.is_terminal):
            self._next_move()

    # ── Polling ───────────────────────────────────────────────────────────

    def _poll(self):
        try:
            while True:
                msg = self._q.get_nowait()

                if msg[0] == "agents":
                    _, ag_a, ag_b = msg
                    self._agent  = ag_a
                    self._agent2 = ag_b
                    self._lbl_turn.config(
                        text=f"Prêt — A:{ag_a.name}" +
                             (f"  B:{ag_b.name}" if ag_b else ""),
                        fg=C_TEXT
                    )

                elif msg[0] == "scores":
                    _, state, scores, best_move, best_score, ag_name = msg

                    # Sauvegarder l'état avant de jouer
                    self._history.append((self._state, dict(self._scores)))
                    self._scores    = scores
                    self._legal_set = set(state.legal_moves())

                    # Afficher la heatmap
                    self._redraw_debug()
                    self._update_moves_panel()

                    # Jouer le coup
                    new_state = state.apply_move(best_move)
                    sg, cell  = best_move // 9, best_move % 9
                    turn      = len(self._history)
                    player    = state.player
                    SYMS      = {1:"X", 2:"O"}

                    self._log_line(
                        f"Tour {turn:2d} | {ag_name[:12]:12s} | "
                        f"J{player}({SYMS[player]}) → SG{sg} c{cell} "
                        f"score={best_score:+.3f}"
                    )

                    self._state     = new_state
                    self._legal_set = set(new_state.legal_moves())
                    self._btn_next.config(state=tk.NORMAL)

                    if new_state.is_terminal:
                        w = new_state.winner
                        msg_end = "Nul !" if w == 0 else f"J{w} ({SYMS[w]}) gagne !"
                        self._lbl_turn.config(text=f"Tour {turn} — {msg_end}", fg=C_P1 if w==1 else C_P2 if w==2 else C_MUTED)
                        self._auto_play = False
                        self._btn_auto.config(text="▶▶  Auto", bg="#1D4ED8")
                        self._log_line(f"{'─'*30}\n  {msg_end}")
                    else:
                        next_ag = self._current_agent()
                        nxt     = next_ag.name if next_ag else "?"
                        self._lbl_turn.config(
                            text=f"Tour {turn} — prochain : {nxt}  (score joué : {best_score:+.3f})",
                            fg=C_P1 if new_state.player==1 else C_P2
                        )
                        if self._auto_play:
                            self.after(400, self._schedule_auto)

                elif msg[0] == "error":
                    messagebox.showerror("Erreur", msg[1])
                    self._auto_play = False
                    self._btn_next.config(state=tk.NORMAL)

        except queue.Empty:
            pass
        self.after(60, self._poll)


# ═════════════════════════════════════════════════════════════════════════════
# FENÊTRE PRINCIPALE
# ═════════════════════════════════════════════════════════════════════════════

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Ultimate Tic-Tac-Toe — IA Studio")
        self.configure(bg=C_BG)
        self.resizable(True, True)

        style = ttk.Style()
        style.theme_use("default")
        style.configure("TNotebook",       background=C_BG, borderwidth=0)
        style.configure("TNotebook.Tab",   background="#E5E7EB", foreground=C_TEXT,
                         font=("Segoe UI", 10), padding=(14, 6))
        style.map("TNotebook.Tab",
                  background=[("selected", C_BG)],
                  foreground=[("selected", C_P1)])

        nb = ttk.Notebook(self)
        nb.pack(fill=tk.BOTH, expand=True)

        nb.add(PlayTab(nb),  text="🎮  Jouer")
        nb.add(MatchTab(nb), text="⚔   Match")
        nb.add(TrainTab(nb), text="📈  Entraînement")
        nb.add(DebugTab(nb), text="🔬  Debug")


if __name__ == "__main__":
    App().mainloop()
