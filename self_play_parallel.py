"""
self_play_parallel.py — Self-play AlphaZero parallèle
======================================================
Architecture :
  InferenceServer  (thread GPU principal)
    ← reçoit des batches d'états de tous les workers
    → retourne (value, policy) en un seul forward GPU

  Worker × N  (threads CPU)
    → jouent des parties MCTS indépendamment
    → délèguent les forwards GPU au InferenceServer via queue

Gain attendu sur RTX 3060/3070 : 3-6× plus de parties/heure
selon le nombre de workers (4-8 recommandé).

Usage
─────
  python self_play_parallel.py                        # 4 workers
  python self_play_parallel.py --workers 8            # 8 workers
  python self_play_parallel.py --resume models/alphazero/best.pth
"""

from __future__ import annotations

import argparse
import os
import queue
import random
import threading
import time
from typing import List, Optional, Tuple

import numpy as np
import torch

from game          import UTTTState
from model         import state_to_tensor
from bot_mcts      import LightEvaluator, MCTSAgent
from replay_buffer import ReplayBuffer
from trainer       import Trainer
from arena         import Arena
from self_play     import evaluate_challenger


# ═════════════════════════════════════════════════════════════════════════════
# INFERENCE SERVER — forward GPU centralisé
# ═════════════════════════════════════════════════════════════════════════════

class InferenceServer:
    """
    Thread GPU unique qui reçoit des états de tous les workers
    et retourne les (value, policy) en batches.

    Protocole :
      worker → server : (request_id, state_tensor (20,9,9))
      server → worker : (value float, policy_logprobs (81,))

    Le serveur accumule max_batch_size requêtes ou attend jusqu'à
    max_wait_ms avant de lancer un forward — whichever comes first.
    """

    def __init__(
        self,
        model:          torch.nn.Module,
        device:         torch.device,
        max_batch_size: int   = 32,
        max_wait_ms:    float = 5.0,
    ):
        self.model          = model
        self.device         = device
        self.max_batch      = max_batch_size
        self.max_wait       = max_wait_ms / 1000.0  # en secondes

        self._in_q:  queue.Queue = queue.Queue()   # (req_id, tensor)
        self._out_qs: dict       = {}              # req_id → Queue(résultat)
        self._lock   = threading.Lock()
        self._stop   = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2.0)

    def register_worker(self, worker_id: int) -> queue.Queue:
        """Enregistre un worker et retourne sa queue de résultats."""
        q = queue.Queue()
        with self._lock:
            self._out_qs[worker_id] = q
        return q

    def submit(self, worker_id: int, req_id: int, tensor: np.ndarray):
        """Worker → envoie un état au serveur."""
        self._in_q.put((worker_id, req_id, tensor))

    def _run(self):
        """Boucle principale du serveur — tourne dans son propre thread."""
        self.model.eval()

        while not self._stop.is_set():
            # Accumule un batch
            batch_wids   = []
            batch_rids   = []
            batch_tensors = []

            deadline = time.time() + self.max_wait

            # Première requête — bloquant avec timeout
            try:
                wid, rid, t = self._in_q.get(timeout=self.max_wait)
                batch_wids.append(wid)
                batch_rids.append(rid)
                batch_tensors.append(t)
            except queue.Empty:
                continue

            # Requêtes suivantes — non bloquant jusqu'au batch max ou deadline
            while len(batch_tensors) < self.max_batch and time.time() < deadline:
                try:
                    wid, rid, t = self._in_q.get_nowait()
                    batch_wids.append(wid)
                    batch_rids.append(rid)
                    batch_tensors.append(t)
                except queue.Empty:
                    time.sleep(0.0005)

            # Forward GPU
            x = torch.from_numpy(
                np.stack(batch_tensors).astype(np.float32)
            ).to(self.device)

            with torch.no_grad():
                log_probs, values = self.model(x)

            lp_np = log_probs.cpu().numpy()
            v_np  = values.cpu().numpy()

            # Dispatch résultats aux workers
            for i, (wid, rid) in enumerate(zip(batch_wids, batch_rids)):
                with self._lock:
                    out_q = self._out_qs.get(wid)
                if out_q is not None:
                    out_q.put((rid, float(v_np[i]), lp_np[i]))

    def update_weights(self, state_dict: dict):
        """Met à jour les poids du modèle (appelé après chaque epoch d'entraînement)."""
        self.model.load_state_dict(state_dict)
        self.model.eval()


# ═════════════════════════════════════════════════════════════════════════════
# ÉVALUATEUR WORKER — délègue les forwards au InferenceServer
# ═════════════════════════════════════════════════════════════════════════════

class ServerEvaluator:
    """
    Remplace LightEvaluator côté worker.
    Au lieu de faire un forward local, envoie l'état au InferenceServer
    et attend le résultat.
    """

    def __init__(self, server: InferenceServer, worker_id: int):
        self.server    = server
        self.worker_id = worker_id
        self._out_q    = server.register_worker(worker_id)
        self._req_id   = 0
        self._cache: dict = {}
        self.device    = server.device   # requis par MCTSEngine

    def clear_cache(self):
        self._cache.clear()

    def _query(self, state: UTTTState) -> Tuple[float, np.ndarray]:
        key = state.to_string()
        if key in self._cache:
            return self._cache[key]

        tensor = state_to_tensor(key)
        self._req_id += 1
        rid = self._req_id

        self.server.submit(self.worker_id, rid, tensor)

        # Attend la réponse (avec timeout de sécurité)
        while True:
            try:
                resp_rid, value, lp = self._out_q.get(timeout=10.0)
                if resp_rid == rid:
                    result = (value, lp)
                    self._cache[key] = result
                    return result
                # Mauvais rid (réponse d'une requête précédente) → ignorer
            except queue.Empty:
                # Timeout — relancer la requête
                self.server.submit(self.worker_id, rid, tensor)

    def evaluate(self, state: UTTTState) -> float:
        return self._query(state)[0]

    def policy_logprobs(self, state: UTTTState) -> np.ndarray:
        return self._query(state)[1]

    def evaluate_and_policy(self, state: UTTTState) -> Tuple[float, np.ndarray]:
        return self._query(state)


# ═════════════════════════════════════════════════════════════════════════════
# MCTS WORKER — joue une partie via le serveur d'inférence
# ═════════════════════════════════════════════════════════════════════════════

def _play_game_via_server(
    evaluator:   ServerEvaluator,
    simulations: int,
    c_puct:      float,
    temp_cutoff: int,
) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    """
    Joue une partie complète de self-play en utilisant ServerEvaluator.
    Même logique que play_self_play_game() mais avec le serveur d'inférence.
    """
    from bot_mcts import MCTSEngine

    engine   = MCTSEngine(evaluator, simulations=simulations, c_puct=c_puct, batch_size=1)

    # Surcharge de _batch_forward : au lieu d'appeler ev.model directement,
    # on envoie chaque état au serveur via ServerEvaluator._query()
    def _server_batch_forward(states):
        import numpy as np
        results = [evaluator._query(s) for s in states]
        values  = np.array([r[0] for r in results], dtype=np.float32)
        lps     = np.stack([r[1] for r in results])
        return values, lps
    engine._batch_forward = _server_batch_forward
    state    = UTTTState.initial()
    examples = []
    move_num = 0

    while not state.is_terminal:
        move_num += 1
        temp = 1.0 if move_num <= temp_cutoff else 0.1
        move, pi = engine.search_with_policy(state, temperature=temp)
        tensor   = state_to_tensor(state.to_string())
        examples.append((tensor, pi, state.player))
        state = state.apply_move(move)

    winner = state.winner
    result = []
    for tensor, pi, player in examples:
        z = 0.0 if winner == 0 else (1.0 if winner == player else -1.0)
        result.append((tensor, pi, z))

        # Augmentation miroir
        t_m = tensor.copy()
        t_m[0:9]  = tensor[9:18].copy()
        t_m[9:18] = tensor[0:9].copy()
        t_m[19]   = 1.0 - tensor[19]
        w_m = 3 - winner if winner in (1, 2) else 0
        p_m = 3 - player
        z_m = 0.0 if w_m == 0 else (1.0 if w_m == p_m else -1.0)
        result.append((t_m, pi, z_m))

    return result


def _worker_loop(
    worker_id:    int,
    server:       InferenceServer,
    result_queue: queue.Queue,
    n_games:      int,
    simulations:  int,
    c_puct:       float,
    temp_cutoff:  int,
    seed:         int,
):
    """Fonction exécutée dans chaque thread worker."""
    random.seed(seed + worker_id)
    np.random.seed(seed + worker_id)

    evaluator = ServerEvaluator(server, worker_id)

    for _ in range(n_games):
        try:
            examples = _play_game_via_server(
                evaluator, simulations, c_puct, temp_cutoff
            )
            result_queue.put(("game", examples))
        except Exception as e:
            result_queue.put(("error", str(e)))

    result_queue.put(("done", worker_id))


# ═════════════════════════════════════════════════════════════════════════════
# SELF-PLAY PARALLÈLE
# ═════════════════════════════════════════════════════════════════════════════

def run_parallel_self_play(
    server:       InferenceServer,
    buffer:       ReplayBuffer,
    n_games:      int,
    n_workers:    int,
    simulations:  int,
    c_puct:       float,
    temp_cutoff:  int,
    seed:         int,
) -> Tuple[int, float]:
    """
    Lance n_workers threads qui jouent n_games parties en parallèle.
    Retourne (n_exemples_ajoutés, temps_écoulé).
    """
    games_per_worker = max(1, n_games // n_workers)
    result_queue     = queue.Queue()
    t0               = time.time()
    n_examples       = 0
    n_done           = 0

    # Lancer les workers
    threads = []
    for wid in range(n_workers):
        # Dernier worker prend le reste
        ng = games_per_worker if wid < n_workers - 1 else n_games - games_per_worker * (n_workers - 1)
        t  = threading.Thread(
            target=_worker_loop,
            args=(wid, server, result_queue, ng, simulations, c_puct, temp_cutoff, seed),
            daemon=True,
        )
        t.start()
        threads.append(t)

    # Collecter les résultats
    while n_done < n_workers:
        try:
            msg = result_queue.get(timeout=60.0)
        except queue.Empty:
            print("  ⚠  Worker timeout — vérifier les workers")
            break

        if msg[0] == "game":
            examples  = msg[1]
            buffer.push_game(examples)
            n_examples += len(examples)
        elif msg[0] == "error":
            print(f"  ⚠  Worker error : {msg[1]}")
        elif msg[0] == "done":
            n_done += 1

    for t in threads:
        t.join(timeout=5.0)

    return n_examples, time.time() - t0


# ═════════════════════════════════════════════════════════════════════════════
# BOUCLE PRINCIPALE
# ═════════════════════════════════════════════════════════════════════════════

def run_alphazero_parallel(
    # Self-play
    simulations:     int   = 300,
    games_per_iter:  int   = 50,
    temp_cutoff:     int   = 10,
    n_workers:       int   = 4,
    # Inference server
    server_batch:    int   = 32,
    server_wait_ms:  float = 5.0,
    # Buffer
    buffer_size:     int   = 100_000,
    min_buffer:      int   = 500,
    # Entraînement
    train_steps:     int   = 100,
    batch_size:      int   = 256,
    lr:              float = 1e-3,
    weight_decay:    float = 1e-4,
    # Évaluation
    eval_games:      int   = 16,
    eval_sims:       int   = 100,
    win_threshold:   float = 0.55,
    eval_every:      int   = 3,
    # Réseau
    num_filters:     int   = 128,
    num_res_blocks:  int   = 4,
    # Persistance
    checkpoint_dir:  str   = "models/alphazero",
    resume:          Optional[str] = None,
    # Durée
    iterations:      int   = 200,
    device:          Optional[str] = None,
    seed:            int   = 42,
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ── Initialisation ─────────────────────────────────────────────────────

    current = LightEvaluator(
        checkpoint     = resume,
        device         = device,
        num_filters    = num_filters,
        num_res_blocks = num_res_blocks,
    )
    champion = LightEvaluator(device=device, num_filters=num_filters, num_res_blocks=num_res_blocks)
    champion.clone_weights_from(current)

    buffer  = ReplayBuffer(max_size=buffer_size)
    trainer = Trainer(current, buffer, lr=lr, weight_decay=weight_decay,
                      total_steps=iterations * train_steps)

    # Inference server — utilise le modèle de current directement
    server = InferenceServer(
        model          = current.model,
        device         = current.device,
        max_batch_size = server_batch,
        max_wait_ms    = server_wait_ms,
    )
    server.start()

    print(f"\n{'═'*62}")
    print(f"  AlphaZero UTTT — Self-play parallèle")
    print(f"  Device         : {current.device}")
    print(f"  Workers        : {n_workers}")
    print(f"  Server batch   : {server_batch}  (wait={server_wait_ms}ms)")
    print(f"  Simulations    : {simulations}")
    print(f"  Parties/iter   : {games_per_iter}")
    print(f"  Checkpoint dir : {checkpoint_dir}")
    print(f"{'═'*62}\n")

    best_wr = 0.0
    c_puct  = 1.5

    try:
        for iteration in range(1, iterations + 1):
            t_iter = time.time()
            print(f"{'─'*62}")
            print(f"  ITÉRATION {iteration}/{iterations}")
            print(f"{'─'*62}")

            # ── 1. SELF-PLAY PARALLÈLE ────────────────────────────────────
            print(f"\n  [1/3] Self-play parallèle ({games_per_iter} parties, "
                  f"{n_workers} workers, {simulations} sims/coup)")

            # Sync poids → server avant le self-play
            server.update_weights(current.model.state_dict())

            n_ex, t_sp = run_parallel_self_play(
                server       = server,
                buffer       = buffer,
                n_games      = games_per_iter,
                n_workers    = n_workers,
                simulations  = simulations,
                c_puct       = c_puct,
                temp_cutoff  = temp_cutoff,
                seed         = seed + iteration,
            )
            tps = n_ex / t_sp if t_sp > 0 else 0
            print(f"  → {n_ex} exemples en {t_sp:.1f}s  "
                  f"({tps:.0f} ex/s)  buffer={len(buffer)}")

            # ── 2. ENTRAÎNEMENT ───────────────────────────────────────────
            if len(buffer) < min_buffer:
                print(f"\n  [2/3] Entraînement ignoré (buffer={len(buffer)} < {min_buffer})")
            else:
                print(f"\n  [2/3] Entraînement ({train_steps} steps, batch={batch_size})")
                t0      = time.time()
                summary = trainer.train_epoch(
                    n_steps=train_steps, batch_size=batch_size, log_every=train_steps // 5
                )
                print(f"  → loss={summary['loss']:.4f}  "
                      f"v={summary['loss_v']:.4f}  "
                      f"p={summary['loss_p']:.4f}  "
                      f"({time.time()-t0:.1f}s)")

            # ── 3. ÉVALUATION ─────────────────────────────────────────────
            if iteration % eval_every == 0 and len(buffer) >= min_buffer:
                print(f"\n  [3/3] Évaluation challenger vs champion ({eval_games} parties)")
                promoted, wr = evaluate_challenger(
                    champion, current,
                    simulations=eval_sims, n_games=eval_games, win_threshold=win_threshold,
                )
                if promoted:
                    print(f"  ✓ Challenger promu ! WR={wr*100:.1f}%")
                    champion.clone_weights_from(current)
                    current.save(os.path.join(checkpoint_dir, f"best_iter{iteration:04d}.pth"))
                    current.save(os.path.join(checkpoint_dir, "best.pth"))
                    best_wr = wr
                else:
                    print(f"  ~ Champion conservé. WR={wr*100:.1f}%")
                    current.clone_weights_from(champion)
            else:
                print(f"\n  [3/3] Évaluation dans "
                      f"{eval_every - (iteration % eval_every)} itérations")

            # ── Checkpoint périodique ──────────────────────────────────────
            current.save(os.path.join(checkpoint_dir, f"iter_{iteration:04d}.pth"))

            elapsed = time.time() - t_iter
            games_per_hour = games_per_iter / elapsed * 3600
            print(f"\n  Itération {iteration} : {elapsed:.1f}s  "
                  f"({games_per_hour:.0f} parties/h)  "
                  f"best WR={best_wr*100:.1f}%\n")

    finally:
        server.stop()
        print("  InferenceServer arrêté.")

    print("═" * 62)
    print(f"  Entraînement terminé. Meilleur WR : {best_wr*100:.1f}%")
    print(f"  Checkpoints → {checkpoint_dir}/")
    print("═" * 62)


# ═════════════════════════════════════════════════════════════════════════════
# POINT D'ENTRÉE
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AlphaZero self-play parallèle pour UTTT"
    )
    parser.add_argument("--workers",       type=int,   default=4,
                        help="Nombre de workers parallèles (défaut : 4)")
    parser.add_argument("--server-batch",  type=int,   default=32,
                        help="Taille max du batch GPU (défaut : 32)")
    parser.add_argument("--server-wait",   type=float, default=5.0,
                        help="Attente max avant forward en ms (défaut : 5)")
    parser.add_argument("--simulations",   type=int,   default=300)
    parser.add_argument("--games-per-iter",type=int,   default=50)
    parser.add_argument("--temp-cutoff",   type=int,   default=10)
    parser.add_argument("--buffer-size",   type=int,   default=100_000)
    parser.add_argument("--min-buffer",    type=int,   default=500)
    parser.add_argument("--train-steps",   type=int,   default=100)
    parser.add_argument("--batch-size",    type=int,   default=256)
    parser.add_argument("--lr",            type=float, default=1e-3)
    parser.add_argument("--weight-decay",  type=float, default=1e-4)
    parser.add_argument("--eval-games",    type=int,   default=16)
    parser.add_argument("--eval-sims",     type=int,   default=100)
    parser.add_argument("--win-threshold", type=float, default=0.55)
    parser.add_argument("--eval-every",    type=int,   default=3)
    parser.add_argument("--num-filters",   type=int,   default=128)
    parser.add_argument("--num-res-blocks",type=int,   default=4)
    parser.add_argument("--checkpoint-dir",default="models/alphazero")
    parser.add_argument("--resume",        default=None)
    parser.add_argument("--iterations",    type=int,   default=200)
    parser.add_argument("--device",        default=None)
    parser.add_argument("--seed",          type=int,   default=42)
    args = parser.parse_args()

    run_alphazero_parallel(
        simulations    = args.simulations,
        games_per_iter = args.games_per_iter,
        temp_cutoff    = args.temp_cutoff,
        n_workers      = args.workers,
        server_batch   = args.server_batch,
        server_wait_ms = args.server_wait,
        buffer_size    = args.buffer_size,
        min_buffer     = args.min_buffer,
        train_steps    = args.train_steps,
        batch_size     = args.batch_size,
        lr             = args.lr,
        weight_decay   = args.weight_decay,
        eval_games     = args.eval_games,
        eval_sims      = args.eval_sims,
        win_threshold  = args.win_threshold,
        eval_every     = args.eval_every,
        num_filters    = args.num_filters,
        num_res_blocks = args.num_res_blocks,
        checkpoint_dir = args.checkpoint_dir,
        resume         = args.resume,
        iterations     = args.iterations,
        device         = args.device,
        seed           = args.seed,
    )
