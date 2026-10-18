"""Real tabular reinforcement learning; no language-model weight updates."""
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor


class GridWorld:
    """Gymnasium-style reset/step API with seeded starts and a finite horizon."""
    actions = ((0, -1), (1, 0), (0, 1), (-1, 0))

    def __init__(self, size=5, horizon=60):
        self.size, self.horizon = size, horizon

    def reset(self, seed=None):
        rng = random.Random(seed)
        self.position = rng.choice([(x,y) for x in range(self.size) for y in range(self.size) if (x,y) != (self.size-1,self.size-1)])
        self.steps = 0
        return self.position, {"goal": (self.size-1, self.size-1)}

    def step(self, action):
        if action not in range(4):
            raise ValueError("Action must be 0..3")
        dx, dy = self.actions[action]
        x, y = self.position
        self.position = (max(0,min(self.size-1,x+dx)),max(0,min(self.size-1,y+dy)))
        self.steps += 1
        terminated = self.position == (self.size-1,self.size-1)
        return self.position, 1.0 if terminated else -0.02, terminated, self.steps >= self.horizon and not terminated, {}


def evaluate(q, size, seeds, stop=None, deadline=None):
    rng = random.Random(741)
    wins, rewards = 0, []
    for seed in seeds:
        env = GridWorld(size)
        state, _ = env.reset(seed)
        total = 0.0
        for _ in range(env.horizon):
            if (stop and stop.is_set()) or (deadline is not None and time.monotonic() >= deadline):
                raise InterruptedError("Experiment stopped or time limit reached")
            values = q.get(state)
            action = rng.randrange(4) if values is None else max(range(4), key=lambda a: values[a])
            state, reward, done, truncated, _ = env.step(action)
            total += reward
            if done or truncated:
                wins += int(done)
                break
        rewards.append(total)
    return {"episodes":len(rewards), "success_rate": wins/len(rewards), "mean_reward":sum(rewards)/len(rewards)}


def train_grid(episodes, size, seed, stop, seconds, progress=None):
    deadline = time.monotonic()+seconds
    rng = random.Random(seed)
    q = {}
    heldout = list(range(100000,100100))
    baseline = evaluate({}, size, heldout, stop, deadline)
    curve = []
    completed = 0
    for episode in range(episodes):
        if stop.is_set() or time.monotonic() >= deadline:
            break
        # Curriculum: start on a smaller board before the requested board size.
        board_size = min(size, 3 + episode // max(1,episodes//max(1,size-2)))
        env = GridWorld(board_size)
        state, _ = env.reset(rng.randrange(100000))
        total = 0
        epsilon = max(0.08, 1-episode/max(1,episodes)*0.95)
        for _ in range(env.horizon):
            if stop.is_set() or time.monotonic() >= deadline:
                break
            # Separate state tables for each curriculum board.
            key = (board_size, *state)
            values = q.setdefault(key, [0.0]*4)
            action = rng.randrange(4) if rng.random()<epsilon else max(range(4),key=lambda a:values[a])
            new_state, reward, done, truncated, _ = env.step(action)
            next_values = q.setdefault((board_size,*new_state), [0.0]*4)
            target = reward if done else reward+0.95*max(next_values)
            values[action] += 0.25*(target-values[action])
            state = new_state
            total += reward
            if done or truncated:
                break
        else:
            continue
        if stop.is_set() or time.monotonic() >= deadline:
            break
        completed += 1
        curve.append({"episode": completed, "reward": round(total,4), "size":board_size})
        if progress and completed % 25 == 0:
            progress(completed)
    final_q = {tuple(k[1:]):v for k,v in q.items() if k[0] == size}
    interrupted = stop.is_set() or time.monotonic() >= deadline
    candidate = None if interrupted else evaluate(final_q,size,heldout,stop,deadline)
    return {"mechanism":"tabular_q_learning", "weight_updates":False, "completed_episodes":completed,
            "baseline":baseline, "candidate":candidate, "curve":curve,
            "policy":{f"{x},{y}":v for (x,y),v in final_q.items()}, "interrupted":interrupted,
            "evaluation_note":"100 fixed held-out start seeds in the same grid dynamics; not evidence of transfer to Minecraft."}


class Experiments:
    def __init__(self, store, stop):
        self.store, self.stop = store, stop
        self.pool = ThreadPoolExecutor(max_workers=1,thread_name_prefix="grid")
        self.lock = threading.Lock()
        self.future = None
        # A previous process cannot leave a job falsely marked as running.
        store.execute("UPDATE experiments SET status='interrupted' WHERE status IN ('queued','running')")

    def start(self, config):
        with self.lock:
            if self.stop.is_set():
                raise ValueError("Emergency stop is active")
            if self.future and not self.future.done():
                raise ValueError("An experiment is already running")
            job = self.store.execute("INSERT INTO experiments(status,config) VALUES ('queued',?)",(json.dumps(config),))
            self.future = self.pool.submit(self.run,job,config)
            return job

    def run(self, job, config):
        self.store.execute("UPDATE experiments SET status='running' WHERE id=?",(job,))
        self.store.event("experiment.start",{"id":job,"config":config})
        try:
            result = train_grid(**config,stop=self.stop)
            status = "stopped" if result["interrupted"] else "completed"
        except InterruptedError as exc:
            status, result = "stopped", {"reason":str(exc)}
        except Exception as exc:
            status, result = "failed", {"error":str(exc)}
        self.store.execute("UPDATE experiments SET status=?,result=? WHERE id=?",(status,json.dumps(result),job))
        self.store.event("experiment.finish",{"id":job,"status":status})

    def close(self):
        self.stop.set()
        self.pool.shutdown(wait=True,cancel_futures=True)
