"""
rl_baseline.py — Phase 6 scaffold: tabular Q-learning vs. the Webster
allocator (Phase 1/2) on a *simulated* 4-way intersection.

SCOPE NOTE — read this before citing numbers from this module anywhere:
This is a lightweight discrete-event simulation, NOT a SUMO integration.
It exists to (a) give a real, reproducible answer to "does a learned
controller beat Webster on the metric Phase 1 already defined", and
(b) fix the MDP shape (state/action/reward) so a later SUMO-backed
environment is a drop-in replacement for SimulatedIntersection rather than
a redesign. Calibrating this against a real SUMO network with your actual
intersection geometry and measured traffic pattern is separate follow-up
work — arrival rates here are illustrative, not measured.

MDP:
  State  — each approach's queue length, bucketed into 4 levels (0 / 1-5 /
           6-15 / 16+) -> a 4^4 = 256-state tabular Q-table.
  Action — one of 5 discrete cycle-allocation *presets* over the same
           TOTAL_CYCLE_S used by intersection.allocate_cycle: an even split,
           or a "boost" preset that gives one approach the bulk of the
           cycle. A continuous action space would need function
           approximation (DQN) instead of a table — out of scope here.
  Reward — negative vehicle-seconds of unserved queue this cycle, using the
           same served/unserved bookkeeping as baseline.estimate_delay, so
           Q-learning directly optimises the metric Phase 1 established
           rather than an unrelated proxy.

Fairness note: all three controllers below (fixed-timer, Webster, trained
agent) are evaluated against the IDENTICAL pre-generated arrival sequence
(see `generate_arrival_sequence`) — without that, differences in outcome
could just be different random draws, not a real difference in policy
quality.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable

from .intersection import APPROACHES, TOTAL_CYCLE_S, allocate_cycle

QUEUE_BUCKETS = (0, 5, 15)  # boundaries: 0 | 1-5 | 6-15 | 16+
ACTIONS = ["EQUAL"] + [f"BOOST_{a}" for a in APPROACHES]


def bucket(queue: float) -> int:
    for i, b in enumerate(QUEUE_BUCKETS):
        if queue <= b:
            return i
    return len(QUEUE_BUCKETS)


def action_to_allocation(action: str) -> dict[str, int]:
    """Translate a discrete action into a concrete cycle allocation, using
    the same allocate_cycle() Webster machinery so every controller shares
    identical bounds/apportionment mechanics — the only thing that differs
    between controllers is which demand vector (real queues, vs. a boosted
    synthetic one) drives the split."""
    if action == "EQUAL":
        demand = {a: 1.0 for a in APPROACHES}
    else:
        boosted = action.split("_", 1)[1]
        demand = {a: (5.0 if a == boosted else 1.0) for a in APPROACHES}
    return allocate_cycle(demand)


def generate_arrival_sequence(
    arrival_rates: dict[str, float], n_cycles: int, seed: int = 0
) -> list[dict[str, float]]:
    """Pre-generate one arrival sequence so every controller under
    comparison faces literally the same traffic, not independent random
    draws that would make the comparison meaningless."""
    rng = random.Random(seed)
    sequence = []
    for _ in range(n_cycles):
        sequence.append({
            a: rng.uniform(0, 2 * arrival_rates[a]) * TOTAL_CYCLE_S / 60.0
            for a in APPROACHES
        })
    return sequence


@dataclass
class SimulatedIntersection:
    """Discrete-event simulation: one step = one full signal cycle.

    Each approach has a queue. An allocation is decided from the queue
    state as observed BEFORE this cycle's arrivals (mirroring the real
    pipeline, which only ever sees demand up to "now"); arrivals are then
    added; then the allocation serves each approach's queue at a fixed
    saturation flow rate. Unserved vehicles carry over to the next cycle.
    """

    arrival_sequence: list[dict[str, float]]
    saturation_flow_per_s: float = 0.5
    queues: dict[str, float] = field(init=False)
    _cycle_idx: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.queues = {a: 0.0 for a in APPROACHES}

    def reset(self) -> dict[str, int]:
        self.queues = {a: 0.0 for a in APPROACHES}
        self._cycle_idx = 0
        return self.state()

    def state(self) -> dict[str, int]:
        return {a: bucket(self.queues[a]) for a in APPROACHES}

    def step_with_allocation_fn(
        self, allocation_fn: Callable[[dict[str, float]], dict[str, int]]
    ) -> tuple[dict[str, int], float, dict]:
        allocation = allocation_fn(dict(self.queues))

        arrivals = self.arrival_sequence[self._cycle_idx % len(self.arrival_sequence)]
        self._cycle_idx += 1
        for a in APPROACHES:
            self.queues[a] += arrivals[a]

        total_unserved = 0.0
        for a in APPROACHES:
            served = min(self.queues[a], self.saturation_flow_per_s * allocation[a])
            self.queues[a] = max(0.0, self.queues[a] - served)
            total_unserved += self.queues[a]

        reward = -total_unserved * TOTAL_CYCLE_S  # vehicle-seconds; same unit as baseline.estimate_delay
        return self.state(), reward, {"allocation": allocation, "queues": dict(self.queues)}

    def step(self, action: str) -> tuple[dict[str, int], float, dict]:
        return self.step_with_allocation_fn(lambda _q: action_to_allocation(action))


class QLearningAgent:
    """Minimal tabular Q-learning — deliberately simple (no replay buffer,
    no function approximation), since the state space here (4^4 = 256
    states x 5 actions) is small enough for a plain table."""

    def __init__(self, actions: list[str], alpha: float = 0.2, gamma: float = 0.9,
                 epsilon: float = 0.2, rng: random.Random | None = None):
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.rng = rng or random.Random()
        self.q: dict[tuple, dict[str, float]] = {}

    def _row(self, state_key: tuple) -> dict[str, float]:
        return self.q.setdefault(state_key, {a: 0.0 for a in self.actions})

    def choose_action(self, state: dict[str, int], greedy: bool = False) -> str:
        key = tuple(state[a] for a in APPROACHES)
        row = self._row(key)
        if not greedy and self.rng.random() < self.epsilon:
            return self.rng.choice(self.actions)
        return max(row, key=row.get)

    def update(self, state: dict, action: str, reward: float, next_state: dict) -> None:
        key = tuple(state[a] for a in APPROACHES)
        next_key = tuple(next_state[a] for a in APPROACHES)
        row = self._row(key)
        next_row = self._row(next_key)
        best_next = max(next_row.values())
        row[action] += self.alpha * (reward + self.gamma * best_next - row[action])


def train(
    arrival_rates: dict[str, float],
    episodes: int = 200,
    steps_per_episode: int = 50,
    seed: int = 0,
) -> QLearningAgent:
    """Train on freshly-sampled traffic each episode (so the agent
    generalises across arrival patterns rather than memorising one
    sequence) — evaluation, separately, always uses one fixed held-out
    sequence for fairness (see generate_arrival_sequence)."""
    agent = QLearningAgent(ACTIONS, rng=random.Random(seed))
    for ep in range(episodes):
        sequence = generate_arrival_sequence(arrival_rates, steps_per_episode, seed=seed * 10_000 + ep)
        env = SimulatedIntersection(arrival_sequence=sequence)
        state = env.reset()
        for _ in range(steps_per_episode):
            action = agent.choose_action(state)
            next_state, reward, _ = env.step(action)
            agent.update(state, action, reward, next_state)
            state = next_state
    return agent


def evaluate_discrete_policy(env: SimulatedIntersection, policy: Callable[[dict], str], steps: int) -> float:
    """Total vehicle-seconds of unserved-queue delay for a discrete-action
    policy (fixed-timer, or a trained agent in greedy mode)."""
    state = env.reset()
    total_delay = 0.0
    for _ in range(steps):
        action = policy(state)
        state, reward, _ = env.step(action)
        total_delay += -reward
    return total_delay


def evaluate_webster(env: SimulatedIntersection, steps: int) -> float:
    """Total vehicle-seconds of unserved-queue delay for the Webster
    allocator, computed fresh from the real (unbucketed) queue snapshot
    each cycle — Webster sees the exact queue, not the RL agent's coarser
    bucketed state, since that's how it already runs in intersection.py."""
    env.reset()
    total_delay = 0.0
    for _ in range(steps):
        _, reward, _ = env.step_with_allocation_fn(lambda q: allocate_cycle(q))
        total_delay += -reward
    return total_delay


def compare_all(
    arrival_rates: dict[str, float],
    episodes: int = 200,
    train_steps_per_episode: int = 50,
    eval_steps: int = 100,
    seed: int = 0,
) -> dict:
    """Train a Q-learning agent, then evaluate fixed-timer, Webster, and the
    trained agent on the SAME held-out arrival sequence. Returns total delay
    for each plus % improvement of the agent over Webster and over fixed."""
    agent = train(arrival_rates, episodes=episodes, steps_per_episode=train_steps_per_episode, seed=seed)

    eval_sequence = generate_arrival_sequence(arrival_rates, eval_steps, seed=seed + 999_999)

    fixed_delay = evaluate_discrete_policy(
        SimulatedIntersection(arrival_sequence=eval_sequence), lambda s: "EQUAL", eval_steps
    )
    webster_delay = evaluate_webster(SimulatedIntersection(arrival_sequence=eval_sequence), eval_steps)
    agent_delay = evaluate_discrete_policy(
        SimulatedIntersection(arrival_sequence=eval_sequence),
        lambda s: agent.choose_action(s, greedy=True),
        eval_steps,
    )

    def pct_improve(worse: float, better: float) -> float:
        return 100.0 * (worse - better) / worse if worse > 0 else 0.0

    return {
        "fixed_delay": fixed_delay,
        "webster_delay": webster_delay,
        "agent_delay": agent_delay,
        "agent_vs_fixed_pct": pct_improve(fixed_delay, agent_delay),
        "agent_vs_webster_pct": pct_improve(webster_delay, agent_delay),
        "webster_vs_fixed_pct": pct_improve(fixed_delay, webster_delay),
    }
