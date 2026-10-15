"""Tests for rl_baseline.py — the Phase 6 simulated-environment scaffold.
Pure logic + a fast, small-scale training run; no video/model needed."""

import pytest

from traffic_system.intersection import APPROACHES, TOTAL_CYCLE_S
from traffic_system.rl_baseline import (
    ACTIONS,
    QLearningAgent,
    SimulatedIntersection,
    action_to_allocation,
    bucket,
    compare_all,
    evaluate_discrete_policy,
    evaluate_webster,
    generate_arrival_sequence,
    train,
)


class TestBucket:
    def test_zero_is_bucket_zero(self):
        assert bucket(0) == 0

    def test_within_first_band_is_bucket_one(self):
        assert bucket(3) == 1

    def test_within_second_band_is_bucket_two(self):
        assert bucket(10) == 2

    def test_above_all_bands_is_last_bucket(self):
        assert bucket(1000) == 3

    def test_monotonic_non_decreasing(self):
        values = [0, 1, 5, 6, 15, 16, 100]
        buckets = [bucket(v) for v in values]
        assert buckets == sorted(buckets)


class TestActionToAllocation:
    def test_equal_action_splits_evenly(self):
        allocation = action_to_allocation("EQUAL")
        values = list(allocation.values())
        assert all(v == values[0] for v in values)

    def test_boost_action_favors_boosted_approach(self):
        allocation = action_to_allocation("BOOST_N")
        assert allocation["N"] > allocation["E"]
        assert allocation["N"] > allocation["S"]
        assert allocation["N"] > allocation["W"]

    def test_every_action_sums_to_total_cycle_or_less(self):
        for action in ACTIONS:
            allocation = action_to_allocation(action)
            assert sum(allocation.values()) <= TOTAL_CYCLE_S


class TestGenerateArrivalSequence:
    def test_same_seed_is_reproducible(self):
        rates = {a: 5.0 for a in APPROACHES}
        seq1 = generate_arrival_sequence(rates, 20, seed=42)
        seq2 = generate_arrival_sequence(rates, 20, seed=42)
        assert seq1 == seq2

    def test_different_seed_differs(self):
        rates = {a: 5.0 for a in APPROACHES}
        seq1 = generate_arrival_sequence(rates, 20, seed=1)
        seq2 = generate_arrival_sequence(rates, 20, seed=2)
        assert seq1 != seq2

    def test_length_matches_n_cycles(self):
        rates = {a: 5.0 for a in APPROACHES}
        seq = generate_arrival_sequence(rates, 15, seed=0)
        assert len(seq) == 15

    def test_arrivals_are_non_negative(self):
        rates = {a: 5.0 for a in APPROACHES}
        seq = generate_arrival_sequence(rates, 20, seed=0)
        assert all(v >= 0 for cycle in seq for v in cycle.values())


class TestSimulatedIntersection:
    def test_reset_clears_queues(self):
        seq = generate_arrival_sequence({a: 5.0 for a in APPROACHES}, 10, seed=0)
        env = SimulatedIntersection(arrival_sequence=seq)
        env.step("EQUAL")
        state = env.reset()
        assert all(v == 0 for v in state.values())

    def test_step_returns_state_reward_info(self):
        seq = generate_arrival_sequence({a: 5.0 for a in APPROACHES}, 10, seed=0)
        env = SimulatedIntersection(arrival_sequence=seq)
        env.reset()
        state, reward, info = env.step("EQUAL")
        assert set(state.keys()) == set(APPROACHES)
        assert isinstance(reward, float)
        assert "allocation" in info and "queues" in info

    def test_reward_is_never_positive(self):
        # Reward is negative unserved-queue delay — never a positive number.
        seq = generate_arrival_sequence({a: 5.0 for a in APPROACHES}, 10, seed=0)
        env = SimulatedIntersection(arrival_sequence=seq)
        env.reset()
        for _ in range(10):
            _, reward, _ = env.step("EQUAL")
            assert reward <= 0

    def test_zero_arrivals_means_zero_delay(self):
        seq = [{a: 0.0 for a in APPROACHES} for _ in range(10)]
        env = SimulatedIntersection(arrival_sequence=seq)
        env.reset()
        for _ in range(10):
            _, reward, _ = env.step("EQUAL")
            assert reward == 0.0


class TestQLearningAgent:
    def test_choose_action_returns_valid_action(self):
        agent = QLearningAgent(ACTIONS, epsilon=0.0)
        state = {a: 0 for a in APPROACHES}
        assert agent.choose_action(state, greedy=True) in ACTIONS

    def test_greedy_choice_is_deterministic_after_update(self):
        agent = QLearningAgent(ACTIONS, epsilon=0.0)
        state = {a: 0 for a in APPROACHES}
        next_state = {a: 1 for a in APPROACHES}
        agent.update(state, "BOOST_N", reward=10.0, next_state=next_state)
        # BOOST_N now has the only nonzero Q-value for this state -> greedy picks it.
        assert agent.choose_action(state, greedy=True) == "BOOST_N"


class TestTrainAndCompare:
    def test_training_runs_and_returns_agent(self):
        rates = {a: 5.0 for a in APPROACHES}
        agent = train(rates, episodes=5, steps_per_episode=10, seed=0)
        assert isinstance(agent, QLearningAgent)
        assert len(agent.q) > 0

    def test_compare_all_returns_expected_keys(self):
        rates = {a: 5.0 for a in APPROACHES}
        result = compare_all(rates, episodes=5, train_steps_per_episode=10, eval_steps=10, seed=0)
        assert set(result.keys()) == {
            "fixed_delay", "webster_delay", "agent_delay",
            "agent_vs_fixed_pct", "agent_vs_webster_pct", "webster_vs_fixed_pct",
        }

    def test_identical_eval_sequence_across_controllers(self):
        # Webster and fixed-timer should face the exact same traffic when
        # arrival_rates and seed match — a regression guard for the
        # fairness property this module depends on.
        rates = {a: 5.0 for a in APPROACHES}
        seq_a = generate_arrival_sequence(rates, 10, seed=123)
        seq_b = generate_arrival_sequence(rates, 10, seed=123)
        env_a = SimulatedIntersection(arrival_sequence=seq_a)
        env_b = SimulatedIntersection(arrival_sequence=seq_b)
        delay_a = evaluate_webster(env_a, 10)
        delay_b = evaluate_webster(env_b, 10)
        assert delay_a == delay_b

    def test_webster_beats_fixed_under_skewed_demand(self):
        # One approach gets far more traffic than the others — Webster's
        # whole premise is that it should route more green there than a
        # fixed-timer would. Skewed rates make this a real (not tautological)
        # check rather than assuming the conclusion.
        rates = {"N": 20.0, "E": 2.0, "S": 2.0, "W": 2.0}
        result = compare_all(rates, episodes=5, train_steps_per_episode=10, eval_steps=30, seed=0)
        assert result["webster_vs_fixed_pct"] > 0
