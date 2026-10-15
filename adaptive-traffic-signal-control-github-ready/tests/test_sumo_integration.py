"""Smoke test for the real SUMO integration (Phase 6). Skipped automatically
if `traci` or the SUMO binary aren't available — this is infrastructure-
dependent in a way the rest of the suite deliberately isn't, so it's kept
separate rather than blocking the pure-logic tests everywhere else."""

import shutil
from pathlib import Path

import pytest

traci = pytest.importorskip("traci", reason="traci not installed — see sumo_env/build_network.sh")

pytestmark = pytest.mark.skipif(
    shutil.which("sumo") is None, reason="sumo binary not found — install the `sumo` package"
)

from traffic_system.sumo_integration import (  # noqa: E402
    NET_FILE,
    SumoTraciEnv,
    APPROACHES,
    build_route_file,
    run_fixed_timer,
    run_webster,
)


@pytest.fixture(scope="module")
def route_file(tmp_path_factory):
    if not NET_FILE.exists():
        pytest.skip(f"{NET_FILE} not built — run sumo_env/build_network.sh first")
    path = tmp_path_factory.mktemp("sumo") / "test.rou.xml"
    build_route_file({"N": 10.0, "E": 4.0, "S": 8.0, "W": 3.0}, path, sim_duration_s=300, seed=0)
    return path


class TestSumoTraciEnv:
    def test_start_and_close_do_not_raise(self, route_file):
        env = SumoTraciEnv(route_file, seed=0)
        env.start()
        env.close()

    def test_queue_lengths_have_all_approaches(self, route_file):
        env = SumoTraciEnv(route_file, seed=0)
        env.start()
        try:
            queues = env.get_queue_lengths()
            assert set(queues.keys()) == set(APPROACHES)
            assert all(v >= 0 for v in queues.values())
        finally:
            env.close()

    def test_run_cycle_advances_simulation(self, route_file):
        env = SumoTraciEnv(route_file, seed=0)
        env.start()
        try:
            env.run_cycle({a: 20 for a in APPROACHES})
            assert env.total_waiting_time() >= 0.0
        finally:
            env.close()


class TestControllerComparison:
    def test_fixed_timer_returns_non_negative_delay(self, route_file):
        delay = run_fixed_timer(route_file, n_cycles=3, fixed_green_s=20, seed=0)
        assert delay >= 0.0

    def test_webster_returns_non_negative_delay(self, route_file):
        delay = run_webster(route_file, n_cycles=3, seed=0)
        assert delay >= 0.0
