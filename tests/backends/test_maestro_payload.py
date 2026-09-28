# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""The Qoro Service form of a :class:`MaestroConfig` carries every setting, losslessly."""

import json
import math

import maestro
import pytest

from divi.backends import MaestroConfig, MaestroSimulator
from divi.backends.runners._maestro_payload import (
    maestro_config_from_payload,
    maestro_config_to_payload,
)

_BELL_QASM = (
    'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\ncreg c[2];\n'
    "h q[0];\ncx q[0], q[1];\n"
    "measure q[0] -> c[0];\nmeasure q[1] -> c[1];\n"
)


def _mixed_noise_model(*, kraus: bool = True):
    """A noise model whose calls carry tuples and, optionally, complex Kraus
    entries — which only density-matrix or MPO backends can run."""
    gamma = 0.19
    noise_model = maestro.NoiseModel()
    noise_model.set_all_depolarizing(2, 0.02)
    noise_model.set_readout_error(0, 0.02, 0.05)
    noise_model.set_multi_correlated_ou(1, [(15.0, 0.5), (3.0, 2.0)], 1e-7)
    if not kraus:
        return noise_model
    noise_model.set_kraus_channel(
        [0],
        [
            [[1, 0], [0, math.sqrt(1 - gamma)]],
            [[0, 1j * math.sqrt(gamma)], [0, 0]],
        ],
    )
    return noise_model


def _over_the_wire(config: MaestroConfig) -> MaestroConfig:
    payload = json.loads(json.dumps(maestro_config_to_payload(config)))
    return maestro_config_from_payload(payload)


def _bell_counts(noise_model) -> dict:
    sim = MaestroSimulator(
        shots=400, maestro_config=MaestroConfig(seed=11, noise_model=noise_model)
    )
    return sim.submit_circuits({"c0": _BELL_QASM}).results[0]["results"]


def test_maestro_names_and_enum_codes():
    payload = maestro_config_to_payload(
        MaestroConfig(
            simulator_type="Gpu",
            simulation_type="PathIntegral",
            max_bond_dimension=32,
            singular_value_threshold=1e-8,
        )
    )

    assert payload == {
        "simulator_type": maestro.SimulatorType.Gpu.value,
        "simulation_type": maestro.SimulationType.PathIntegral.value,
        "max_bond_dimension": 32,
        "singular_value_threshold": 1e-8,
    }


def test_only_set_options_are_sent():
    assert maestro_config_to_payload(MaestroConfig()) == {}


def test_round_trip():
    config = MaestroConfig(
        simulator_type="QCSim",
        simulation_type="MatrixProductState",
        max_bond_dimension=64,
        seed=3,
        distributed_options={"distributed_devices": "0,1"},
        mps_svd_solver="gesvdj",
    )
    assert _over_the_wire(config) == config


def test_a_config_from_a_simulator_config_round_trips():
    config = MaestroConfig.from_simulator_config(
        maestro.SimulatorConfig(
            simulation_type=maestro.SimulationType.MatrixProductState,
            max_bond_dimension=8,
        ),
        noise_seed=4,
    )
    assert _over_the_wire(config) == config


@pytest.mark.parametrize(
    "simulation_type, realizations", [("DensityMatrix", 1), ("Statevector", None)]
)
def test_noise_realizations_are_resolved_as_a_local_run_would(
    simulation_type, realizations
):
    noise_model = maestro.NoiseModel()
    noise_model.set_all_depolarizing(2, 0.02)
    payload = maestro_config_to_payload(
        MaestroConfig(simulation_type=simulation_type, noise_model=noise_model)
    )

    assert payload.get("noise_realizations") == realizations


def test_noise_model_round_trip():
    config = MaestroConfig(noise_model=_mixed_noise_model())
    rebuilt = _over_the_wire(config)

    assert maestro_config_to_payload(rebuilt) == maestro_config_to_payload(config)


def test_rebuilt_noise_model_samples_identically():
    noise_model = _mixed_noise_model(kraus=False)
    rebuilt = _over_the_wire(MaestroConfig(noise_model=noise_model)).noise_model

    assert _bell_counts(rebuilt) == _bell_counts(noise_model)


def test_an_unknown_enum_code_is_rejected():
    with pytest.raises(ValueError, match="99 is not a valid SimulationType"):
        maestro_config_from_payload({"simulation_type": 99})


def test_unknown_keys_are_dropped_with_a_warning():
    with pytest.warns(UserWarning, match=r"\['noisy_device'\]"):
        config = maestro_config_from_payload(
            {"max_bond_dimension": 64, "noisy_device": "ibm_torino"}
        )

    assert config == MaestroConfig(max_bond_dimension=64)


def test_a_failed_setter_is_refused():
    """The call log records a setter before it runs, including one that raised."""
    noise_model = maestro.NoiseModel()
    with pytest.raises(ValueError):
        noise_model.set_depolarizing(0, 2.0)

    with pytest.raises(ValueError, match="do not replay"):
        maestro_config_to_payload(MaestroConfig(noise_model=noise_model))


def test_only_setters_are_replayed():
    """Server-supplied data must not name arbitrary methods to call."""
    payload = {"noise_model": [{"method": "compute_damping", "args": [], "kwargs": {}}]}
    with pytest.raises(ValueError, match="not a set_\\* method"):
        maestro_config_from_payload(payload)
