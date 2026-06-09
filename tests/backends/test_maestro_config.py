# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :class:`divi.backends.MaestroConfig`.

Targets the dataclass directly — constructor validation, default values,
override semantics, and pass-through into :class:`MaestroSimulator`.  The
simulator-side noisy-execution paths are exercised in
``test_maestro_simulator.py``; this module guards the config object.
"""

from dataclasses import asdict, fields, replace

import pytest

from divi.backends import MaestroConfig, MaestroSimulator


class TestDefaults:
    """Documented defaults on a bare ``MaestroConfig()``."""

    def test_noise_fields_default_none_and_42(self):
        config = MaestroConfig()
        assert config.noise_model is None
        assert config.noise_seed == 42
        assert config.noise_realizations is None

    def test_all_fields_have_documented_defaults(self):
        """Spot-check every non-noise field against the docstring values."""
        config = MaestroConfig()
        assert config.simulator_type is None
        assert config.simulation_type is None
        assert config.max_bond_dimension is None
        assert config.singular_value_threshold is None
        assert config.use_double_precision is False
        assert config.disable_optimized_swapping is False
        assert config.lookahead_depth == -1
        assert config.mps_measure_no_collapse is True
        assert config.mps_qubit_threshold == 22


class TestExplicitConstruction:
    """Constructing with each noise field set; equality semantics."""

    def test_noise_seed_and_realizations(self):
        config = MaestroConfig(noise_seed=7, noise_realizations=4)
        assert config.noise_seed == 7
        assert config.noise_realizations == 4

    def test_carries_noise_model_object_through(self, mocker):
        """A ``noise_model`` is held by reference, not copied or wrapped."""
        nm = mocker.MagicMock(name="NoiseModel")
        config = MaestroConfig(noise_model=nm)
        assert config.noise_model is nm

    def test_equality_on_value(self):
        """Frozen dataclass — value-equal configs compare equal."""
        a = MaestroConfig(noise_seed=11, noise_realizations=8)
        b = MaestroConfig(noise_seed=11, noise_realizations=8)
        assert a == b

    def test_asdict_round_trip_scalar_fields(self):
        """``asdict`` round-trips configs whose ``noise_model`` is ``None``.

        Scoped to scalar fields — :meth:`dataclasses.asdict` deep-copies
        and recurses, which is not safe in general for a
        ``maestro.NoiseModel`` (a C++-binding object).  See
        :func:`test_carries_noise_model_object_through` for the
        identity-preserving construction path that downstream code
        actually relies on.
        """
        a = MaestroConfig(noise_seed=11, noise_realizations=8)
        b = MaestroConfig(**asdict(a))
        assert a == b

    def test_round_trip_via_replace_preserves_noise_model_identity(self, mocker):
        """``dataclasses.replace`` rebuilds a config without recursing into
        ``noise_model``, so the mock object survives by reference."""
        nm = mocker.MagicMock(name="NoiseModel")
        a = MaestroConfig(noise_model=nm, noise_seed=7)
        b = replace(a)
        assert b.noise_model is nm
        assert b.noise_seed == 7


class TestUnknownKwargRejection:
    """Unknown options must raise — no silent ``**kwargs`` passthrough."""

    def test_unknown_kwarg_raises_type_error(self):
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            MaestroConfig(no_such_field=1)

    def test_unknown_noise_kwarg_raises_type_error(self):
        """Guards against typos like ``noise_realisations`` (British spelling)."""
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            MaestroConfig(noise_realisations=4)


class TestOverride:
    """``MaestroConfig.override`` keeps left-hand values where right-hand is default."""

    def test_override_only_applies_non_default_fields(self):
        base = MaestroConfig(noise_seed=11)
        # Empty override leaves every field at default → noise_seed=11 survives.
        merged = base.override(MaestroConfig())
        assert merged.noise_seed == 11
        assert merged.noise_realizations is None

    def test_override_replaces_when_other_is_non_default(self):
        base = MaestroConfig(noise_seed=11)
        merged = base.override(MaestroConfig(noise_seed=99, noise_realizations=4))
        assert merged.noise_seed == 99
        assert merged.noise_realizations == 4

    def test_override_returns_new_instance(self):
        """``override`` is non-mutating — returns a fresh ``MaestroConfig``."""
        base = MaestroConfig(noise_seed=11)
        merged = base.override(MaestroConfig(noise_seed=99))
        assert merged is not base
        assert base.noise_seed == 11

    def test_override_returns_maestro_config(self):
        merged = MaestroConfig().override(MaestroConfig())
        assert isinstance(merged, MaestroConfig)

    def test_override_handles_pauli_propagation_knobs(self):
        base = MaestroConfig(pp_coefficient_threshold=1e-3)
        # Default (None) override leaves the base value; a non-default replaces.
        assert base.override(MaestroConfig()).pp_coefficient_threshold == 1e-3
        merged = base.override(MaestroConfig(pp_coefficient_threshold=1e-6))
        assert merged.pp_coefficient_threshold == 1e-6

    def test_field_list_has_no_unknown_keys(self):
        """If a field is ever added, this asserts the test suite covers it."""
        known = {
            "simulator_type",
            "simulation_type",
            "max_bond_dimension",
            "singular_value_threshold",
            "use_double_precision",
            "disable_optimized_swapping",
            "lookahead_depth",
            "mps_measure_no_collapse",
            "pp_coefficient_threshold",
            "pp_pauli_weight_threshold",
            "pp_steps_between_trims",
            "mps_qubit_threshold",
            "noise_model",
            "noise_seed",
            "noise_realizations",
        }
        actual = {f.name for f in fields(MaestroConfig)}
        assert actual == known, (
            f"MaestroConfig fields drifted; review override semantics. "
            f"missing={known - actual}, extra={actual - known}"
        )


class TestSimulatorPassThrough:
    """``MaestroSimulator(MaestroConfig(...))`` carries the config verbatim."""

    def test_config_object_reachable_from_simulator(self):
        config = MaestroConfig(noise_seed=13, noise_realizations=5)
        sim = MaestroSimulator(config=config)
        assert sim.config is config

    def test_loose_noise_kwarg_rejected_on_simulator(self):
        """``MaestroSimulator`` does not accept loose noise kwargs — they live on
        :class:`MaestroConfig`."""
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            MaestroSimulator(noise_model=None)
