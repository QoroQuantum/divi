# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :class:`divi.backends.MaestroConfig`.

Targets the dataclass directly — constructor validation, default values,
override semantics, and pass-through into :class:`MaestroSimulator`.  The
simulator-side noisy-execution paths are exercised in
``test_maestro_simulator.py``; this module guards the config object.
"""

import warnings

import maestro
import pytest
from pydantic import ValidationError

from divi.backends import MaestroConfig, MaestroSimulator
from divi.backends.runners._maestro import (
    BOOLEAN_FLAG_FIELDS,
    GPU_SVD_FLAG_GROUPS,
    KRAUS_COMPLETENESS_CHECKS,
    TRUNCATION_MODES,
)

# MaestroConfig fields never forwarded to maestro.SimulatorConfig.
DIVI_ONLY_FIELDS = frozenset(
    {"mps_qubit_threshold", "noise_model", "noise_seed", "noise_realizations"}
)


def upstream_config_fields() -> set[str]:
    """Writable data attributes bound on ``maestro.SimulatorConfig``."""
    return {
        name
        for name in dir(maestro.SimulatorConfig)
        if not name.startswith("_")
        and not callable(getattr(maestro.SimulatorConfig, name))
    }


class TestDefaults:
    """Documented defaults on a bare ``MaestroConfig()``."""

    def test_noise_fields_default_none(self):
        config = MaestroConfig()
        assert config.noise_model is None
        assert config.noise_seed is None
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
        assert config.truncation_mode is None
        assert config.seed is None
        assert config.gpu_device is None
        assert config.distributed_options is None
        assert config.mpo_kraus_completeness_check is None
        for field in BOOLEAN_FLAG_FIELDS:
            assert getattr(config, field) is False, field


class TestExplicitConstruction:
    """Constructing with each noise field set; equality semantics."""

    def test_noise_seed_and_realizations(self):
        config = MaestroConfig(noise_seed=7, noise_realizations=4)
        assert config.noise_seed == 7
        assert config.noise_realizations == 4

    def test_carries_noise_model_object_through(self):
        """A ``noise_model`` is held by reference, not copied or wrapped."""
        nm = maestro.NoiseModel()
        config = MaestroConfig(noise_model=nm)
        assert config.noise_model is nm

    def test_frozen(self):
        """Fields cannot be reassigned after construction."""
        config = MaestroConfig(simulation_type="Statevector")
        with pytest.raises(ValidationError):
            config.simulation_type = "MatrixProductState"

    def test_equality_on_value(self):
        """Value-equal configs compare equal."""
        a = MaestroConfig(noise_seed=11, noise_realizations=8)
        b = MaestroConfig(noise_seed=11, noise_realizations=8)
        assert a == b

    def test_dump_round_trip_scalar_fields(self):
        """``model_dump`` round-trips configs whose ``noise_model`` is ``None``.

        Scoped to scalar fields — dumping recurses, which is not safe in
        general for a ``maestro.NoiseModel`` (a C++-binding object).  See
        :func:`test_carries_noise_model_object_through` for the
        identity-preserving construction path that downstream code
        actually relies on.
        """
        a = MaestroConfig(noise_seed=11, noise_realizations=8)
        b = MaestroConfig(**a.model_dump())
        assert a == b

    def test_round_trip_via_copy_preserves_noise_model_identity(self):
        """``model_copy`` rebuilds a config without recursing into
        ``noise_model``, so the object survives by reference."""
        nm = maestro.NoiseModel()
        a = MaestroConfig(noise_model=nm, noise_seed=7)
        b = a.model_copy()
        assert b.noise_model is nm
        assert b.noise_seed == 7


class TestUnknownKwargRejection:
    """Unknown options must raise — no silent ``**kwargs`` passthrough."""

    def test_unknown_kwarg_is_rejected(self):
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            MaestroConfig(no_such_field=1)

    def test_unknown_noise_kwarg_is_rejected(self):
        """Guards against typos like ``noise_realisations`` (British spelling)."""
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
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
            "precision",
            "disable_optimized_swapping",
            "lookahead_depth",
            "mps_measure_no_collapse",
            "pp_coefficient_threshold",
            "pp_pauli_weight_threshold",
            "pp_steps_between_trims",
            "pp_steps_between_deduplications",
            "path_integral_threshold",
            "truncation_mode",
            "seed",
            "gpu_device",
            "distributed_options",
            "mpo_kraus_completeness_check",
            "mps_qubit_threshold",
            "noise_model",
            "noise_seed",
            "noise_realizations",
            *BOOLEAN_FLAG_FIELDS,
        }
        actual = set(MaestroConfig.model_fields)
        assert actual == known, (
            f"MaestroConfig fields drifted; review override semantics. "
            f"missing={known - actual}, extra={actual - known}"
        )


class TestUpstreamFieldParity:
    """``MaestroConfig`` and ``maestro.SimulatorConfig`` expose the same knobs."""

    def test_every_forwarded_field_exists_upstream(self):
        """A renamed or removed upstream knob fails here, not at execution time."""
        forwarded = set(MaestroConfig.model_fields) - DIVI_ONLY_FIELDS
        missing = forwarded - upstream_config_fields()
        assert not missing, (
            "MaestroConfig forwards fields that maestro.SimulatorConfig no longer "
            f"binds: {sorted(missing)}. Rename them or move them into "
            "DIVI_ONLY_FIELDS."
        )

    def test_every_upstream_field_is_exposed(self):
        """A knob added upstream is invisible to Divi users until wired in here."""
        unexposed = upstream_config_fields() - set(MaestroConfig.model_fields)
        assert not unexposed, (
            "maestro.SimulatorConfig binds knobs MaestroConfig does not expose: "
            f"{sorted(unexposed)}. Add a field and forward it in "
            "_to_maestro_config."
        )

    def test_divi_only_fields_are_real_fields(self):
        """Guards the exclusion set itself against a rename on the Divi side."""
        assert DIVI_ONLY_FIELDS <= set(MaestroConfig.model_fields)


class TestToMaestroConfig:
    """``_to_maestro_config`` builds a real ``maestro.SimulatorConfig``."""

    def test_pauli_propagation_knobs_reach_maestro(self):
        """Maestro binds these as properties, not constructor arguments."""
        config = MaestroConfig(
            simulation_type="PauliPropagator",
            pp_coefficient_threshold=1e-3,
            pp_pauli_weight_threshold=4,
            pp_steps_between_trims=2,
            pp_steps_between_deduplications=3,
        )
        sim_config = config._to_maestro_config(n_qubits=6)
        assert sim_config.pp_coefficient_threshold == 1e-3
        assert sim_config.pp_pauli_weight_threshold == 4
        assert sim_config.pp_steps_between_trims == 2
        assert sim_config.pp_steps_between_deduplications == 3

    def test_unset_pauli_propagation_knobs_stay_none(self):
        sim_config = MaestroConfig(
            simulation_type="PauliPropagator"
        )._to_maestro_config(n_qubits=6)
        assert sim_config.pp_coefficient_threshold is None
        assert sim_config.pp_pauli_weight_threshold is None
        assert sim_config.pp_steps_between_trims is None

    def test_precision_and_path_integral_threshold_reach_maestro(self):
        """Also property-only knobs, like the pp_* family."""
        sim_config = MaestroConfig(
            simulation_type="PathIntegral",
            precision=True,
            path_integral_threshold=1e-6,
        )._to_maestro_config(n_qubits=4)
        assert sim_config.precision is True
        assert sim_config.path_integral_threshold == 1e-6

    def test_mps_fields_still_forwarded(self):
        config = MaestroConfig(
            simulation_type="MatrixProductState",
            max_bond_dimension=32,
            singular_value_threshold=1e-8,
        )
        sim_config = config._to_maestro_config(n_qubits=6)
        assert sim_config.max_bond_dimension == 32
        assert sim_config.singular_value_threshold == 1e-8

    def test_constructor_knobs_reach_maestro(self):
        """``truncation_mode``, ``seed``, ``gpu_device`` and
        ``distributed_options`` are constructor args."""
        options = {"distributed_devices": "0,1", "mpi_communicator": "0"}
        sim_config = MaestroConfig(
            simulation_type="MatrixProductState",
            truncation_mode="relative_max",
            seed=1234,
            gpu_device=1,
            distributed_options=options,
        )._to_maestro_config(n_qubits=6)
        assert sim_config.truncation_mode == "relative_max"
        assert sim_config.seed == 1234
        assert sim_config.gpu_device == 1
        assert sim_config.distributed_options == options

    def test_unset_distributed_options_stay_empty(self):
        sim_config = MaestroConfig()._to_maestro_config(n_qubits=4)
        assert sim_config.distributed_options == {}

    def test_mpo_kraus_completeness_check_reaches_maestro(self):
        """Property-only, like the pp_* family."""
        sim_config = MaestroConfig(
            simulation_type="MatrixProductOperator",
            mpo_kraus_completeness_check="strict",
        )._to_maestro_config(n_qubits=4)
        assert sim_config.mpo_kraus_completeness_check == "strict"

    @pytest.mark.parametrize("solver", ("gesvd", "gesvdj", "gesvdp", "gesvdr"))
    def test_boolean_flags_reach_maestro_when_set(self, solver):
        """One solver per group is legal — the exclusion is within a group."""
        flags = {
            f"{prefix}_use_{solver}": True
            for prefix in ("mps", "mpo", "tensor_network")
        }
        flags["mpo_restore_trace_after_truncation"] = True
        flags["mpo_hermitize_after_truncation"] = True

        sim_config = MaestroConfig(**flags)._to_maestro_config(n_qubits=4)

        for field in flags:
            assert getattr(sim_config, field) is True, field

    def test_unset_boolean_flags_stay_false(self):
        """Nothing is forwarded on a bare config."""
        sim_config = MaestroConfig()._to_maestro_config(n_qubits=4)
        for field in BOOLEAN_FLAG_FIELDS:
            assert getattr(sim_config, field) is False, field

    def test_auto_mps_selection_survives(self):
        """Above ``mps_qubit_threshold`` the built config switches to MPS."""
        sim_config = MaestroConfig(mps_qubit_threshold=4)._to_maestro_config(n_qubits=8)
        assert sim_config.simulation_type == maestro.SimulationType.MatrixProductState
        assert sim_config.max_bond_dimension == 64


class TestPauliPropagationValidation:
    """Rejects unusable Pauli-propagation settings and flags no-op ones."""

    @pytest.mark.parametrize(
        "field", ["pp_steps_between_trims", "pp_steps_between_deduplications"]
    )
    @pytest.mark.parametrize("cadence", [0, -1])
    def test_non_positive_cadence_rejected(self, field, cadence):
        """Maestro takes these modulo a gate index; 0 aborts with SIGFPE."""
        with pytest.raises(ValueError, match="must be a positive integer"):
            MaestroConfig(simulation_type="PauliPropagator", **{field: cadence})

    @pytest.mark.parametrize(
        "field, value",
        [("pp_coefficient_threshold", -1e-3), ("pp_pauli_weight_threshold", -1)],
    )
    def test_negative_threshold_rejected(self, field, value):
        with pytest.raises(ValueError, match="must be non-negative"):
            MaestroConfig(simulation_type="PauliPropagator", **{field: value})

    @pytest.mark.parametrize(
        "field, value",
        [("pp_coefficient_threshold", 1e-3), ("pp_pauli_weight_threshold", 2)],
    )
    def test_threshold_without_cadence_warns(self, field, value):
        config = MaestroConfig(simulation_type="PauliPropagator", **{field: value})
        with pytest.warns(UserWarning, match="no effect unless"):
            config._to_maestro_config(n_qubits=6)

    def test_cadence_makes_threshold_active_without_warning(self):
        config = MaestroConfig(
            simulation_type="PauliPropagator",
            pp_coefficient_threshold=1e-3,
            pp_steps_between_deduplications=1,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            config._to_maestro_config(n_qubits=6)

    def test_pp_knobs_on_other_simulation_type_warn(self):
        config = MaestroConfig(
            simulation_type="MatrixProductState", pp_steps_between_trims=1
        )
        with pytest.warns(UserWarning, match="only apply to PauliPropagator"):
            config._to_maestro_config(n_qubits=6)

    def test_partial_config_used_as_override_delta_is_silent(self):
        """A threshold-only delta is legitimate; the merged config is what counts."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            delta = MaestroConfig(pp_coefficient_threshold=1e-6)
            merged = MaestroConfig(
                simulation_type="PauliPropagator", pp_steps_between_trims=1
            ).override(delta)
            merged._to_maestro_config(n_qubits=6)

    def test_weight_threshold_at_qubit_count_warns(self):
        config = MaestroConfig(
            simulation_type="PauliPropagator",
            pp_pauli_weight_threshold=4,
            pp_steps_between_trims=1,
        )
        with pytest.warns(UserWarning, match="disables weight filtering"):
            config._to_maestro_config(n_qubits=4)

    def test_weight_threshold_below_qubit_count_is_silent(self):
        config = MaestroConfig(
            simulation_type="PauliPropagator",
            pp_pauli_weight_threshold=3,
            pp_steps_between_trims=1,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            config._to_maestro_config(n_qubits=4)


class TestBackendKnobValidation:
    """Rejects values maestro would silently ignore or abort on."""

    def test_invalid_truncation_mode_rejected(self):
        with pytest.raises(ValueError, match="truncation_mode must be one of"):
            MaestroConfig(truncation_mode="relative")

    def test_invalid_kraus_completeness_check_rejected(self):
        with pytest.raises(
            ValueError, match="mpo_kraus_completeness_check must be one of"
        ):
            MaestroConfig(mpo_kraus_completeness_check="raise")

    def test_allow_listed_values_are_accepted(self):
        for mode in TRUNCATION_MODES:
            assert MaestroConfig(truncation_mode=mode).truncation_mode == mode
        for check in KRAUS_COMPLETENESS_CHECKS:
            config = MaestroConfig(mpo_kraus_completeness_check=check)
            assert config.mpo_kraus_completeness_check == check

    def test_negative_gpu_device_rejected(self):
        """Maestro's own setter raises; divi fails at construction instead."""
        with pytest.raises(ValueError, match="gpu_device must be a non-negative"):
            MaestroConfig(gpu_device=-1)

    @pytest.mark.parametrize("field", ["seed", "noise_seed"])
    @pytest.mark.parametrize("value", [-1, 2**32])
    def test_seed_outside_uint32_rejected(self, field, value):
        """Maestro takes an unsigned 32-bit seed."""
        with pytest.raises(ValueError, match=rf"{field} must be an integer in"):
            MaestroConfig(**{field: value})

    @pytest.mark.parametrize("field", ["simulator_type", "simulation_type"])
    def test_unknown_enum_name_rejected(self, field):
        with pytest.raises(ValueError, match=rf"{field} must be one of"):
            MaestroConfig(**{field: "Nope"})

    @pytest.mark.parametrize("realizations", [0, -1])
    def test_non_positive_noise_realizations_rejected(self, realizations):
        with pytest.raises(ValueError, match="noise_realizations must be None"):
            MaestroConfig(noise_realizations=realizations)

    def test_non_maestro_noise_model_rejected(self, mocker):
        with pytest.raises(ValueError, match="must be a maestro.NoiseModel"):
            MaestroConfig(noise_model=mocker.MagicMock(name="NoiseModel"))

    def test_unprefixed_distributed_option_rejected(self):
        """Maestro raises at execution time; divi fails at construction instead."""
        with pytest.raises(ValueError, match=r"Got \['gpu_device'\]"):
            MaestroConfig(
                distributed_options={"distributed_flags": "8", "gpu_device": "0"}
            )

    def test_non_string_distributed_option_value_rejected(self):
        with pytest.raises(ValidationError, match="distributed_options"):
            MaestroConfig(distributed_options={"distributed_flags": 8})

    @pytest.mark.parametrize("group", GPU_SVD_FLAG_GROUPS)
    def test_two_svd_solvers_in_one_group_rejected(self, group):
        """Maestro applies one solver per backend, so the winner would be arbitrary."""
        with pytest.raises(ValueError, match="At most one of"):
            MaestroConfig(**{group[0]: True, group[1]: True})


class TestSimulatorPassThrough:
    """``MaestroSimulator(MaestroConfig(...))`` carries the config verbatim."""

    def test_config_object_reachable_from_simulator(self):
        config = MaestroConfig(noise_seed=13, noise_realizations=5)
        sim = MaestroSimulator(config=config)
        assert sim.config is config

    def test_set_seed_replaces_only_the_seed(self):
        nm = maestro.NoiseModel()
        sim = MaestroSimulator(
            config=MaestroConfig(seed=1, noise_seed=13, noise_model=nm)
        )
        sim.set_seed(2)
        assert sim.config.seed == 2
        assert sim.config.noise_seed == 13
        assert sim.config.noise_model is nm

    def test_set_seed_rejects_negative(self):
        sim = MaestroSimulator()
        with pytest.raises(ValueError, match="seed must be an integer in"):
            sim.set_seed(-1)

    def test_loose_noise_kwarg_rejected_on_simulator(self):
        """``MaestroSimulator`` does not accept loose noise kwargs — they live on
        :class:`MaestroConfig`."""
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            MaestroSimulator(noise_model=None)
