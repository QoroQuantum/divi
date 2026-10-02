# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :class:`divi.backends.MaestroConfig`.

Guards the config object itself: which simulator options it holds, how it
hands their validation to ``maestro.SimulatorConfig``, and override
semantics.  Execution paths are exercised in ``test_maestro_simulator.py``.
"""

import maestro
import pytest
from pydantic import ValidationError

from divi.backends import MaestroConfig, MaestroSimulator
from tests._helpers import exact_match
from tests.backends._helpers import reset_unknown_message, validation_error_message

_MPS = maestro.SimulationType.MatrixProductState

_SIMULATOR_CONFIG_REFUSAL = (
    "maestro_config must be a MaestroConfig, got SimulatorConfig; "
    "convert it with MaestroConfig.from_simulator_config(...)."
)


def _unknown_option_message(name: str, suggestion: str) -> str:
    return (
        f"MaestroConfig got unknown options '{name}' (did you mean '{suggestion}'?)."
        " Simulator options are maestro.SimulatorConfig's: "
        f"{sorted(maestro.SimulatorConfig._fields)}."
    )


def _mps_simulator_config(**options) -> "maestro.SimulatorConfig":
    return maestro.SimulatorConfig(simulation_type=_MPS, **options)


class TestConstruction:
    def test_bare_config_sets_nothing(self):
        config = MaestroConfig()
        assert config._simulator_options() == {}
        assert config.noise_model is None
        assert config.noise_seed is None
        assert config.noise_realizations is None

    def test_unset_options_read_as_maestros_defaults(self):
        config = MaestroConfig(max_bond_dimension=16, precision="double")
        defaults = maestro.SimulatorConfig()
        assert config.max_bond_dimension == 16
        assert config.simulation_type == defaults.simulation_type
        assert config.lookahead_depth == defaults.lookahead_depth
        assert config.seed is None
        assert config._simulator_options() == {
            "max_bond_dimension": 16,
            "precision": "double",
        }

    def test_unknown_attributes_still_raise(self):
        with pytest.raises(AttributeError):
            MaestroConfig().no_such_option

    @pytest.mark.parametrize("simulation_type", ["MatrixProductState", _MPS])
    def test_enums_accept_names_and_members(self, simulation_type):
        config = MaestroConfig(simulation_type=simulation_type)
        assert config.simulation_type == _MPS

    def test_noise_model_is_held_by_reference(self):
        noise_model = maestro.NoiseModel()
        assert MaestroConfig(noise_model=noise_model).noise_model is noise_model


class TestFromSimulatorConfig:
    def test_copies_the_options_that_differ_from_maestros_defaults(self):
        config = MaestroConfig.from_simulator_config(
            _mps_simulator_config(max_bond_dimension=8)
        )
        assert config == MaestroConfig(simulation_type=_MPS, max_bond_dimension=8)

    def test_later_changes_to_the_object_do_not_leak_in(self):
        simulator_config = _mps_simulator_config(max_bond_dimension=8)
        config = MaestroConfig.from_simulator_config(simulator_config)

        simulator_config.max_bond_dimension = 99

        assert config.max_bond_dimension == 8

    def test_further_fields_are_passed_through(self):
        noise_model = maestro.NoiseModel()
        config = MaestroConfig.from_simulator_config(
            _mps_simulator_config(max_bond_dimension=8),
            max_bond_dimension=16,
            noise_model=noise_model,
        )
        assert config.max_bond_dimension == 16
        assert config.simulation_type == _MPS
        assert config.noise_model is noise_model


class TestUnknownOptions:
    @pytest.mark.parametrize(
        "name, suggestion",
        [
            ("max_bond_dim", "max_bond_dimension"),
            ("noise_realisations", "noise_realizations"),
        ],
    )
    def test_a_misspelt_option_names_the_closest_match(self, name, suggestion):
        assert validation_error_message(
            lambda: MaestroConfig(**{name: 1})
        ) == _unknown_option_message(name, suggestion)

    def test_shots_point_to_where_they_are_set(self):
        with pytest.raises(
            ValidationError, match="set on MaestroSimulator or JobConfig"
        ):
            MaestroConfig(shots=100)

    def test_the_error_lists_maestro_options(self):
        with pytest.raises(ValidationError, match="pp_gates_between_trims"):
            MaestroConfig(no_such_option=1)


class TestMaestroValidates:
    """Values are checked by ``maestro.SimulatorConfig``, when the config is built."""

    def test_an_invalid_value_is_rejected(self):
        with pytest.raises(ValidationError, match="precision must be one of"):
            MaestroConfig(precision="quad")

    def test_a_wrongly_typed_value_is_rejected(self):
        with pytest.raises(ValidationError, match="rejected the options"):
            MaestroConfig(max_bond_dimension="large")

    def test_an_unknown_enum_name_is_rejected(self):
        assert validation_error_message(
            lambda: MaestroConfig(simulation_type="Nope")
        ) == (
            f"simulation_type must be one of "
            f"{sorted(maestro.SimulationType.__members__)}. Got 'Nope'."
        )


class TestOverrideAndReset:
    def test_override_sets_the_given_fields_and_keeps_the_rest(self):
        noise_model = maestro.NoiseModel()
        base = MaestroConfig(simulation_type=_MPS, max_bond_dimension=16, noise_seed=1)
        result = base.override(max_bond_dimension=32, noise_model=noise_model)
        assert result == MaestroConfig(
            simulation_type=_MPS,
            max_bond_dimension=32,
            noise_seed=1,
            noise_model=noise_model,
        )

    @pytest.mark.parametrize(
        "change",
        [
            lambda: MaestroConfig().override(max_bond_dim=32),
            lambda: MaestroConfig().model_copy(update={"max_bond_dim": 32}),
        ],
        ids=["override", "model_copy"],
    )
    def test_changes_validate_like_the_constructor(self, change):
        assert validation_error_message(change) == _unknown_option_message(
            "max_bond_dim", "max_bond_dimension"
        )

    def test_reset_restores_maestros_default(self):
        base = MaestroConfig(simulation_type=_MPS, max_bond_dimension=16)
        assert base.reset("simulation_type") == MaestroConfig(max_bond_dimension=16)

    def test_reset_noise_model_turns_noise_off(self):
        base = MaestroConfig(noise_model=maestro.NoiseModel())
        assert base.reset("noise_model").noise_model is None

    def test_reset_rejects_unknown_fields(self):
        with pytest.raises(
            ValueError,
            match=exact_match(
                reset_unknown_message("max_bond_dim", "max_bond_dimension")
            ),
        ):
            MaestroConfig().reset("max_bond_dim")

    def test_simulator_options_are_listed_by_dir(self):
        assert "max_bond_dimension" in dir(MaestroConfig())

    def test_model_copy_without_update_is_an_equal_copy(self):
        config = MaestroConfig(max_bond_dimension=8)
        assert config.model_copy() == config


class TestSimulatorPassThrough:
    def test_set_seed_replaces_only_the_seed(self):
        noise_model = maestro.NoiseModel()
        sim = MaestroSimulator(
            maestro_config=MaestroConfig(
                seed=1, max_bond_dimension=8, noise_model=noise_model
            )
        )
        sim.set_seed(2)
        assert sim.maestro_config.seed == 2
        assert sim.maestro_config.max_bond_dimension == 8
        assert sim.maestro_config.noise_model is noise_model

    @pytest.mark.parametrize(
        "give",
        [
            lambda value: MaestroSimulator(maestro_config=value),
            lambda value: setattr(MaestroSimulator(), "maestro_config", value),
        ],
        ids=["constructor", "assignment"],
    )
    @pytest.mark.parametrize(
        "make_value, message",
        [
            (maestro.SimulatorConfig, _SIMULATOR_CONFIG_REFUSAL),
            (dict, "maestro_config must be a MaestroConfig, got dict."),
        ],
        ids=["simulator_config_with_pointer", "dict"],
    )
    def test_a_non_maestro_config_is_refused(self, give, make_value, message):
        value = make_value()
        with pytest.raises(TypeError, match=exact_match(message)):
            give(value)

    def test_loose_noise_kwarg_rejected_on_simulator(self):
        """Noise settings live on :class:`MaestroConfig`, not the simulator."""
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            MaestroSimulator(noise_model=None)
