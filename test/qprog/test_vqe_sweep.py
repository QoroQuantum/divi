from itertools import product

import pytest

from divi.qprog import Optimizers, VQEAnsatze, VQEHyperparameterSweep


@pytest.fixture
def vqe_sweep():
    bond_lengths = [0.5, 1.0, 1.5]
    ansatze = [VQEAnsatze.UCCSD, VQEAnsatze.RY]
    symbols = ["H", "H"]
    coordinate_structure = [[0, 0, 0], [0, 0, 1]]
    optimizer = Optimizers.MONTE_CARLO
    max_iterations = 10
    shots = 5000

    return VQEHyperparameterSweep(
        bond_lengths=bond_lengths,
        ansatze=ansatze,
        symbols=symbols,
        coordinate_structure=coordinate_structure,
        optimizer=optimizer,
        max_iterations=max_iterations,
        shots=shots,
    )


def test_correct_number_of_programs_created(mocker, vqe_sweep):
    mocker.patch("divi.qprog.VQE")

    vqe_sweep.create_programs()

    assert len(vqe_sweep.programs) == len(vqe_sweep.bond_lengths) * len(
        vqe_sweep.ansatze
    )

    assert all(
        (ansatz, bond_length) in vqe_sweep.programs
        for ansatz, bond_length in product(vqe_sweep.ansatze, vqe_sweep.bond_lengths)
    )

    # Assert common values propagated to all programs
    for program in vqe_sweep.programs.values():
        assert program.optimizer == Optimizers.MONTE_CARLO
        assert program.max_iterations == 10
        assert program.shots == 5000
        assert program.coordinate_structure == [[0, 0, 0], [0, 0, 1]]
        assert program.symbols == ["H", "H"]


def test_fail_if_creating_programs_twice(mocker, vqe_sweep):
    mocker.patch("divi.qprog.VQE")
    vqe_sweep.programs = {"dummy": "program"}

    with pytest.raises(RuntimeError, match="Some programs already exist"):
        vqe_sweep.create_programs()


def test_results_aggregated_correctly(mocker, vqe_sweep):
    mocker.patch("divi.qprog.VQE")

    mock_program_1 = mocker.MagicMock()
    mock_program_1.energies = [{0: -1.0}]

    mock_program_2 = mocker.MagicMock()
    mock_program_2.energies = [{0: -2.0}]

    vqe_sweep.programs = {
        (VQEAnsatze.UCCSD, 0.5): mock_program_1,
        (VQEAnsatze.RY, 1.0): mock_program_2,
    }

    smallest_key, smallest_value = vqe_sweep.aggregate_results()

    assert smallest_key == (VQEAnsatze.RY, 1.0)
    assert smallest_value == -2.0


def test_results_aggregated_correctly_multiple_energies(mocker, vqe_sweep):
    mocker.patch("divi.qprog.VQE")

    mock_program_1 = mocker.MagicMock()
    mock_program_1.energies = [{0: -1.0}, {0: -0.8}]
    mock_program_2 = mocker.MagicMock()
    mock_program_2.energies = [{0: -0.9}, {0: -0.7}]

    vqe_sweep.programs = {
        (VQEAnsatze.UCCSD, 0.5): mock_program_1,
        (VQEAnsatze.RY, 1.0): mock_program_2,
    }

    smallest_key, smallest_value = vqe_sweep.aggregate_results()

    assert smallest_key == (VQEAnsatze.UCCSD, 0.5)
    assert smallest_value == -0.8


def test_fail_when_aggregating_with_no_programs(vqe_sweep):
    vqe_sweep.programs = {}
    with pytest.raises(RuntimeError, match="No programs to aggregate"):
        vqe_sweep.aggregate_results()


@pytest.mark.parametrize("graph_type", ["line", "scatter"])
def test_visualize_results(mocker, vqe_sweep, graph_type):
    mocker.patch("divi.qprog.VQE")

    # Just to silence the matplotlib warnings
    mocker.patch("matplotlib.pyplot.legend")

    mock_plot = mocker.patch(
        f"matplotlib.pyplot.{graph_type if graph_type == 'scatter' else 'plot'}"
    )
    mock_show = mocker.patch("matplotlib.pyplot.show")

    mock_program = mocker.MagicMock()
    mock_program.energies = [{0: -1.0}]

    vqe_sweep.programs = {
        (ansatz, bond_length): mock_program
        for ansatz, bond_length in product(vqe_sweep.ansatze, vqe_sweep.bond_lengths)
    }

    vqe_sweep.visualize_results(graph_type=graph_type)
    mock_show.assert_called_once()
    mock_plot.assert_called()


def test_visualize_results_with_invalid_graph_type(mocker, vqe_sweep):
    mocker.patch("divi.qprog.VQE")
    mock_show = mocker.patch("matplotlib.pyplot.show")

    mock_program = mocker.MagicMock()
    mock_program.energies = [{0: -1.0}]

    vqe_sweep.programs = {
        (ansatz, bond_length): mock_program
        for ansatz, bond_length in product(vqe_sweep.ansatze, vqe_sweep.bond_lengths)
    }

    with pytest.raises(ValueError, match="Invalid graph type"):
        vqe_sweep.visualize_results(graph_type="invalid")

    mock_show.assert_not_called()
