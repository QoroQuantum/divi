import pytest
from qprog.vqe.vqe import VQE


@pytest.mark.algo
def test_vqe_initialization():
    vqe_problem = VQE(symbols=["H", "H"], bond_lengths=[
                      0.5, 1.0], coordinate_structure=[(1, 0, 0), (0, -1, 0)])
    assert vqe_problem is not None, "VQE should be initialized"
    assert len(
        vqe_problem.hamiltonian_ops) == 2, "Hamiltonian operators should be generated"


@pytest.mark.algo
def test_vqe_initialization_fail():
    # Need to have the same number of symbols and coordinates
    pytest.raises(AssertionError, VQE, symbols=["H", "H", "H"], bond_lengths=[
        0.5, 1.0], coordinate_structure=[(1, 0, 0), (0, -1, 0)])


@pytest.mark.algo
def test_all_equal():
    def all_equal(iterator):
        iterator = iter(iterator)
        try:
            first = next(iterator)
        except StopIteration:
            return True
        return all(first == x for x in iterator)

    assert True == all_equal([1, 1, 1]), "All elements should be equal"
    assert False == all_equal([1, 2, 1]), "Not all elements are equal"
