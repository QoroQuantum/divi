import pytest
from qprog.vqe.vqe import VQE


@pytest.mark.algo
def test_vqe_initialization():
    vqe_problem = VQE(symbols=["H", "H"], bond_lengths=[
                      0.5, 1.0], coordinate_structure=[(1, 0, 0), (0, -1, 0)])
    assert vqe_problem is not None, "VQE should be initialized"
    assert len(
        vqe_problem.hamiltonian_ops) == 2, "Hamiltonian operators should be generated"
