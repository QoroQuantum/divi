# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pennylane as qml
import pytest

from divi.qprog import VQE, GenericLayerAnsatz
from divi.viz import GradientMethod, compute_hessian


@pytest.fixture
def basic_ansatz():
    return GenericLayerAnsatz([qml.RY, qml.RZ])


@pytest.fixture
def vqe_program(dummy_simulator, basic_ansatz):
    return VQE(
        hamiltonian=qml.Z(0),
        n_electrons=1,
        ansatz=basic_ansatz,
        n_layers=1,
        backend=dummy_simulator,
    )


class TestComputeHessian:
    def test_returns_symmetric_matrix(self, vqe_program):
        center = np.zeros(2)
        result = compute_hessian(vqe_program, center)

        np.testing.assert_allclose(result.hessian, result.hessian.T, atol=1e-8)

    def test_shapes(self, vqe_program):
        center = np.zeros(2)
        result = compute_hessian(vqe_program, center)

        assert result.hessian.shape == (2, 2)
        assert result.eigenvalues.shape == (2,)
        assert result.eigenvectors.shape == (2, 2)
        np.testing.assert_allclose(result.center, center)
        assert result.program_type == "VQE"

    def test_eigenvalues_ascending(self, vqe_program):
        center = np.array([0.5, 0.3])
        result = compute_hessian(vqe_program, center)

        assert np.all(np.diff(result.eigenvalues) >= -1e-12)

    def test_eigenvectors_orthonormal(self, vqe_program):
        center = np.array([0.1, 0.2])
        result = compute_hessian(vqe_program, center)

        product = result.eigenvectors.T @ result.eigenvectors
        np.testing.assert_allclose(product, np.eye(2), atol=1e-8)

    def test_top_eigenvectors(self, vqe_program):
        center = np.zeros(2)
        result = compute_hessian(vqe_program, center)

        top = result.top_eigenvectors(k=1)
        assert len(top) == 1
        np.testing.assert_allclose(top[0], result.eigenvectors[:, -1])

    def test_bottom_eigenvectors(self, vqe_program):
        center = np.zeros(2)
        result = compute_hessian(vqe_program, center)

        bottom = result.bottom_eigenvectors(k=1)
        assert len(bottom) == 1
        np.testing.assert_allclose(bottom[0], result.eigenvectors[:, 0])

    def test_rejects_wrong_shape(self, vqe_program):
        with pytest.raises(ValueError, match="center must have shape"):
            compute_hessian(vqe_program, np.zeros(5))

    def test_rejects_non_positive_eps(self, vqe_program):
        with pytest.raises(ValueError, match="eps must be positive"):
            compute_hessian(
                vqe_program,
                np.zeros(2),
                gradient_method=GradientMethod.FINITE_DIFFERENCE,
                eps=0.0,
            )

    def test_hessian_values_correct_for_known_quadratic(self, vqe_program, mocker):
        """Verify Hessian of f(x) = 3*x0^2 + 5*x1^2 + 2*x0*x1 is [[6, 2], [2, 10]]."""
        A = np.array([[3.0, 1.0], [1.0, 5.0]])

        def _mock_eval(param_sets, **kwargs):
            return {
                i: float(p @ A @ p) for i, p in enumerate(np.atleast_2d(param_sets))
            }

        mocker.patch.object(
            vqe_program, "_evaluate_cost_param_sets", side_effect=_mock_eval
        )
        center = np.array([1.0, -0.5])
        result = compute_hessian(
            vqe_program,
            center,
            gradient_method=GradientMethod.FINITE_DIFFERENCE,
            eps=1e-4,
        )

        # Hessian of x^T A x is 2A (symmetric part = A + A^T = 2A when A is symmetric).
        # The mock is an exact quadratic (zero fourth derivatives), so truncation
        # error is zero — only float rounding through the eps^2 divisor.
        expected = 2 * A
        np.testing.assert_allclose(result.hessian, expected, atol=1e-10)

    def test_hessian_values_correct_for_trig_function(self, vqe_program, mocker):
        """Verify Hessian of f(x) = sin(x0)*cos(x1) at (pi/4, 0)."""

        def _mock_eval(param_sets, **kwargs):
            ps = np.atleast_2d(param_sets)
            return {i: float(np.sin(p[0]) * np.cos(p[1])) for i, p in enumerate(ps)}

        mocker.patch.object(
            vqe_program, "_evaluate_cost_param_sets", side_effect=_mock_eval
        )
        # At (pi/4, 0):
        #   d2f/dx0^2 = -sin(pi/4)*cos(0) = -sqrt(2)/2
        #   d2f/dx1^2 = -sin(pi/4)*cos(0) = -sqrt(2)/2
        #   d2f/dx0dx1 = cos(pi/4)*(-sin(0)) = 0
        center = np.array([np.pi / 4, 0.0])
        result = compute_hessian(
            vqe_program,
            center,
            gradient_method=GradientMethod.FINITE_DIFFERENCE,
            eps=1e-4,
        )

        s = -np.sqrt(2) / 2
        expected = np.array([[s, 0.0], [0.0, s]])
        np.testing.assert_allclose(result.hessian, expected, atol=1e-5)

    def test_single_parameter(self, vqe_program, mocker):
        """n_params=1: off-diagonal loop is empty, 1x1 Hessian."""
        mocker.patch("divi.viz._api._n_program_params", return_value=1)

        def _mock_eval(param_sets, **kwargs):
            ps = np.atleast_2d(param_sets)
            return {i: float(p[0] ** 2) for i, p in enumerate(ps)}

        mocker.patch.object(
            vqe_program, "_evaluate_cost_param_sets", side_effect=_mock_eval
        )
        result = compute_hessian(
            vqe_program,
            np.array([1.0]),
            gradient_method=GradientMethod.FINITE_DIFFERENCE,
            eps=1e-4,
        )

        assert result.hessian.shape == (1, 1)
        np.testing.assert_allclose(result.hessian[0, 0], 2.0, atol=1e-8)

    def test_parameter_shift_hessian_for_trig(self, vqe_program, mocker):
        """Verify parameter-shift Hessian of f(x) = sin(x0)*cos(x1) at (pi/4, 0)."""

        def _mock_eval(param_sets, **kwargs):
            ps = np.atleast_2d(param_sets)
            return {i: float(np.sin(p[0]) * np.cos(p[1])) for i, p in enumerate(ps)}

        mocker.patch.object(
            vqe_program, "_evaluate_cost_param_sets", side_effect=_mock_eval
        )
        center = np.array([np.pi / 4, 0.0])
        result = compute_hessian(
            vqe_program,
            center,
            gradient_method=GradientMethod.PARAMETER_SHIFT,
        )

        s = -np.sqrt(2) / 2
        expected = np.array([[s, 0.0], [0.0, s]])
        # Parameter-shift is exact for trig functions.
        np.testing.assert_allclose(result.hessian, expected, atol=1e-10)

    def test_finite_difference_hessian_explicit(self, vqe_program, mocker):
        """Verify finite-difference method can be selected explicitly."""
        A = np.array([[3.0, 1.0], [1.0, 5.0]])

        def _mock_eval(param_sets, **kwargs):
            return {
                i: float(p @ A @ p) for i, p in enumerate(np.atleast_2d(param_sets))
            }

        mocker.patch.object(
            vqe_program, "_evaluate_cost_param_sets", side_effect=_mock_eval
        )
        result = compute_hessian(
            vqe_program,
            np.array([1.0, -0.5]),
            gradient_method=GradientMethod.FINITE_DIFFERENCE,
            eps=1e-4,
        )

        np.testing.assert_allclose(result.hessian, 2 * A, atol=1e-10)

    def test_fluent_api(self, vqe_program):
        center = np.zeros(2)
        result = vqe_program.viz.compute_hessian(center)

        assert result.hessian.shape == (2, 2)
