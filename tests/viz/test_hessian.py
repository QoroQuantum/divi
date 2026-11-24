# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from divi.qprog import ParameterSpace
from divi.viz import GradientMethod, HessianResult, compute_hessian

_QUADRATIC = np.array([[3.0, 1.0], [1.0, 5.0]])


def _quadratic(p):
    return p @ _QUADRATIC @ p


def _sin_cos(p):
    return np.sin(p[0]) * np.cos(p[1])


class TestComputeHessian:
    def test_uses_one_direct_session_and_closes_it_on_failure(
        self, vqe_program, mocker
    ):
        session = mocker.MagicMock()
        session.__enter__.return_value = session
        direct = mocker.patch(
            "divi.qprog.quantum_program.ProgressSession.direct",
            return_value=session,
        )

        with pytest.raises(ValueError, match="center must have shape"):
            compute_hessian(vqe_program, np.zeros(5))

        direct.assert_called_once()
        session.__enter__.assert_called_once_with()
        assert session.__exit__.call_count == 1
        assert session.__exit__.call_args.args[0] is ValueError

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

    def test_hessian_values_correct_for_known_quadratic(
        self, vqe_program, mock_landscape
    ):
        """Verify Hessian of f(x) = 3*x0^2 + 5*x1^2 + 2*x0*x1 is [[6, 2], [2, 10]]."""
        A = _QUADRATIC
        mock_landscape(vqe_program, _quadratic)
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

    def test_hessian_values_correct_for_trig_function(
        self, vqe_program, mock_landscape
    ):
        """Verify Hessian of f(x) = sin(x0)*cos(x1) at (pi/4, 0)."""
        mock_landscape(vqe_program, _sin_cos)
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

    def test_single_parameter(self, vqe_program, mocker, mock_landscape):
        """n_params=1 gives a 1x1 Hessian."""
        mocker.patch("divi.viz._api._n_program_params", return_value=1)
        mock_landscape(vqe_program, lambda p: p[0] ** 2)
        result = compute_hessian(
            vqe_program,
            np.array([1.0]),
            gradient_method=GradientMethod.FINITE_DIFFERENCE,
            eps=1e-4,
        )

        assert result.hessian.shape == (1, 1)
        np.testing.assert_allclose(result.hessian[0, 0], 2.0, atol=1e-8)

    @pytest.mark.parametrize(
        "frequencies, gradient, expected",
        [
            pytest.param(
                ((0.5, 1), (1.0, 1)),
                lambda p: [
                    np.cos(p[0] / 2) * np.cos(p[1]) / 2,
                    -np.sin(p[0] / 2) * np.sin(p[1]),
                ],
                lambda x0, x1: [
                    [
                        -np.sin(x0 / 2) * np.cos(x1) / 4,
                        -np.cos(x0 / 2) * np.sin(x1) / 2,
                    ],
                    [-np.cos(x0 / 2) * np.sin(x1) / 2, -np.sin(x0 / 2) * np.cos(x1)],
                ],
                id="sin(x0/2)*cos(x1)",
            ),
            pytest.param(
                ((1.0, 2), (0.5, 1)),
                lambda p: [
                    -np.sin(p[0]) - 2 * np.sin(2 * p[0]) * np.sin(p[1] / 2),
                    np.cos(2 * p[0]) * np.cos(p[1] / 2) / 2,
                ],
                lambda x0, x1: [
                    [
                        -np.cos(x0) - 4 * np.cos(2 * x0) * np.sin(x1 / 2),
                        -np.sin(2 * x0) * np.cos(x1 / 2),
                    ],
                    [
                        -np.sin(2 * x0) * np.cos(x1 / 2),
                        -np.cos(2 * x0) * np.sin(x1 / 2) / 4,
                    ],
                ],
                id="cos(x0)+cos(2x0)*sin(x1/2)",
            ),
        ],
    )
    def test_program_hessian_applies_the_shift_rule_to_the_gradients(
        self, vqe_program, mocker, frequencies, gradient, expected
    ):
        mocker.patch.object(
            vqe_program,
            "_grad_shift_rule",
            ParameterSpace(frequencies).compute_shift_rule(),
        )
        gradients = mocker.patch.object(
            vqe_program,
            "_evaluate_gradients",
            side_effect=lambda param_sets, cotangents: np.array(
                [gradient(p) for p in param_sets]
            ),
        )
        x0, x1 = 0.4, -0.9

        result = compute_hessian(vqe_program, np.array([x0, x1]))

        np.testing.assert_allclose(result.hessian, expected(x0, x1), atol=1e-10)
        gradients.assert_called_once()
        shifts, _ = vqe_program._grad_shift_rule
        np.testing.assert_array_equal(
            gradients.call_args.args[1], np.ones((len(shifts), 1))
        )

    @pytest.mark.parametrize("program", ["exact_vqe", "supervised_qnn", "pce_program"])
    def test_program_hessian_matches_finite_differences_of_the_loss(
        self, request, program
    ):
        program = request.getfixturevalue(program)
        center = np.random.default_rng(3).uniform(-1.0, 1.0, program.n_params)

        exact = compute_hessian(program, center)
        reference = compute_hessian(
            program,
            center,
            gradient_method=GradientMethod.FINITE_DIFFERENCE,
            eps=1e-3,
        )

        np.testing.assert_allclose(exact.hessian, reference.hessian, atol=1e-5)

    @pytest.mark.parametrize(
        "make_program, atol",
        [
            pytest.param("make_two_qubit_vqe", 0.05, id="linear-vqe"),
            pytest.param("make_supervised_qnn", 0.05, id="supervised-qnn"),
            pytest.param("make_pce", 0.45, id="pce"),
        ],
    )
    def test_program_hessian_stays_accurate_on_a_sampling_backend(
        self,
        request,
        default_test_simulator,
        sampling_test_simulator,
        make_program,
        atol,
    ):
        """Shot noise enters at its own scale, not divided by a small step."""
        build = request.getfixturevalue(make_program)
        program = build(default_test_simulator)
        center = np.random.default_rng(3).uniform(-1.0, 1.0, program.n_params)

        exact = compute_hessian(program, center)
        sampling_test_simulator.set_seed(1997)
        sampled = compute_hessian(build(sampling_test_simulator), center)

        np.testing.assert_allclose(sampled.hessian, exact.hessian, atol=atol)

    def test_fluent_api(self, vqe_program):
        center = np.zeros(2)
        result = vqe_program.viz.compute_hessian(center)

        assert result.hessian.shape == (2, 2)

    def test_fluent_api_defaults_to_best_params(
        self, vqe_program, mocker, mock_landscape
    ):
        best = np.array([0.4, -0.3])
        mocker.patch.object(vqe_program, "_has_run_optimization", return_value=True)
        vqe_program._best_params = best
        mock_landscape(vqe_program, _quadratic)

        result = vqe_program.viz.compute_hessian(
            gradient_method=GradientMethod.FINITE_DIFFERENCE, eps=1e-4
        )

        np.testing.assert_array_equal(result.center, best)
        np.testing.assert_allclose(result.hessian, 2 * _QUADRATIC, atol=1e-8)


_EIGENVECTORS = np.linalg.qr(np.random.default_rng(0).normal(size=(3, 3)))[0]
_HESSIAN = HessianResult(
    hessian=np.eye(3),
    eigenvalues=np.array([1.0, 2.0, 3.0]),
    eigenvectors=_EIGENVECTORS,
    center=np.zeros(3),
    program_type="Synthetic",
)


@pytest.mark.parametrize(
    "method, columns",
    [
        pytest.param("top_eigenvectors", [-1, -2], id="top"),
        pytest.param("bottom_eigenvectors", [0, 1], id="bottom"),
    ],
)
@pytest.mark.parametrize("kwargs", [{}, {"k": 2}], ids=["default_k", "k_2"])
def test_extreme_eigenvectors(method, columns, kwargs):
    vectors = getattr(_HESSIAN, method)(**kwargs)

    assert len(vectors) == 2
    for vector, column in zip(vectors, columns):
        np.testing.assert_array_equal(vector, _EIGENVECTORS[:, column])
