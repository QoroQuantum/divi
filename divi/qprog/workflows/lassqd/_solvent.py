# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""The continuum solvent of a LASSQD run and its reaction field."""

from collections.abc import Callable, Sequence
from math import isfinite
from numbers import Real
from typing import Any

import numpy as np
from pyscf.solvent import ddcosmo, pcm, smd

from ._checkpoint import _molecule_fingerprint
from ._config import _checkpoint_digest

#: The PySCF solvent-model settings that change its reaction field.
_SOLVENT_SETTINGS = (
    "eps",
    "method",
    "lmax",
    "lebedev_order",
    "eta",
    "radii_table",
    "atom_radii",
    "solvent",
)


def _is_dielectric_constant(value: object) -> bool:
    """Whether ``value`` is a finite real number of at least 1."""
    return (
        isinstance(value, Real)
        and not isinstance(value, bool)
        and isfinite(value)
        and float(value) >= 1.0
    )


def _static_eps(model: ddcosmo.DDCOSMO | pcm.PCM) -> Any:
    """A solvent model's static dielectric constant; SMD takes its solvent's
    unless ``eps`` overrides it."""
    if isinstance(model, smd.SMD) and model.eps is None:
        return (model.solvent_descriptors or smd.solvent_db[model.solvent])[5]
    return model.eps


class _SolventField:
    """A copy of the caller's PySCF solvent model and the reaction field of the
    last density it was advanced with.

    Raises:
        TypeError: If ``model`` is not a ddCOSMO, ddPCM, PCM or SMD model.
        ValueError: If ``model`` was built on a different molecule or its
            dielectric constant is not a finite number of at least 1.
    """

    def __init__(self, model: ddcosmo.DDCOSMO | pcm.PCM, mol):
        if not isinstance(model, (ddcosmo.DDCOSMO, pcm.PCM)):
            raise TypeError(
                "solvent must be a PySCF ddCOSMO, ddPCM, PCM or SMD model, such "
                f"as pyscf.solvent.ddCOSMO(mol); got {type(model).__name__}."
            )
        if _molecule_fingerprint(model.mol) != _molecule_fingerprint(mol):
            raise ValueError(
                "solvent was built on a different molecule; build it on the "
                "molecule the problem was made from."
            )
        if not _is_dielectric_constant(_static_eps(model)):
            raise ValueError(
                "solvent's dielectric constant must be a finite number of at "
                f"least 1; got {_static_eps(model)!r}. Set solvent.eps, or for "
                "SMD its solvent, e.g. smd.solvent = 'water'."
            )
        # kernel caches its last field on the model; keep the caller's clean.
        self._model = model.copy()
        self.field: tuple[float, np.ndarray] | None = None

    def reset(self) -> None:
        """Drop the reaction field."""
        self.field = None

    def energy_offset(self, density: np.ndarray) -> float:
        """What to add to an energy computed with the field in the core
        Hamiltonian: trade the field's expectation for its solvation energy.
        Zero before the first field."""
        if self.field is None:
            return 0.0
        field_energy, potential = self.field
        return field_energy - float(np.einsum("ij,ji->", potential, density))

    def advance(self, density: np.ndarray) -> None:
        """Replace the reaction field with the one ``density`` induces."""
        # pyrefly: ignore[bad-argument-type]  # PySCF's PCM reuses ddCOSMO.kernel
        field_energy, potential = self._model.kernel(density)
        self.field = (float(field_energy), np.asarray(potential))

    def checkpoint_record(self) -> str:
        """Digest of the model type and the settings that fix its field."""
        return _checkpoint_digest(
            (
                type(self._model).__name__,
                {name: getattr(self._model, name, None) for name in _SOLVENT_SETTINGS},
            )
        )

    def nonequilibrium(
        self,
        densities: Callable[[], Sequence[np.ndarray]],
        optical_eps: float,
    ) -> list[float]:
        """Each density's fast-response energy, ``1/2 tr(dD v_fast[dD])``, with
        ``dD`` its change from the first density and ``v_fast`` the model's
        response at ``optical_eps``. ``densities`` is called after
        ``optical_eps`` is validated.

        Raises:
            ValueError: If ``optical_eps`` is not a finite number of at least
                1, or exceeds the solvent's static dielectric constant.
        """
        static_eps = _static_eps(self._model)
        if not _is_dielectric_constant(optical_eps):
            raise ValueError(
                "optical_eps must be a finite number of at least 1; got "
                f"{optical_eps!r}."
            )
        if optical_eps > static_eps:
            raise ValueError(
                f"optical_eps={optical_eps} exceeds the solvent's static "
                f"eps={static_eps}: the fast response cannot outweigh the full one. "
                "Use the solvent's refractive index squared."
            )
        resolved = densities()
        fast = self._model.copy()
        fast.reset()
        fast.eps = optical_eps
        fast.build()
        return [
            float(0.5 * np.einsum("ij,ji->", fast._B_dot_x(change), change))
            for change in (density - resolved[0] for density in resolved)
        ]
