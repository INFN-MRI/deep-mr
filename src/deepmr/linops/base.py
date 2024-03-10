"""Base linear operator."""

__all__ = ["Linop"]

import deepinv as dinv


class Linop(dinv.physics.LinearPhysics):
    """
    Abstraction class for Linear operators.

    This is an alias for ``deepinv.physics.LinearPhysics``,
    but provides a convenient method for ``A_adjoint`` (i.e., ``Linop.H``)
    """

    @property
    def H(self):
        A = lambda x: self.A_adjoint(x)
        A_adjoint = lambda x: self.A(x)
        noise = self.noise_model
        sensor = self.sensor_model
        return Linop(
            A=A,
            A_adjoint=A_adjoint,
            noise_model=noise,
            sensor_model=sensor,
            max_iter=self.max_iter,
            tol=self.tol,
        )

    def __add__(self, other):
        tmp = super().__add__(other)
        return Linop(
            A=tmp.A,
            A_adjoint=tmp.A_adjoint,
            noise_model=tmp.noise_model,
            sensor_model=tmp.sensor_model,
            max_iter=tmp.max_iter,
            tol=tmp.tol,
        )

    def __mul__(self, other):
        tmp = super().__mul__(other)
        return Linop(
            A=tmp.A,
            A_adjoint=tmp.A_adjoint,
            noise_model=tmp.noise_model,
            sensor_model=tmp.sensor_model,
            max_iter=tmp.max_iter,
            tol=tmp.tol,
        )
