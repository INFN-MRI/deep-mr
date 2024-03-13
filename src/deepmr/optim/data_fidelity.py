"""Batched data fidelity terms."""

__all__ = ["L2"]

import deepinv as dinv

from ..linops import NormalLinop


class L2(dinv.optim.L2):
    """Wrapper to deepinv.optim.L2 to support NormalLinop."""

    def grad(self, x, y, physics, *args, **kwargs):
        r"""
        Calculates the gradient of the data fidelity term :math:`\datafidname` at :math:`x`.

        Arguments
        ---------
        x : torch.Tensor
            Variable :math:`x` at which the gradient is computed.
        y : torch.Tensor
            Data :math:`y`.
        physics : deepinv.physics.Physics physics
            Physics model.

        Returns
        -------
        output : torch.Tensor
            Gradient :math:`\nabla_x\datafid{x}{y}`, computed in :math:`x`.

        """
        if isinstance(physics, NormalLinop):
            return self.grad_d(physics.A(x), y, *args, **kwargs)

        return physics.A_adjoint(self.grad_d(physics.A(x), y, *args, **kwargs))
