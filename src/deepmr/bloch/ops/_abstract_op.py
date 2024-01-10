"""Fundamental EPG operators. Can be used to simulate multi-pulse sequences."""

__all__ = ["Operator", "CompositeOperator", "Identity"]

import uuid


class Operator:
    """
    Base class of an operator acting on an epg object. Application of the operator will alter the EPG object.

    All derived operators should make sure to return the epg on application to allow operator chaining.

    Operators should encapsulate a abstract modification of the spin sates

    For physical simulation there shall be another layer that ties the operators together...
    """

    def __init__(self, name="", *args, **kwargs):  # noqa
        if name:
            self.name = name  # Optional name for operators
        else:
            # Unique names for easier possibility of serialisation
            self.name = str(uuid.uuid4())

    def apply(self, epg):  # noqa
        return epg

    def __mul__(self, other):  # noqa
        if isinstance(other, Operator):
            return CompositeOperator(self, other)
        elif hasattr(other, "state"):
            return self.apply(other)
        else:
            raise NotImplementedError("Can not apply operator to non-EPGs")

    def __call__(self, *args, **kwargs):  # noqa
        return self.apply(*args, **kwargs)


class CompositeOperator(Operator):
    """Composite operator that contains several operators."""

    def __init__(self, *args, **kwargs):  # noqa
        super().__init__(**kwargs)
        self._operators = []

        for op in args:
            self.append(op)

    def __getitem__(self, name):  # noqa
        for op in self._operators:
            if op.name == name:
                return op
        return None

    def prepend(self, operator):  # noqa
        self._operators.insert(0, operator)
        return self

    def append(self, operator):  # noqa
        self._operators.append(operator)
        return self

    def apply(self, epg):
        """
        Apply the composite operator to an EPG by consecutive application of the contained operators.

        Args:
            epg (EPGMatrix): epg matrix to be operated on.

        Returns:
            epg after operator application.
        """
        epg_dash = epg

        for op in reversed(self._operators):
            epg_dash = op.apply(epg_dash)

        return epg_dash

    def __mul__(self, other):  # noqa
        if hasattr(other, "state"):
            return self.apply(other)
        elif isinstance(other, Operator):
            return self.append(other)
        else:
            raise NotImplementedError("Object can not be added to composite operator")

    def __rmul__(self, other):  # noqa
        if isinstance(other, Operator):
            self.prepend(other)
        else:
            raise NotImplementedError("No viable multiplication for composite operator")


class Identity(Operator):
    """Dummy operator."""

    pass
