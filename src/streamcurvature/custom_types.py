"""Curvature analysis functions."""

__all__: list[str] = []

from typing import TypeAlias

import unxt as u
from jaxtyping import Array, Real

Sz0: TypeAlias = Real[Array, ""]
LikeSz0: TypeAlias = Real[Array, ""] | float | int
LikeQorVSz0: TypeAlias = Real[u.Quantity, ""] | LikeSz0
Sz2: TypeAlias = Real[Array, "2"]
SzN: TypeAlias = Real[Array, "N"]
SzN2: TypeAlias = Real[Array, "N 2"]
SzN3: TypeAlias = Real[Array, "N 3"]
QuSzN3: TypeAlias = Real[u.AbstractQuantity, "N 3"]
QorVSzN3: TypeAlias = SzN3 | QuSzN3

SzData: TypeAlias = Real[Array, "data"]
SzData2: TypeAlias = Real[Array, "data 2"]
