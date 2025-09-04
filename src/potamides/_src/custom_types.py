"""Custom types.

Glossary:

- "Sz": sizes of each dimension -- aka the shape, which doesn't have as good an
  abbreviation. It is proceeded by a description of the shape.
- "Like": Prefix denoting that it is compatible with some non-array inputs, e.g. a float.
- "Q": `unxt.Quantity`
- "F": features.

"""

__all__: tuple[str, ...] = ()

from typing import TypeAlias

import unxt as u
from jaxtyping import Array, Bool, Real

Sz0: TypeAlias = Real[Array, ""]
LikeSz0: TypeAlias = Real[Array, ""] | float | int
LikeQorVSz0: TypeAlias = Real[u.AbstractQuantity, ""] | LikeSz0
Sz2: TypeAlias = Real[Array, "2"]
SzF: TypeAlias = Real[Array, "F"]
SzN: TypeAlias = Real[Array, "N"]
SzN2: TypeAlias = Real[Array, "N 2"]
SzN3: TypeAlias = Real[Array, "N 3"]
SzNF: TypeAlias = Real[Array, "N F"]
QuSzN3: TypeAlias = Real[u.AbstractQuantity, "N 3"]
QorVSzN3: TypeAlias = SzN3 | QuSzN3

SzData: TypeAlias = Real[Array, "data"]
SzData2: TypeAlias = Real[Array, "data 2"]

SzGamma: TypeAlias = Real[Array, "gamma"]
SzGamma2: TypeAlias = Real[Array, "gamma 2"]
SzGammaF: TypeAlias = Real[Array, "gamma F"]

BoolSzGamma: TypeAlias = Bool[Array, "gamma"]
