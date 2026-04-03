"""Compatibility helpers for legacy third-party dependencies."""

import numpy as np


def ensure_chumpy_numpy_compat():
    """Restore NumPy aliases removed in 2.x that chumpy still imports."""

    legacy_aliases = {
        "bool": np.bool_,
        "int": np.int_,
        "float": np.float64,
        "complex": np.complex128,
        "object": np.object_,
        "unicode": np.str_,
        "str": np.str_,
    }
    for alias, value in legacy_aliases.items():
        if alias not in np.__dict__:
            setattr(np, alias, value)
