"""Normal integration.

```python
from mbipy import normal_integration
```

!!! info "Attributes of the [mbipy.normal_integration.fourier][] and \
    [mbipy.normal_integration.least_squares][] namespaces are available."
"""

from __future__ import annotations

__all__ = [
    "FFTMethod",
    "Li",
    "Southwell",
    "arnison",
    "dct_poisson",
    "dst_poisson",
    "frankot",
    "kottler",
    "li",
    "southwell",
]

from .fourier import arnison, dct_poisson, dst_poisson, frankot, kottler
from .fourier._utils import FFTMethod
from .least_squares import Li, Southwell, li, southwell
