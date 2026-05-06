"""Normal integration.

```python
from mbipy import normal_integration
```

!!! info "Attributes of the [mbipy.normal_integration.fourier][] and \
    [mbipy.normal_integration.least_squares][] namespaces are available."
"""

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
    "padding",
    "southwell",
]

from . import padding
from ._fourier import arnison, dct_poisson, dst_poisson, frankot, kottler
from ._fourier._utils import FFTMethod
from .least_squares import Li, Southwell, li, southwell
