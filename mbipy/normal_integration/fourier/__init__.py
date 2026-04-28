"""Normal integration using Fourier methods.

```python
from mbipy.normal_integration import fourier
```

!!! info "All attributes are also available in the main \
    [mbipy.normal_integration][] namespace."
"""

__all__ = ["arnison", "dct_poisson", "dst_poisson", "frankot", "kottler"]

from ._arnison import arnison
from ._dct import dct_poisson
from ._dst import dst_poisson
from ._frankot import frankot
from ._kottler import kottler
