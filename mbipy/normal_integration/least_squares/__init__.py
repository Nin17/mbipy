"""Normal integration using sparse least squares methods.

```python
from mbipy.normal_integration import least_squares
```

!!! info "All attributes are also available in the main \
    [mbipy.normal_integration][] namespace."
"""

__all__ = ["Li", "Southwell", "li", "southwell"]

from ._li import Li, li
from ._southwell import Southwell, southwell
