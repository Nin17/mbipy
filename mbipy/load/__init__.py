"""
Convenience functions for reading images.

Requires [fabio][]

```python
from mbipy import load
```
"""

__all__ = ["load_data", "load_paths", "load_stack"]

from ._load import load_data, load_paths, load_stack
