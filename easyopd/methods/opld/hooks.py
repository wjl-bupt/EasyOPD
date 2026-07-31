"""Hook adapters for the ``opld`` (listwise) method.

OPLD deliberately implements **no** hooks.

The listwise group softmax needs all K rollouts of a prompt simultaneously.
``LossHook`` is a per-microbatch surface, and the K candidates of a ``uid`` are
not guaranteed to land in the same microbatch or on the same rank -- computing
the group softmax there would silently normalize over a truncated subset and
produce wrong (but plausible-looking) advantages.

So the whole method lives on the driver instead, as an advantage estimator:
see ``advantage_estimator.py``. This module is intentionally left without hook
classes; ``HookDispatcher._build_hooks`` looks for ``OPLD*Hook`` names here and
correctly finds none.
"""

from __future__ import annotations
