"""llmperfscope: whole-lifecycle profiler for LLM inference engines."""

__version__ = "0.1.0"

from .markers import mark  # noqa: E402,F401
from .api import Profile, TokenTimer, mark_tokens, phase  # noqa: E402,F401
