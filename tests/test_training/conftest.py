"""Training test package config.

Guardrail: do not globally inject stubs into ``sys.modules`` at import time.
That pattern leaks fake modules into unrelated tests and breaks collection.
"""
