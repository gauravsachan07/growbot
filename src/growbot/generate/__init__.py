"""Grounded generation: prompt policy + LLM call.

Generation sees only retrieved context. `source_url` and `last_updated` come
from chunk metadata, never from the model (architecture §6.4).
"""
