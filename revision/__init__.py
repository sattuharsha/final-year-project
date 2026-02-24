"""
MODULE 6 — Selective Revision.
Rewrite only hallucinated spans; preserve supported content; evidence-conditioned prompt.
"""
from revision.selective_revision import selective_revise

__all__ = ["selective_revise"]
