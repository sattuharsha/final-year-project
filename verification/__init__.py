"""
Verification modules: token-level faithfulness (MODULE 4) and numerical fact verification (MODULE 5).
"""
from verification.token_faithfulness import (
    compute_token_faithfulness,
    split_into_spans,
)
from verification.numeric_verification import (
    extract_numeric_entities,
    verify_numerics_against_evidence,
)

__all__ = [
    "compute_token_faithfulness",
    "split_into_spans",
    "extract_numeric_entities",
    "verify_numerics_against_evidence",
]
