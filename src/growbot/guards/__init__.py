"""Deterministic guards that run before retrieval.

PII -> advice -> returns comparison, in that order, so a message that contains
both a PAN and a "should I buy" is never stored or answered (architecture §8).
"""
