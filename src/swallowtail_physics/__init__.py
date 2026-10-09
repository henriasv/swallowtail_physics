"""Locate phase equilibria from the swallowtail self-intersection of G(P) data."""

from .intersect import Coexistence, find_coexistence, find_equilibrium

__all__ = ["Coexistence", "find_coexistence", "find_equilibrium"]
__version__ = "0.2.0"
