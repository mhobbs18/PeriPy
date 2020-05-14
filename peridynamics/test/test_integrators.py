"""Tests for the integrators module."""
from ..integrators import Euler, Integrator
import numpy as np
import pytest


class TestIntegrator:
    """ABC class tests."""

    def test_not_implemented_error(self):
        """Ensure the ABC cannot be instantiated."""
        with pytest.raises(TypeError):
            Integrator()


class TestEuler:
    """Euler integrator tests."""

    def test_basic_integration(self):
        """Test integration."""
        integrator = Euler(1)
        u = np.zeros((2, 3))
        f = np.array([
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0]
            ])

        u = integrator(u, f)

        assert np.all(u == f)

    def test_basic_integration2(self):
        """Test integration."""
        integrator = Euler(2.0)
        u = np.full((2, 3), 1.0)
        f = np.array([
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0]
            ])

        u = integrator(u, f)

        assert np.all(u == 2.0*f+1.0)

    def test_basic_integration3(self):
        """Test integration with dampening."""
        integrator = Euler(2.0, dampening=0.7)
        u = np.full((2, 3), 2.5)
        f = np.array([
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0]
            ])

        u = integrator(u, f)

        assert np.all(u == 2.0*0.7*f+2.5)
