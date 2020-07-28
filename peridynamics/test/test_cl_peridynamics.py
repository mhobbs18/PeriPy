"""Tests for the OpenCL kernels."""
from .conftest import context_available
from ..cl import get_context, pad, kernel_source
from ..cl.utilities import DOUBLE_FP_SUPPORT
import numpy as np
from peridynamics.neighbour_list import (create_neighbour_list_cython,
                                         create_neighbour_list_cl)
import pyopencl as cl
from pyopencl import mem_flags as mf
import pytest


@pytest.fixture(scope="module")
def context():
    """Create a context using the default platform, prefer GPU."""
    return get_context()


@context_available
@pytest.fixture(scope="module")
def queue(context):
    """Create a CL command queue."""
    return cl.CommandQueue(context)


@context_available
@pytest.fixture(scope="module")
def program(context):
    """Create a program object from the kernel source."""
    return cl.Program(context, kernel_source).build()


@context_available
def test_damage(context, queue, program):
    """Test damage kernel."""
    n_neigh = np.array([5, 5, 3, 0, 4, 5, 8, 3, 2, 1], dtype=np.int32)
    family = np.array([10, 5, 5, 1, 5, 7, 10, 3, 3, 4], dtype=np.int32)
    damage = np.empty(n_neigh.shape, dtype=np.float64)

    # Create buffers
    n_neigh_d = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR,
                          hostbuf=n_neigh)
    family_d = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR,
                         hostbuf=family)
    damage_d = cl.Buffer(context, mf.WRITE_ONLY, damage.nbytes)

    # Call kernel
    damage_kernel = program.damage
    damage_kernel(queue, family.shape, None, n_neigh_d, family_d, damage_d)
    cl.enqueue_copy(queue, damage, damage_d)

    damage_expected = (family - n_neigh) / family
    assert np.allclose(damage, damage_expected)


class TestForce():
    """Test force calculation."""

    @context_available
    def test_initial_force(self, context, queue, program):
        """Ensure forces are zero when there is no displacement."""
        r0 = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            ], dtype=np.float64)
        horizon = 1.1
        volume = np.ones(5, dtype=np.float64)
        bond_stiffness = 1.0
        max_neigh = 3
        nlist, n_neigh = create_neighbour_list(r0, horizon, max_neigh)

        force_expected = np.zeros((5, 3), dtype=np.float64)
        force_actual = np.empty_like(force_expected)

        # Create buffers
        r_d = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR,
                        hostbuf=r0)
        r0_d = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR,
                         hostbuf=r0)
        nlist_d = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR,
                            hostbuf=nlist)
        n_neigh_d = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR,
                              hostbuf=n_neigh)
        volume_d = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR,
                             hostbuf=volume)
        force_d = cl.Buffer(context, mf.WRITE_ONLY, force_expected.nbytes)

        # Call kernel
        bond_force = program.bond_force
        bond_force(queue, n_neigh.shape, None, r_d, r0_d, nlist_d, n_neigh_d,
                   np.int32(max_neigh), volume_d, np.float64(bond_stiffness),
                   force_d)
        cl.enqueue_copy(queue, force_actual, force_d)

        assert np.allclose(force_actual, force_expected)

    @context_available
    def test_force(self, context, queue, program):
        """Ensure forces are in the correct direction using a minimal model."""
        r0 = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            ], dtype=np.float64)
        horizon = 1.01
        elastic_modulus = 0.05
        bond_stiffness = 18.0 * elastic_modulus / (np.pi * horizon**4)
        max_neigh = 3
        volume = np.full(3, 0.16666667, dtype=np.float64)
        nlist, n_neigh = create_neighbour_list(r0, horizon, max_neigh)

        # Displace particles, but do not update neighbour list
        r = r0 + np.array([
            [0.0, 0.0, 0.0],
            [0.05, 0.0, 0.0],
            [0.05, 0.05, 0.0]
            ], dtype=np.float64)

        force_value = 0.00229417
        force_expected = np.array([
            [force_value, 0., 0.],
            [-force_value, force_value, 0.],
            [0., -force_value, 0.]
            ])
        force_actual = np.empty_like(force_expected)

        # Create buffers
        r_d = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR,
                        hostbuf=r)
        r0_d = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR,
                         hostbuf=r0)
        nlist_d = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR,
                            hostbuf=nlist)
        n_neigh_d = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR,
                              hostbuf=n_neigh)
        volume_d = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR,
                             hostbuf=volume)
        force_d = cl.Buffer(context, mf.WRITE_ONLY, force_expected.nbytes)

        # Call kernel
        bond_force = program.bond_force
        bond_force(queue, n_neigh.shape, None, r_d, r0_d, nlist_d, n_neigh_d,
                   np.int32(max_neigh), volume_d, np.float64(bond_stiffness),
                   force_d)
        cl.enqueue_copy(queue, force_actual, force_d)

        assert np.allclose(force_actual, force_expected)


@pytest.fixture
def basic_model_2d(data_path):
    """Create a basic 2D model object."""
    mesh_file = data_path / "example_mesh.vtk"
    model = ModelCL(mesh_file, horizon=0.1, critical_stretch=0.05,
                    bond_stiffness=18.0 * 0.05 / (np.pi * 0.1**4))
    return model


@pytest.fixture()
def basic_model_3d(data_path):
    """Create a basic 3D model object."""
    mesh_file = data_path / "example_mesh_3d.vtk"
    model = ModelCL(mesh_file, horizon=0.1, critical_stretch=0.05,
                    bond_stiffness=18.0 * 0.05 / (np.pi * 0.1**4),
                    dimensions=3)
    return model


def test_no_context(data_path, monkeypatch):
    """Test raising error when no suitable device is found."""
    from .. import model_cl

    # Mock the get_context function to return None as it would if no suitable
    # device is found.
    def return_none():
        return None
    monkeypatch.setattr(model_cl, "get_context", return_none)

    mesh_file = data_path / "example_mesh.vtk"
    with pytest.raises(ContextError) as exception:
        ModelCL(mesh_file, horizon=0.1, critical_stretch=0.05,
                bond_stiffness=18.0 * 0.05 / (np.pi * 0.1**4))

        assert "No suitable context was found." in exception.value


@context_available
def test_custom_context(data_path):
    """Test constructing a ModelCL object using the context argument."""
    mesh_file = data_path / "example_mesh_3d.vtk"
    context = get_context()
    model = ModelCL(mesh_file, horizon=0.1, critical_stretch=0.05,
                    bond_stiffness=18.0 * 0.05 / (np.pi * 0.1**4),
                    dimensions=3, context=context)

    assert model.context is context


def test_invalid_custom_context(data_path):
    """Test constructing a ModelCL object using the context argument."""
    mesh_file = data_path / "example_mesh_3d.vtk"
    with pytest.raises(TypeError) as exception:
        ModelCL(mesh_file, horizon=0.1, critical_stretch=0.05,
                bond_stiffness=18.0 * 0.05 / (np.pi * 0.1**4),
                dimensions=3, context=5)

        assert "context must be a pyopencl Context object" in exception.value


def test_initial_damage_2d(basic_model_2d):
    """Ensure initial damage is zero."""
    model = basic_model_2d
    context = model.context
    queue = model.queue
    nlist, n_neigh = model.initial_connectivity

    n_neigh_d = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR,
                          hostbuf=n_neigh)
    family_d = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR,
                         hostbuf=model.family)
    damage = np.empty(n_neigh.shape, dtype=np.float64)
    damage_d = cl.Buffer(context, mf.WRITE_ONLY, damage.nbytes)

    model._damage(n_neigh_d, family_d, damage_d)
    cl.enqueue_copy(queue, damage, damage_d)

    assert np.all(damage == 0)


def test_initial_damage_3d(basic_model_3d):
    """Ensure initial damage is zero."""
    model = basic_model_3d
    context = model.context
    queue = model.queue
    nlist, n_neigh = model.initial_connectivity

    n_neigh_d = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR,
                          hostbuf=n_neigh)
    family_d = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR,
                         hostbuf=model.family)
    damage = np.empty(n_neigh.shape, dtype=np.float64)
    damage_d = cl.Buffer(context, mf.WRITE_ONLY, damage.nbytes)

    model._damage(n_neigh_d, family_d, damage_d)
    cl.enqueue_copy(queue, damage, damage_d)

    assert np.all(damage == 0)