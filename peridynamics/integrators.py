"""Integrators."""
from abc import ABC, abstractmethod
from peridynamics.probabilistic import noise
import pyopencl as cl
import numpy as np
import sys
import pathlib
from peridynamics.post_processing import vtk
sys.path.insert(1, pathlib.Path(__file__).parent.absolute() / 'peridynamics/kernels')

class Integrator(ABC):
    """
    Base class for integrators.

    All integrators must define a call method which performs one
    integration step and returns the updated displacements.
    """

    @abstractmethod
    def __call__(self):
        """
        Conduct one iteraction of the integrator.

        This method should be implemennted in every concrete integrator.
        """
class Euler(Integrator):
    r"""
    Euler integrator.

    The Euler method is a first-order numerical integration method. The
    integration is given by,

    .. math::
        u(t + \delta t) = u(t) + \delta t f(t) d

    where :math:`u(t)` is the displacement at time :math:`t`, :math:`f(t)` is
    the force at time :math:`t`, :math:`\delta t` is the time step and
    :math:`d` is a dampening factor.
    """

    def __init__(self, dt, dampening=1.0):
        """
        Create an :class:`Euler` integrator object.

        :arg float dt: The integration time step.
        :arg float dampening: The dampening factor. The default is 1.0

        :returns: A :class:`Euler` object
        """
        self.dt = dt
        self.dampening = dampening

    def __call__(self, u, f):
        """
        Conduct one iteration of the integrator.

        :arg u: A (`nnodes`, 3) array containing the displacements of all
            nodes.
        :type u: :class:`numpy.ndarray`
        :arg f: A (`nnodes`, 3) array containing the components of the force
            acting on each node.
        :type f: :class:`numpy.ndarray`

        :returns: The new displacements after integration.
        :rtype: :class:`numpy.ndarray`
        """
        return u + self.dt * f * self.dampening

class EulerCromerOpenCL(Integrator):
    r"""
    Dynamic Euler integrator using OpenCL kernels.

    The Euler-Cromer method is a first-order, dynamic (acceleration term is not neglected), numerical integration method. The
    integration is given by,

    .. math::
        u(t + \delta t) = u(t) + \delta t v(t)
        v(t + \delta t) = v(t) + \delta t a(t)

    where :math:`u(t)` is the displacement at time :math:`t`, :math:`v(t)` is
    the velocity at time :math:`t`, :math:`a(t)` is the acceleration at time :math:`t`,
    :math:`\delta t` is the time step.
    """
    def __init__(self, model):
        """ Initialise the integration scheme for Euler Cromer
        """
        def output_device_info(device_id):
            sys.stdout.write("Device is ")
            sys.stdout.write(device_id.name)
            if device_id.type == cl.device_type.GPU:
                sys.stdout.write("GPU from ")
            elif device_id.type == cl.device_type.CPU:
                sys.stdout.write("CPU from ")
            else:
                sys.stdout.write("non CPU of GPU processor from ")
            sys.stdout.write(device_id.vendor)
            sys.stdout.write(" with a max of ")
            sys.stdout.write(str(device_id.max_compute_units))
            sys.stdout.write(" compute units\n")
            sys.stdout.flush()

        # Initializing OpenCL
        self.context = cl.create_some_context()
        self.queue = cl.CommandQueue(self.context)   

        # Print out device info
        output_device_info(self.context.devices[0])

        # Build the OpenCL program from file
        kernelsource = open(pathlib.Path(__file__).parent.absolute() / "kernels/opencl_euler_cromer.cl").read()

        # JIT compiler's command line arguments
        SEP = " "

        options_string = (
            "-cl-fast-relaxed-math" + SEP
            + "-DPD_DPN_NODE_NO=" + str(model.degrees_freedom * model.nnodes) + SEP
            + "-DPD_NODE_NO=" + str(model.nnodes) + SEP
            + "-DMAX_HORIZON_LENGTH=" + str(model.max_horizon_length) + SEP
            + "-DPD_DT=" + str(model.dt) + SEP
            + "-DPD_RHO=" + str(model.density) + SEP
            + "-DPD_ETA=" + str(model.damping) + SEP)

        # Build the programs
        program = cl.Program(self.context, kernelsource).build([options_string])
        self.cl_kernel_time_marching_1 = program.TimeMarching1
        self.cl_kernel_time_marching_2 = program.TimeMarching2
        self.cl_kernel_time_marching_3 = program.TimeMarching3
        self.cl_kernel_check_bonds = program.CheckBonds
        self.cl_kernel_calculate_damage = program.CalculateDamage

        # Set initial values in host memory
        # horizons and horizons lengths
        self.h_horizons = model.horizons
        self.h_horizons_lengths = model.horizons_lengths
        # Nodal coordinates
        self.h_coords = np.ascontiguousarray(model.coords, dtype=np.float64)
        # Displacement boundary conditions types and delta values
        self.h_bc_types = model.bc_types
        self.h_bc_values = model.bc_values
        # Force boundary conditions types and values
        self.h_force_bc_types = model.force_bc_types
        self.h_force_bc_values = model.force_bc_values

        # Nodal volumes
        self.h_vols = model.V
        # Bond stiffnesses
        self.h_bond_stiffness = np.ascontiguousarray(model.bond_stiffness, dtype=np.float64)
        self.h_bond_critical_stretch = np.ascontiguousarray(model.bond_critical_stretch, dtype=np.float64)

        # Displacement
        self.h_un = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)
        # Velocity
        self.h_udn = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)
        # Acceleration
        self.h_uddn = np.empty((model.nnodes, model.degrees_freedom), dtype = np.float64)

        # Damage vector
        self.h_damage = np.empty(model.nnodes).astype(np.float64)

        # For measuring tip displacemens (host memory only)
        self.h_tip_types = model.tip_types
        print(np.max(self.h_tip_types), 'max tiptypes')

        if model.v == True:

            # Print the dtypes
            print("horizons", self.h_horizons.dtype)
            print("horizons_length", self.h_horizons_lengths.dtype)
            print("force_bc_types", self.h_bc_types.dtype)
            print("force_bc_values", self.h_bc_values.dtype)
            print("bc_types", self.h_bc_types.dtype)
            print("bc_values", self.h_bc_values.dtype)
            print("coords", self.h_coords.dtype)
            print("vols", self.h_vols.dtype)
            print("un", self.h_un.dtype)
            print("udn", self.h_udn.dtype)
            print("damage", self.h_damage.dtype)
            print("stiffness", self.h_bond_stiffness.dtype)
            print("stretch", self.h_bond_critical_stretch.dtype)

        # Build OpenCL data structures
        # Read only
        self.d_coords = cl.Buffer(self.context,
                             cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                             hostbuf=self.h_coords)
        self.d_bc_types = cl.Buffer(self.context,
                              cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                              hostbuf=self.h_bc_types)
        self.d_bc_values = cl.Buffer(self.context,
                               cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                               hostbuf=self.h_bc_values)
        self.d_force_bc_types = cl.Buffer(self.context,
                              cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                              hostbuf=self.h_force_bc_types)
        self.d_force_bc_values = cl.Buffer(self.context,
                               cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                               hostbuf=self.h_force_bc_values)
        self.d_vols = cl.Buffer(self.context,
                           cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                           hostbuf=self.h_vols)
        self.d_bond_stiffness = cl.Buffer(self.context,
                           cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                           hostbuf=self.h_bond_stiffness)
        self.d_bond_critical_stretch = cl.Buffer(self.context,
                           cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                           hostbuf=self.h_bond_critical_stretch)
        self.d_horizons_lengths = cl.Buffer(
                self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                hostbuf=self.h_horizons_lengths)

        # Read and write
        self.d_horizons = cl.Buffer(
                self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
                hostbuf=self.h_horizons)
        self.d_un = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_un.nbytes)
        self.d_udn = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_udn.nbytes)
        self.d_uddn = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_uddn.nbytes)
        # Write only
        self.d_damage = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, self.h_damage.nbytes)
        # Initialize kernel parameters
        self.cl_kernel_time_marching_1.set_scalar_arg_dtypes(
            [None, None, None, None])
        self.cl_kernel_time_marching_2.set_scalar_arg_dtypes(
            [None, None, None, None, None, None, None, None, None])
        self.cl_kernel_time_marching_3.set_scalar_arg_dtypes([None, None])
        self.cl_kernel_check_bonds.set_scalar_arg_dtypes([None, None, None, None])
        self.cl_kernel_calculate_damage.set_scalar_arg_dtypes([None, None, None])
        return None

    def __call__(self):
        """
        Conduct one iteration of the integrator.

        :arg u: A (`nnodes`, 3) array containing the displacements of all
            nodes.
        :type u: :class:`numpy.ndarray`
        :arg f: A (`nnodes`, 3) array containing the components of the force
            acting on each node.
        :type f: :class:`numpy.ndarray`

        :returns: The new displacements after integration.
        :rtype: :class:`numpy.ndarray`
        """

    def runtime(self, model):
        """ Run time integration for Euler Cromer scheme
        """
        # Time marching Part 1
        self.cl_kernel_time_marching_1(self.queue, (model.degrees_freedom * model.nnodes,),
                                  None, self.d_udn, self.d_un, self.d_bc_types,
                                  self.d_bc_values)

        # Time marching Part 2
        self.cl_kernel_time_marching_2(self.queue, (model.nnodes,), None, self.d_uddn, self.d_udn,
                                  self.d_un, self.d_vols, self.d_horizons, self.d_coords, self.d_bond_stiffness, self.d_force_bc_types, self.d_force_bc_values)

        # Time marching Part 3
        self.cl_kernel_time_marching_3(self.queue, (model.degrees_freedom * model.nnodes,),
                                  None, self.d_udn, self.d_uddn)

        # Check for broken bonds
        self.cl_kernel_check_bonds(self.queue,
                              (model.nnodes, model.max_horizon_length),
                              None, self.d_horizons, self.d_un, self.d_coords, self.d_bond_critical_stretch)
    def write(self, model, t, sample):
        """ Write a mesh file for the current timestep
        """
        self.cl_kernel_calculate_damage(self.queue, (model.nnodes,), None, 
                                           self.d_damage, self.d_horizons,
                                           self.d_horizons_lengths)
        cl.enqueue_copy(self.queue, self.h_damage, self.d_damage)
        cl.enqueue_copy(self.queue, self.h_un, self.d_un)
        # TODO define a failure criterion, idea: rate of change of damage goes to 0 after it has started increasing
        tip_displacement = 0
        tmp = 0
        for i in range(model.nnodes):
            if self.h_tip_types[i] == 1:
                tmp +=1
                tip_displacement += self.h_un[i][2]
        if tmp != 0:
            tip_displacement /= tmp
        else:
            tip_displacement = None
        vtk.write("output/U_"+"sample" + str(sample) +"t"+str(t) + ".vtk", "Solution time step = "+str(t),
                  model.coords, self.h_damage, self.h_un)
        #vtk.writeDamage("output/damage_" + str(t)+ "sample" + str(sample) + ".vtk", "Title", self.h_damage)
        return self.h_damage, tip_displacement

    def incrementLoad(self, model, load_scale):
        if model.num_force_bc_nodes != 0:
            tmp = -1. * model.max_reaction * load_scale / (model.num_force_bc_nodes)
            # update the host force_bc_values
            self.h_force_bc_values = tmp * np.ones((model.nnodes, model.degrees_freedom), dtype=np.float64)
            #print(h_force_bc_values)
            # update the GPU force_bc_values
            self.d_force_bc_values = cl.Buffer(self.context,
                               cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                               hostbuf=self.h_force_bc_values)
class EulerOpenCL(Integrator):
    r"""
    Static Euler integrator for quasi-static loading, using OpenCL kernels.

    The Euler method is a first-order numerical integration method. The
    integration is given by,

    .. math::
        u(t + \delta t) = u(t) + \delta t f(t) d

    where :math:`u(t)` is the displacement at time :math:`t`, :math:`f(t)` is
    the force at time :math:`t`, :math:`\delta t` is the time step and
    :math:`d` is a dampening factor.
    """
    def __init__(self, model):
        """ Initialise the integration scheme
        """
        def output_device_info(device_id):
            sys.stdout.write("Device is ")
            sys.stdout.write(device_id.name)
            if device_id.type == cl.device_type.GPU:
                sys.stdout.write("GPU from ")
            elif device_id.type == cl.device_type.CPU:
                sys.stdout.write("CPU from ")
            else:
                sys.stdout.write("non CPU of GPU processor from ")
            sys.stdout.write(device_id.vendor)
            sys.stdout.write(" with a max of ")
            sys.stdout.write(str(device_id.max_compute_units))
            sys.stdout.write(" compute units\n")
            sys.stdout.flush()

        # Initializing OpenCL
        self.context = cl.create_some_context()
        self.queue = cl.CommandQueue(self.context)   

        # Print out device info
        output_device_info(self.context.devices[0])

        # Build the OpenCL program from file
        kernelsource = open(pathlib.Path(__file__).parent.absolute() / "kernels/opencl_euler.cl").read()
        SEP = " "

        options_string = (
            "-cl-fast-relaxed-math" + SEP
            + "-DPD_DPN_NODE_NO=" + str(model.degrees_freedom * model.nnodes) + SEP
            + "-DPD_NODE_NO=" + str(model.nnodes) + SEP
            + "-DMAX_HORIZON_LENGTH=" + str(model.max_horizon_length) + SEP
            + "-DPD_DT=" + str(model.dt) + SEP)

        from pathlib import Path
        print(Path.cwd())
        program = cl.Program(self.context, kernelsource).build([options_string])
        self.cl_kernel_time_marching_1 = program.TimeMarching1
        self.cl_kernel_time_marching_2 = program.TimeMarching2
        self.cl_kernel_check_bonds = program.CheckBonds
        self.cl_kernel_calculate_damage = program.CalculateDamage

        # Set initial values in host memory

        # horizons and horizons lengths
        self.h_horizons = model.horizons
        self.h_horizons_lengths = model.horizons_lengths
        print(self.h_horizons_lengths)
        print(self.h_horizons)
        print("shape horizons lengths", self.h_horizons_lengths.shape)
        print("shape horizons lengths", self.h_horizons.shape)
        print(self.h_horizons_lengths.dtype, "dtype")

        # Nodal coordinates
        self.h_coords = np.ascontiguousarray(model.coords, dtype=np.float64)

        # Displacement boundary conditions types and delta values
        self.h_bc_types = model.bc_types
        self.h_bc_values = model.bc_values

        self.h_tip_types = model.tip_types

        # Force boundary conditions types and values
        self.h_force_bc_types = model.force_bc_types
        self.h_force_bc_values = model.force_bc_values

        # Nodal volumes
        self.h_vols = model.V

        # Bond stiffnesses
        self.h_bond_stiffness =  np.ascontiguousarray(model.bond_stiffness, dtype=np.float64)
        self.h_bond_critical_stretch = np.ascontiguousarray(model.bond_critical_stretch, dtype=np.float64)

        # Displacements
        self.h_un = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)

        # Forces
        self.h_udn1 = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)

        # Damage vector
        self.h_damage = np.empty(model.nnodes).astype(np.float64)

        if model.v == True:
            # Print the dtypes
            print("horizons", self.h_horizons.dtype)
            print("horizons_length", self.h_horizons_lengths.dtype)
            print("force_bc_types", self.h_bc_types.dtype)
            print("force_bc_values", self.h_bc_values.dtype)
            print("bc_types", self.h_bc_types.dtype)
            print("bc_values", self.h_bc_values.dtype)
            print("coords", self.h_coords.dtype)
            print("vols", self.h_vols.dtype)
            print("un", self.h_un.dtype)
            print("udn1", self.h_udn1.dtype)
            print("damage", self.h_damage.dtype)

        # Build OpenCL data structures

        # Read only
        self.d_coords = cl.Buffer(self.context,
                             cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                             hostbuf=self.h_coords)
        self.d_bc_types = cl.Buffer(self.context,
                              cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                              hostbuf=self.h_bc_types)
        self.d_bc_values = cl.Buffer(self.context,
                               cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                               hostbuf=self.h_bc_values)
        self.d_force_bc_types = cl.Buffer(self.context,
                              cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                              hostbuf=self.h_force_bc_types)
        self.d_force_bc_values = cl.Buffer(self.context,
                               cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                               hostbuf=self.h_force_bc_values)
        self.d_vols = cl.Buffer(self.context,
                           cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                           hostbuf=self.h_vols)
        self.d_bond_stiffness = cl.Buffer(self.context,
                           cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                           hostbuf=self.h_bond_stiffness)
        self.d_bond_critical_stretch = cl.Buffer(self.context,
                           cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                           hostbuf=self.h_bond_critical_stretch)
        self.d_horizons_lengths = cl.Buffer(
                self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                hostbuf=self.h_horizons_lengths)

        # Read and write
        self.d_horizons = cl.Buffer(
                self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
                hostbuf=self.h_horizons)
        self.d_un = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_un.nbytes)
        self.d_udn1 = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_udn1.nbytes)

        # Write only
        self.d_damage = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, self.h_damage.nbytes)
        # Initialize kernel parameters
        self.cl_kernel_time_marching_1.set_scalar_arg_dtypes(
            [None, None, None, None])
        self.cl_kernel_time_marching_2.set_scalar_arg_dtypes(
            [None, None, None, None, None, None, None, None])
        self.cl_kernel_check_bonds.set_scalar_arg_dtypes([None, None, None, None])
        self.cl_kernel_calculate_damage.set_scalar_arg_dtypes([None, None, None])
    def __call__(self):
        """
        Conduct one iteration of the integrator.

        :arg u: A (`nnodes`, 3) array containing the displacements of all
            nodes.
        :type u: :class:`numpy.ndarray`
        :arg f: A (`nnodes`, 3) array containing the components of the force
            acting on each node.
        :type f: :class:`numpy.ndarray`

        :returns: The new displacements after integration.
        :rtype: :class:`numpy.ndarray`
        """

    def runtime(self, model):
        # Time marching Part 1
        self.cl_kernel_time_marching_1(self.queue, (model.degrees_freedom * model.nnodes,),
                                  None, self.d_udn1, self.d_un, self.d_bc_types,
                                  self.d_bc_values)

        # Time marching Part 2
        self.cl_kernel_time_marching_2(self.queue, (model.nnodes,), None, self.d_udn1,
                                  self.d_un, self.d_vols, self.d_horizons, self.d_coords, self.d_bond_stiffness, self.d_force_bc_types, self.d_force_bc_values)

        # Check for broken bonds
        self.cl_kernel_check_bonds(self.queue,
                              (model.nnodes, model.max_horizon_length),
                              None, self.d_horizons, self.d_un, self.d_coords, self.d_bond_critical_stretch)
    def write(self, model, t, sample):
        """ Write a mesh file for the current timestep
        """
        self.cl_kernel_calculate_damage(self.queue, (model.nnodes,), None, 
                                           self.d_damage, self.d_horizons,
                                           self.d_horizons_lengths)
        cl.enqueue_copy(self.queue, self.h_damage, self.d_damage)
        cl.enqueue_copy(self.queue, self.h_un, self.d_un)
        # TODO define a failure criterion, idea: rate of change of damage goes to 0 after it has started increasing
        tip_displacement = 0
        tmp = 0
        for i in range(model.nnodes):
            if self.h_tip_types[i] == 1:
                tmp +=1
                tip_displacement += self.h_un[i][2]
        if tmp != 0:
            tip_displacement /= tmp
        else:
            tip_displacement = None
        vtk.write("output/U_"+"sample" + str(sample) +"t"+str(t) + ".vtk", "Solution time step = "+str(t),
                  model.coords, self.h_damage, self.h_un)
        #vtk.writeDamage("output/damage_" + str(t)+ "sample" + str(sample) + ".vtk", "Title", self.h_damage)
        return self.h_damage, tip_displacement

    def incrementLoad(self, model, load_scale):
        if model.num_force_bc_nodes != 0:
            tmp = -1. * model.max_reaction * load_scale / (model.num_force_bc_nodes)
            # update the host force_bc_values
            self.h_force_bc_values = tmp * np.ones((model.nnodes, model.degrees_freedom), dtype=np.float64)
            #print(h_force_bc_values)
            # update the GPU force_bc_values
            self.d_force_bc_values = cl.Buffer(self.context,
                               cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                               hostbuf=self.h_force_bc_values)

class EulerCromerOpenCLOptimised(Integrator):
    r"""
    Static Euler integrator for quasi-static loading, using OpenCL kernels.

    The Euler method is a first-order numerical integration method. The
    integration is given by,

    .. math::
        u(t + \delta t) = u(t) + \delta t f(t) d

    where :math:`u(t)` is the displacement at time :math:`t`, :math:`f(t)` is
    the force at time :math:`t`, :math:`\delta t` is the time step and
    :math:`d` is a dampening factor.
    
    These kernels are optimised over the naive kernels in EulerOpenCL
    """
    def __init__(self, model):
        """ Initialise the integration scheme
        """

        def output_device_info(device_id):
            sys.stdout.write("Device is ")
            sys.stdout.write(device_id.name)
            if device_id.type == cl.device_type.GPU:
                sys.stdout.write("GPU from ")
            elif device_id.type == cl.device_type.CPU:
                sys.stdout.write("CPU from ")
            else:
                sys.stdout.write("non CPU of GPU processor from ")
            sys.stdout.write(device_id.vendor)
            sys.stdout.write(" with a max of ")
            sys.stdout.write(str(device_id.max_compute_units))
            sys.stdout.write(" compute units\n")
            sys.stdout.flush()

        # Initializing OpenCL
        self.context = cl.create_some_context()
        self.queue = cl.CommandQueue(self.context)   

        # Print out device info
        output_device_info(self.context.devices[0])

        # Build the OpenCL program from file
        kernelsource = open(pathlib.Path(__file__).parent.absolute() / "kernels/opencl_euler_cromer_lumped_reduction.cl").read()
        SEP = " "

        options_string = (
            "-cl-fast-relaxed-math" + SEP
            + "-DPD_DPN_NODE_NO=" + str(model.nnodes * model.degrees_freedom) + SEP
            + "-DPD_NODE_NO=" + str(model.nnodes) + SEP
            + "-DMAX_HORIZON_LENGTH=" + str(model.max_horizon_length) + SEP
            + "-DPD_DT=" + str(model.dt) + SEP
            + "-DPD_RHO=" + str(model.density) + SEP
            + "-DPD_ETA=" + str(model.damping) + SEP)

        program = cl.Program(self.context, kernelsource).build([options_string])
        self.cl_kernel_time_marching_1 = program.TimeMarching1
        self.cl_kernel_time_marching_2 = program.TimeMarching2
        self.cl_kernel_reduce = program.reduce
        self.cl_kernel_calculate_damage = program.CalculateDamage

        # Set initial values in host memory

        # horizons and horizons lengths
        self.h_horizons = model.horizons
        self.h_horizons_lengths = model.horizons_lengths
        print(self.h_horizons_lengths)
        print(self.h_horizons)
        print("shape horizons lengths", self.h_horizons_lengths.shape)
        print("shape horizons lengths", self.h_horizons.shape)
        print(self.h_horizons_lengths.dtype, "dtype")

        # Nodal coordinates
        self.h_coords = np.ascontiguousarray(model.coords, dtype=np.float64)

        # Displacement boundary conditions types and delta values
        self.h_bc_types = model.bc_types
        self.h_bc_values = model.bc_values

        # Force boundary conditions types and values
        self.h_force_bc_types = model.force_bc_types
        self.h_force_bc_values = model.force_bc_values

        # For measuring tip displacemens (host memory only)
        self.h_tip_types = model.tip_types

        # Nodal volumes
        self.h_vols = model.V

        # Bond stiffnesses
        self.h_bond_stiffness =  np.ascontiguousarray(model.bond_stiffness, dtype=np.float64)
        self.h_bond_critical_stretch = np.ascontiguousarray(model.bond_critical_stretch, dtype=np.float64)

        # Displacements
        self.h_un = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)

        # Velocity
        self.h_udn = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)

        # Acceleration
        self.h_uddn = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)

        # Damage vector
        self.h_damage = np.empty(model.nnodes).astype(np.float64)

        # Bond forces
        self.h_forces = np.empty((model.nnodes, model.degrees_freedom, model.max_horizon_length), dtype = np.float64)

        if model.v == True:
            # Print the dtypes
            print("horizons", self.h_horizons.dtype)
            print("horizons_length", self.h_horizons_lengths.dtype)
            print("force_bc_types", self.h_bc_types.dtype)
            print("force_bc_values", self.h_bc_values.dtype)
            print("bc_types", self.h_bc_types.dtype)
            print("bc_values", self.h_bc_values.dtype)
            print("coords", self.h_coords.dtype)
            print("vols", self.h_vols.dtype)
            print("un", self.h_un.dtype)
            print("udn1", self.h_udn.dtype)
            print("damage", self.h_damage.dtype)

        # Build OpenCL data structures

        # Read only
        self.d_coords = cl.Buffer(self.context,
                             cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                             hostbuf=self.h_coords)
        self.d_bc_types = cl.Buffer(self.context,
                              cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                              hostbuf=self.h_bc_types)
        self.d_bc_values = cl.Buffer(self.context,
                               cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                               hostbuf=self.h_bc_values)
        self.d_force_bc_types = cl.Buffer(self.context,
                              cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                              hostbuf=self.h_force_bc_types)
        self.d_force_bc_values = cl.Buffer(self.context,
                               cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                               hostbuf=self.h_force_bc_values)
        self.d_vols = cl.Buffer(self.context,
                           cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                           hostbuf=self.h_vols)
        self.d_bond_stiffness = cl.Buffer(self.context,
                           cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                           hostbuf=self.h_bond_stiffness)
        self.d_bond_critical_stretch = cl.Buffer(self.context,
                           cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                           hostbuf=self.h_bond_critical_stretch)
        self.d_horizons_lengths = cl.Buffer(
                self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                hostbuf=self.h_horizons_lengths)

        # Read and write
        self.d_horizons = cl.Buffer(
                self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
                hostbuf=self.h_horizons)
        self.d_un = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_un.nbytes)
        self.d_udn = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_udn.nbytes)
        self.d_uddn = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_uddn.nbytes)
        self.d_forces = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_forces.nbytes)
        # Write only
        self.d_damage = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, self.h_damage.nbytes)
        # Initialize kernel parameters
        self.cl_kernel_time_marching_1.set_scalar_arg_dtypes(
            [None, None, None, None])
        self.cl_kernel_time_marching_2.set_scalar_arg_dtypes(
            [None, None, None, None, None, None, None])
        self.cl_kernel_reduce.set_scalar_arg_dtypes(
            [None, None, None, None, None])
        self.cl_kernel_calculate_damage.set_scalar_arg_dtypes([None, None, None])
    def __call__(self):
        """
        Conduct one iteration of the integrator.

        :arg u: A (`nnodes`, 3) array containing the displacements of all
            nodes.
        :type u: :class:`numpy.ndarray`
        :arg f: A (`nnodes`, 3) array containing the components of the force
            acting on each node.
        :type f: :class:`numpy.ndarray`

        :returns: The new displacements after integration.
        :rtype: :class:`numpy.ndarray`
        """

    def runtime(self, model):
        # Time marching Part 1
        self.cl_kernel_time_marching_1(self.queue, (model.degrees_freedom * model.nnodes,),
                                  None, self.d_udn, self.d_un, self.d_bc_types, self.d_bc_values)
        # Time marching Part 2
        self.cl_kernel_time_marching_2(self.queue, (model.nnodes, model.max_horizon_length),
                                          None, self.d_un, self.d_forces, self.d_vols, self.d_horizons, self.d_coords, self.d_bond_stiffness, self.d_bond_critical_stretch)
         # Reduce
        self.cl_kernel_reduce(self.queue, (model.degrees_freedom * model.nnodes,),
                                  None, self.d_forces, self.d_udn, self.d_uddn, self.d_force_bc_types, self.d_force_bc_values)

    def write(self, model, t, sample):
        """ Write a mesh file for the current timestep
        """
        self.cl_kernel_calculate_damage(self.queue, (model.nnodes,), None, 
                                           self.d_damage, self.d_horizons,
                                           self.d_horizons_lengths)
        cl.enqueue_copy(self.queue, self.h_damage, self.d_damage)
        cl.enqueue_copy(self.queue, self.h_un, self.d_un)
        # TODO define a failure criterion, idea: rate of change of damage goes to 0 after it has started increasing
        tip_displacement = 0
        tmp = 0
        for i in range(model.nnodes):
            if self.h_tip_types[i] == 1:
                tmp +=1
                tip_displacement += self.h_un[i][2]
        if tmp != 0:
            tip_displacement /= tmp
        else:
            tip_displacement = None
        vtk.write("output/U_"+"sample" + str(sample) +"t"+str(t) + ".vtk", "Solution time step = "+str(t),
                  model.coords, self.h_damage, self.h_un)
        #vtk.writeDamage("output/damage_" + str(t)+ "sample" + str(sample) + ".vtk", "Title", self.h_damage)
        return self.h_damage, tip_displacement

    def incrementLoad(self, model, load_scale):
        if model.num_force_bc_nodes != 0:
            tmp = -1. * model.max_reaction * load_scale / (model.num_force_bc_nodes)
            # update the host force_bc_values
            self.h_force_bc_values = tmp * np.ones((model.nnodes, model.degrees_freedom), dtype=np.float64)
            #print(h_force_bc_values)
            # update the GPU force_bc_values
            self.d_force_bc_values = cl.Buffer(self.context,
                               cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                               hostbuf=self.h_force_bc_values)

class MidpointOpenCL(Integrator):
    r"""
    Static Euler integrator for quasi-static loading, using OpenCL kernels.

    The Euler method is a first-order numerical integration method. The
    integration is given by,

    .. math::
        u(t + \delta t) = u(t) + \delta t f(t) d

    where :math:`u(t)` is the displacement at time :math:`t`, :math:`f(t)` is
    the force at time :math:`t`, :math:`\delta t` is the time step and
    :math:`d` is a dampening factor.
    """
    def __init__(self, model):
        """ Initialise the integration scheme
        """

        def output_device_info(device_id):
            sys.stdout.write("Device is ")
            sys.stdout.write(device_id.name)
            if device_id.type == cl.device_type.GPU:
                sys.stdout.write("GPU from ")
            elif device_id.type == cl.device_type.CPU:
                sys.stdout.write("CPU from ")
            else:
                sys.stdout.write("non CPU of GPU processor from ")
            sys.stdout.write(device_id.vendor)
            sys.stdout.write(" with a max of ")
            sys.stdout.write(str(device_id.max_compute_units))
            sys.stdout.write(" compute units\n")
            sys.stdout.flush()

        # Initializing OpenCL
        self.context = cl.create_some_context()
        self.queue = cl.CommandQueue(self.context)   

        # Print out device info
        output_device_info(self.context.devices[0])

        # Build the OpenCL program from file
        kernelsource = open(pathlib.Path(__file__).parent.absolute() / "kernels/opencl_midpoint.cl").read()
        SEP = " "

        options_string = (
            "-cl-fast-relaxed-math" + SEP
            + "-DPD_DPN_NODE_NO=" + str(model.degrees_freedom * model.nnodes) + SEP
            + "-DPD_NODE_NO=" + str(model.nnodes) + SEP
            + "-DMAX_HORIZON_LENGTH=" + str(model.max_horizon_length) + SEP
            + "-DPD_DT=" + str(model.dt) + SEP
            + "-DPD_RHO=" + str(model.density) + SEP
            + "-DPD_ETA=" + str(model.damping) + SEP)

        program = cl.Program(self.context, kernelsource).build([options_string])
        self.cl_kernel_calc_bond_force = program.CalcBondForce
        self.cl_kernel_partial_displacement_update = program.PartialUpdateDisplacement
        self.cl_kernel_full_displacement_update = program.FullUpdateDisplacement
        self.cl_kernel_check_bonds = program.CheckBonds
        self.cl_kernel_calculate_damage = program.CalculateDamage

        # Set initial values in host memory

        # horizons and horizons lengths
        self.h_horizons = model.horizons
        self.h_horizons_lengths = model.horizons_lengths
        print(self.h_horizons_lengths)
        print(self.h_horizons)
        print("shape horizons lengths", self.h_horizons_lengths.shape)
        print("shape horizons lengths", self.h_horizons.shape)
        print(self.h_horizons_lengths.dtype, "dtype")

        # Nodal coordinates
        self.h_coords = np.ascontiguousarray(model.coords, dtype=np.float64)

        # Displacement boundary conditions types and delta values
        self.h_bc_types = model.bc_types
        self.h_bc_values = model.bc_values

        self.h_tip_types = model.tip_types

        # Force boundary conditions types and values
        self.h_force_bc_types = model.force_bc_types
        self.h_force_bc_values = model.force_bc_values

        # Nodal volumes
        self.h_vols = model.V

        # Bond stiffnesses
        self.h_bond_stiffness =  np.ascontiguousarray(model.bond_stiffness, dtype=np.float64)
        self.h_bond_critical_stretch = np.ascontiguousarray(model.bond_critical_stretch, dtype=np.float64)

        # Displacements
        self.h_un = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)
        self.h_un1 = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)

        # Forces
        self.h_k1dn = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)
        self.h_k2dn = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)

        # Damage vector
        self.h_damage = np.empty(model.nnodes).astype(np.float64)

        if model.v == True:
            # Print the dtypes
            print("horizons", self.h_horizons.dtype)
            print("horizons_length", self.h_horizons_lengths.dtype)
            print("force_bc_types", self.h_bc_types.dtype)
            print("force_bc_values", self.h_bc_values.dtype)
            print("bc_types", self.h_bc_types.dtype)
            print("bc_values", self.h_bc_values.dtype)
            print("coords", self.h_coords.dtype)
            print("vols", self.h_vols.dtype)
            print("un", self.h_un.dtype)
            print("damage", self.h_damage.dtype)

        # Build OpenCL data structures

        # Read only
        self.d_coords = cl.Buffer(self.context,
                             cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                             hostbuf=self.h_coords)
        self.d_bc_types = cl.Buffer(self.context,
                              cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                              hostbuf=self.h_bc_types)
        self.d_bc_values = cl.Buffer(self.context,
                               cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                               hostbuf=self.h_bc_values)
        self.d_force_bc_types = cl.Buffer(self.context,
                              cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                              hostbuf=self.h_force_bc_types)
        self.d_force_bc_values = cl.Buffer(self.context,
                               cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                               hostbuf=self.h_force_bc_values)
        self.d_vols = cl.Buffer(self.context,
                           cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                           hostbuf=self.h_vols)
        self.d_bond_stiffness = cl.Buffer(self.context,
                           cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                           hostbuf=self.h_bond_stiffness)
        self.d_bond_critical_stretch = cl.Buffer(self.context,
                           cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                           hostbuf=self.h_bond_critical_stretch)
        self.d_horizons_lengths = cl.Buffer(
                self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                hostbuf=self.h_horizons_lengths)

        # Read and write
        self.d_horizons = cl.Buffer(
                self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
                hostbuf=self.h_horizons)
        self.d_un = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_un.nbytes)
        self.d_un1 = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_un1.nbytes)
        self.d_k1dn = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_k1dn.nbytes)
        self.d_k2dn = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_k2dn.nbytes)
        # Write only
        self.d_damage = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, self.h_damage.nbytes)
        # Initialize kernel parameters
        self.cl_kernel_calc_bond_force.set_scalar_arg_dtypes(
            [None, None, None, None, None, None, None, None])
        self.cl_kernel_partial_displacement_update.set_scalar_arg_dtypes(
            [None, None, None])
        self.cl_kernel_full_displacement_update.set_scalar_arg_dtypes(
            [None, None, None, None])
        self.cl_kernel_check_bonds.set_scalar_arg_dtypes([None, None, None, None])
        self.cl_kernel_calculate_damage.set_scalar_arg_dtypes([None, None, None])
    def __call__(self):
        """
        Conduct one iteration of the integrator.

        :arg u: A (`nnodes`, 3) array containing the displacements of all
            nodes.
        :type u: :class:`numpy.ndarray`
        :arg f: A (`nnodes`, 3) array containing the components of the force
            acting on each node.
        :type f: :class:`numpy.ndarray`

        :returns: The new displacements after integration.
        :rtype: :class:`numpy.ndarray`
        """

    def runtime(self, model):
        # Find k1dn (forces)
        self.cl_kernel_calc_bond_force(self.queue, 
                                          (model.nnodes,), 
                                          None, self.d_k1dn, self.d_un, self.d_vols, self.d_horizons, self.d_coords, self.d_bond_stiffness, self.d_force_bc_types, self.d_force_bc_values)
        # Partial update of displacement
        self.cl_kernel_partial_displacement_update(self.queue,
                                                   (model.degrees_freedom * model.nnodes,),
                                                     None, self.d_k1dn, self.d_un, self.d_un1)
        # Find k2dn (forces)
        self.cl_kernel_calc_bond_force(self.queue, 
                                          (model.nnodes,), 
                                          None, self.d_k2dn, self.d_un1, self.d_vols, self.d_horizons, self.d_coords, self.d_bond_stiffness, self.d_force_bc_types, self.d_force_bc_values)
        # Full update of displacement
        self.cl_kernel_full_displacement_update(self.queue, (model.degrees_freedom * model.nnodes,),
                                                      None, self.d_k2dn, self.d_un, self.d_bc_types, self.d_bc_values)
        # Check for broken bonds
        self.cl_kernel_check_bonds(self.queue,
                              (model.nnodes, model.max_horizon_length),
                              None, self.d_horizons, self.d_un, self.d_coords, self.d_bond_critical_stretch)
    def write(self, model, t, sample):
        """ Write a mesh file for the current timestep
        """
        self.cl_kernel_calculate_damage(self.queue, (model.nnodes,), None, 
                                           self.d_damage, self.d_horizons,
                                           self.d_horizons_lengths)
        cl.enqueue_copy(self.queue, self.h_damage, self.d_damage)
        cl.enqueue_copy(self.queue, self.h_un, self.d_un)
        # TODO define a failure criterion, idea: rate of change of damage goes to 0 after it has started increasing
        tip_displacement = 0
        tmp = 0
        for i in range(model.nnodes):
            if self.h_tip_types[i] == 1:
                tmp +=1
                tip_displacement += self.h_un[i][2]
        if tmp != 0:
            tip_displacement /= tmp
        else:
            tip_displacement = None
        vtk.write("output/U_"+"sample" + str(sample) +"t"+str(t) + ".vtk", "Solution time step = "+str(t),
                  model.coords, self.h_damage, self.h_un)
        #vtk.writeDamage("output/damage_" + str(t)+ "sample" + str(sample) + ".vtk", "Title", self.h_damage)
        return self.h_damage, tip_displacement

    def incrementLoad(self, model, load_scale):
        if model.num_force_bc_nodes != 0:
            tmp = -1. * model.max_reaction * load_scale / (model.num_force_bc_nodes)
            # update the host force_bc_values
            self.h_force_bc_values = tmp * np.ones((model.nnodes, model.degrees_freedom), dtype=np.float64)
            #print(h_force_bc_values)
            # update the GPU force_bc_values
            self.d_force_bc_values = cl.Buffer(self.context,
                               cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                               hostbuf=self.h_force_bc_values)

class VelocityVerletOpenCL(Integrator):
    """ TODO: check implementation is correct, is unstable as is
    """
    def __init__(self, model):
        """ Initialise the integration scheme
        """
        
        def output_device_info(device_id):
            sys.stdout.write("Device is ")
            sys.stdout.write(device_id.name)
            if device_id.type == cl.device_type.GPU:
                sys.stdout.write("GPU from ")
            elif device_id.type == cl.device_type.CPU:
                sys.stdout.write("CPU from ")
            else:
                sys.stdout.write("non CPU of GPU processor from ")
            sys.stdout.write(device_id.vendor)
            sys.stdout.write(" with a max of ")
            sys.stdout.write(str(device_id.max_compute_units))
            sys.stdout.write(" compute units\n")
            sys.stdout.flush()

        # Initializing OpenCL
        self.context = cl.create_some_context()
        self.queue = cl.CommandQueue(self.context)   

        # Print out device info
        output_device_info(self.context.devices[0])

        # Build the OpenCL program from file
        kernelsource = open(pathlib.Path(__file__).parent.absolute() / "kernels/opencl_velocity_verlet.cl").read()
        SEP = " "

        options_string = (
            "-cl-fast-relaxed-math" + SEP
            + "-DPD_DPN_NODE_NO=" + str(model.degrees_freedom * model.nnodes) + SEP
            + "-DPD_NODE_NO=" + str(model.nnodes) + SEP
            + "-DMAX_HORIZON_LENGTH=" + str(model.max_horizon_length) + SEP
            + "-DPD_DT=" + str(model.dt) + SEP
            + "-DPD_RHO=" + str(model.density) + SEP
            + "-DPD_ETA=" + str(model.damping) + SEP)

        program = cl.Program(self.context, kernelsource).build([options_string])
        self.cl_kernel_initial_values = program.InitialValues
        self.cl_kernel_time_marching_1 = program.TimeMarching1
        self.cl_kernel_time_marching_2 = program.TimeMarching2
        self.cl_kernel_time_marching_3 = program.TimeMarching3
        self.cl_kernel_check_bonds = program.CheckBonds
        self.cl_kernel_calculate_damage = program.CalculateDamage

        # Set initial values in host memory
        # horizons and horizons lengths
        self.h_horizons = model.horizons
        self.h_horizons_lengths = model.horizons_lengths
        # Nodal coordinates
        self.h_coords = np.ascontiguousarray(model.coords, dtype=np.float64)
        # Displacement boundary conditions types and delta values
        self.h_bc_types = model.bc_types
        self.h_bc_values = model.bc_values

        self.h_tip_types = model.tip_types

        # Force boundary conditions types and values
        self.h_force_bc_types = model.force_bc_types
        self.h_force_bc_values = model.force_bc_values

        # Nodal volumes
        self.h_vols = model.V

        # Bond stiffnesses
        self.h_bond_stiffness =  np.ascontiguousarray(model.bond_stiffness, dtype=np.float64)
        self.h_bond_critical_stretch = np.ascontiguousarray(model.bond_critical_stretch, dtype=np.float64)

        # Displacement
        self.h_un = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)
        # Velocity
        self.h_udn = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)
        # Acceleration
        self.h_uddn = np.empty((model.nnodes, model.degrees_freedom), dtype = np.float64)
        self.h_uddn1 = np.empty((model.nnodes, model.degrees_freedom), dtype = np.float64)
        # Damage vector
        self.h_damage = np.empty(model.nnodes).astype(np.float64)


        # Build OpenCL data structures
        # Read only
        self.d_coords = cl.Buffer(self.context,
                             cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                             hostbuf=self.h_coords)
        self.d_bctypes = cl.Buffer(self.context,
                              cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                              hostbuf=self.h_bc_types)
        self.d_bcvalues = cl.Buffer(self.context,
                               cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                               hostbuf=self.h_bc_values)
        self.d_force_bctypes = cl.Buffer(self.context,
                              cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                              hostbuf=self.h_force_bc_types)
        self.d_force_bcvalues = cl.Buffer(self.context,
                               cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                               hostbuf=self.h_force_bc_values)
        self.d_vols = cl.Buffer(self.context,
                           cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                           hostbuf=self.h_vols)
        self.d_bond_stiffness = cl.Buffer(self.context,
                           cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                           hostbuf=self.h_bond_stiffness)
        self.d_bond_critical_stretch = cl.Buffer(self.context,
                           cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                           hostbuf=self.h_bond_critical_stretch)
        self.d_horizons_lengths = cl.Buffer(
                self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                hostbuf=self.h_horizons_lengths)

        # Read and write
        self.d_horizons = cl.Buffer(
                self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
                hostbuf=self.h_horizons)
        self.d_un = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_un.nbytes)
        self.d_udn = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_udn.nbytes)
        self.d_uddn = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_uddn.nbytes)
        self.d_uddn1 = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_uddn1.nbytes)
        # Write only
        self.d_damage = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, self.h_damage.nbytes)
        # Initialize kernel parameters
        self.cl_kernel_initial_values.set_scalar_arg_dtypes(
            [None, None, None, None, None])
        self.cl_kernel_time_marching_1.set_scalar_arg_dtypes(
            [None, None, None])
        self.cl_kernel_time_marching_2.set_scalar_arg_dtypes(
            [None, None, None, None, None, None, None, None, None])
        self.cl_kernel_time_marching_3.set_scalar_arg_dtypes(
            [None, None, None, None, None])
        self.cl_kernel_check_bonds.set_scalar_arg_dtypes([None, None, None, None])
        self.cl_kernel_calculate_damage.set_scalar_arg_dtypes([None, None, None])
        self.cl_kernel_initial_values(self.queue, (model.degrees_freedom * model.nnodes,),
                                  None, self.d_un, self.d_udn, self.d_uddn, self.d_bctypes, self.d_bcvalues)

    def __call__(self):
        """
        Conduct one iteration of the integrator.

        :arg u: A (`nnodes`, 3) array containing the displacements of all
            nodes.
        :type u: :class:`numpy.ndarray`
        :arg f: A (`nnodes`, 3) array containing the components of the force
            acting on each node.
        :type f: :class:`numpy.ndarray`

        :returns: The new displacements after integration.
        :rtype: :class:`numpy.ndarray`
        """

    def runtime(self, model):
        # Time marching Part 1
        self.cl_kernel_time_marching_1(self.queue, (model.degrees_freedom * model.nnodes,),
                                  None, self.d_un, self.d_udn, self.d_uddn)
        # Time marching Part 2
        self.cl_kernel_time_marching_2(self.queue, (model.nnodes,), 
                                          None, self.d_uddn1, self.d_un, self.d_udn, self.d_horizons, self.d_coords, self.d_vols, self.d_bond_stiffness, self.d_force_bctypes, self.d_force_bcvalues)
        # Time marching Part 3
        self.cl_kernel_time_marching_3(self.queue, (model.degrees_freedom * model.nnodes,),
                                  None, self.d_udn, self.d_uddn, self.d_uddn1, self.d_bctypes, self.d_bcvalues)
        # Check for broken bonds
        self.cl_kernel_check_bonds(self.queue,
                              (model.nnodes, model.max_horizon_length),
                              None, self.d_horizons, self.d_un, self.d_coords, self.d_bond_critical_stretch)

    def write(self, model, t, sample):
        """ Write a mesh file for the current timestep
        """
        self.cl_kernel_calculate_damage(self.queue, (model.nnodes,), None, 
                                           self.d_damage, self.d_horizons,
                                           self.d_horizons_lengths)
        cl.enqueue_copy(self.queue, self.h_damage, self.d_damage)
        cl.enqueue_copy(self.queue, self.h_un, self.d_un)
        # TODO define a failure criterion, idea: rate of change of damage goes to 0 after it has started increasing
        tip_displacement = 0
        tmp = 0
        for i in range(model.nnodes):
            if self.h_tip_types[i] == 1:
                tmp +=1
                tip_displacement += self.h_un[i][2]
        if tmp != 0:
            tip_displacement /= tmp
        else:
            tip_displacement = None
        vtk.write("output/U_"+"sample" + str(sample) +"t"+str(t) + ".vtk", "Solution time step = "+str(t),
                  model.coords, self.h_damage, self.h_un)
        #vtk.writeDamage("output/damage_" + str(t)+ "sample" + str(sample) + ".vtk", "Title", self.h_damage)
        return self.h_damage, tip_displacement

    def incrementLoad(self, model, load_scale):
        if model.num_force_bc_nodes != 0:
            tmp = -1. * model.max_reaction * load_scale / (model.num_force_bc_nodes)
            # update the host force_bc_values
            self.h_force_bc_values = tmp * np.ones((model.nnodes, model.degrees_freedom), dtype=np.float64)
            #print(h_force_bc_values)
            # update the GPU force_bc_values
            self.d_force_bc_values = cl.Buffer(self.context,
                               cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                               hostbuf=self.h_force_bc_values)
class Kutta3(Integrator):
    r"""
    Static Euler integrator for quasi-static loading, using OpenCL kernels.

    The Euler method is a first-order numerical integration method. The
    integration is given by,

    .. math::
        u(t + \delta t) = u(t) + \delta t f(t) d

    where :math:`u(t)` is the displacement at time :math:`t`, :math:`f(t)` is
    the force at time :math:`t`, :math:`\delta t` is the time step and
    :math:`d` is a dampening factor.
    """
    def __init__(self, model):
        """ Initialise the integration scheme
        """
        
        def output_device_info(device_id):
            sys.stdout.write("Device is ")
            sys.stdout.write(device_id.name)
            if device_id.type == cl.device_type.GPU:
                sys.stdout.write("GPU from ")
            elif device_id.type == cl.device_type.CPU:
                sys.stdout.write("CPU from ")
            else:
                sys.stdout.write("non CPU of GPU processor from ")
            sys.stdout.write(device_id.vendor)
            sys.stdout.write(" with a max of ")
            sys.stdout.write(str(device_id.max_compute_units))
            sys.stdout.write(" compute units\n")
            sys.stdout.flush()

        # Initializing OpenCL
        self.context = cl.create_some_context()
        self.queue = cl.CommandQueue(self.context)   

        # Print out device info
        output_device_info(self.context.devices[0])

        # Build the OpenCL program from file
        kernelsource = open(pathlib.Path(__file__).parent.absolute() / "kernels/opencl_kutta3.cl").read()
        SEP = " "

        options_string = (
            "-cl-fast-relaxed-math" + SEP
            + "-DPD_DPN_NODE_NO=" + str(model.degrees_freedom * model.nnodes) + SEP
            + "-DPD_NODE_NO=" + str(model.nnodes) + SEP
            + "-DMAX_HORIZON_LENGTH=" + str(model.max_horizon_length) + SEP
            + "-DPD_DT=" + str(model.dt) + SEP
            + "-DPD_RHO=" + str(model.density) + SEP
            + "-DPD_ETA=" + str(model.damping) + SEP)

        program = cl.Program(self.context, kernelsource).build([options_string])
        self.cl_kernel_calc_bond_force = program.CalcBondForce
        self.cl_kernel_displacement_update = program.UpdateDisplacement
        self.cl_kernel_partial_displacement_update = program.PartialUpdateDisplacement
        self.cl_kernel_partial_displacement_update2 = program.PartialUpdateDisplacement2
        self.cl_kernel_check_bonds = program.CheckBonds
        self.cl_kernel_calculate_damage = program.CalculateDamage

        # Set initial values in host memory

        # horizons and horizons lengths
        self.h_horizons = model.horizons
        self.h_horizons_lengths = model.horizons_lengths
        print(self.h_horizons_lengths)
        print(self.h_horizons)
        print("shape horizons lengths", self.h_horizons_lengths.shape)
        print("shape horizons lengths", self.h_horizons.shape)
        print(self.h_horizons_lengths.dtype, "dtype")

        # Nodal coordinates
        self.h_coords = np.ascontiguousarray(model.coords, dtype=np.float64)

        # Displacement boundary conditions types and delta values
        self.h_bc_types = model.bc_types
        self.h_bc_values = model.bc_values

        self.h_tip_types = model.tip_types

        # Force boundary conditions types and values
        self.h_force_bc_types = model.force_bc_types
        self.h_force_bc_values = model.force_bc_values

        # Nodal volumes
        self.h_vols = model.V

        # Bond stiffnesses
        self.h_bond_stiffness =  np.ascontiguousarray(model.bond_stiffness, dtype=np.float64)
        self.h_bond_critical_stretch = np.ascontiguousarray(model.bond_critical_stretch, dtype=np.float64)

        # Displacements
        self.h_un = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)
        self.h_un1 = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)
        self.h_un2 = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)

        # Forces
        self.h_k1dn = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)
        self.h_k2dn = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)
        self.h_k3dn = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)

        # Damage vector
        self.h_damage = np.empty(model.nnodes).astype(np.float64)

        if model.v == True:
            # Print the dtypes
            print("horizons", self.h_horizons.dtype)
            print("horizons_length", self.h_horizons_lengths.dtype)
            print("force_bc_types", self.h_bc_types.dtype)
            print("force_bc_values", self.h_bc_values.dtype)
            print("bc_types", self.h_bc_types.dtype)
            print("bc_values", self.h_bc_values.dtype)
            print("coords", self.h_coords.dtype)
            print("vols", self.h_vols.dtype)
            print("un", self.h_un.dtype)
            print("damage", self.h_damage.dtype)

        # Build OpenCL data structures

        # Read only
        self.d_coords = cl.Buffer(self.context,
                             cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                             hostbuf=self.h_coords)
        self.d_bc_types = cl.Buffer(self.context,
                              cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                              hostbuf=self.h_bc_types)
        self.d_bc_values = cl.Buffer(self.context,
                               cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                               hostbuf=self.h_bc_values)
        self.d_force_bc_types = cl.Buffer(self.context,
                              cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                              hostbuf=self.h_force_bc_types)
        self.d_force_bc_values = cl.Buffer(self.context,
                               cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                               hostbuf=self.h_force_bc_values)
        self.d_vols = cl.Buffer(self.context,
                           cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                           hostbuf=self.h_vols)
        self.d_bond_stiffness = cl.Buffer(self.context,
                           cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                           hostbuf=self.h_bond_stiffness)
        self.d_bond_critical_stretch = cl.Buffer(self.context,
                           cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                           hostbuf=self.h_bond_critical_stretch)
        self.d_horizons_lengths = cl.Buffer(
                self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                hostbuf=self.h_horizons_lengths)

        # Read and write
        self.d_horizons = cl.Buffer(
                self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
                hostbuf=self.h_horizons)
        self.d_un = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_un.nbytes)
        self.d_un1 = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_k1dn.nbytes)
        self.d_un2 = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_k1dn.nbytes)
        self.d_k1dn = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_k1dn.nbytes)
        self.d_k2dn = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_k2dn.nbytes)
        self.d_k3dn = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_k3dn.nbytes)
        # Write only
        self.d_damage = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, self.h_damage.nbytes)
        # Initialize kernel parameters
        self.cl_kernel_calc_bond_force.set_scalar_arg_dtypes(
            [None, None, None, None, None, None, None, None])
        self.cl_kernel_displacement_update.set_scalar_arg_dtypes(
            [None, None, None, None, None, None])
        self.cl_kernel_partial_displacement_update.set_scalar_arg_dtypes(
            [None, None, None])
        self.cl_kernel_partial_displacement_update2.set_scalar_arg_dtypes(
            [None, None, None, None])
        self.cl_kernel_check_bonds.set_scalar_arg_dtypes([None, None, None, None])
        self.cl_kernel_calculate_damage.set_scalar_arg_dtypes([None, None, None])
    def __call__(self):
        """
        Conduct one iteration of the integrator.

        :arg u: A (`nnodes`, 3) array containing the displacements of all
            nodes.
        :type u: :class:`numpy.ndarray`
        :arg f: A (`nnodes`, 3) array containing the components of the force
            acting on each node.
        :type f: :class:`numpy.ndarray`

        :returns: The new displacements after integration.
        :rtype: :class:`numpy.ndarray`
        """

    def runtime(self, model):
        # Find k1dn (forces)
        self.cl_kernel_calc_bond_force(self.queue, 
                                          (model.nnodes,), 
                                          None, self.d_k1dn, self.d_un, self.d_vols, self.d_horizons, self.d_coords, self.d_bond_stiffness, self.d_force_bc_types, self.d_force_bc_values)
        # Partial update of displacements
        self.cl_kernel_partial_displacement_update(self.queue, (model.nnodes * model.degrees_freedom,),
                                                     None, self.d_k1dn, self.d_un, self.d_un1)
        # Find k2dn (forces)
        self.cl_kernel_calc_bond_force(self.queue, 
                                          (model.nnodes,), 
                                          None, self.d_k2dn, self.d_un1, self.d_vols, self.d_horizons, self.d_coords, self.d_bond_stiffness, self.d_force_bc_types, self.d_force_bc_values)
        # Partial update displacements 2
        self.cl_kernel_partial_displacement_update2(self.queue, (model.nnodes * model.degrees_freedom,),
                                                      None, self.d_k2dn, self.d_k1dn, self.d_un, self.d_un2)
        # Find k3dn (forces)
        self.cl_kernel_calc_bond_force(self.queue, 
                                          (model.nnodes,), 
                                          None, self.d_k3dn, self.d_un2, self.d_vols, self.d_horizons, self.d_coords, self.d_bond_stiffness, self.d_force_bc_types, self.d_force_bc_values)
        # Finally update the displacements using weighted average of 3 incriments
        self.cl_kernel_displacement_update(self.queue, 
                                              (model.nnodes * model.degrees_freedom,), 
                                              None, self.d_k1dn, self.d_k2dn, self.d_k3dn, self.d_bc_types, self.d_bc_values, self.d_un)
        # Check for broken bonds
        self.cl_kernel_check_bonds(self.queue,
                              (model.nnodes, model.max_horizon_length),
                              None, self.d_horizons, self.d_un, self.d_coords, self.d_bond_critical_stretch)
    def write(self, model, t, sample):
        """ Write a mesh file for the current timestep
        """
        self.cl_kernel_calculate_damage(self.queue, (model.nnodes,), None, 
                                           self.d_damage, self.d_horizons,
                                           self.d_horizons_lengths)
        cl.enqueue_copy(self.queue, self.h_damage, self.d_damage)
        cl.enqueue_copy(self.queue, self.h_un, self.d_un)
        # TODO define a failure criterion, idea: rate of change of damage goes to 0 after it has started increasing
        tip_displacement = 0
        tmp = 0
        for i in range(model.nnodes):
            if self.h_tip_types[i] == 1:
                tmp +=1
                tip_displacement += self.h_un[i][2]
        if tmp != 0:
            tip_displacement /= tmp
        else:
            tip_displacement = None
        vtk.write("output/U_"+"sample" + str(sample) +"t"+str(t) + ".vtk", "Solution time step = "+str(t),
                  model.coords, self.h_damage, self.h_un)
        #vtk.writeDamage("output/damage_" + str(t)+ "sample" + str(sample) + ".vtk", "Title", self.h_damage)
        return self.h_damage, tip_displacement

    def incrementLoad(self, model, load_scale):
        if model.num_force_bc_nodes != 0:
            tmp = -1. * model.max_reaction * load_scale / (model.num_force_bc_nodes)
            # update the host force_bc_values
            self.h_force_bc_values = tmp * np.ones((model.nnodes, model.degrees_freedom), dtype=np.float64)
            #print(h_force_bc_values)
            # update the GPU force_bc_values
            self.d_force_bc_values = cl.Buffer(self.context,
                               cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                               hostbuf=self.h_force_bc_values)
            
class RK4(Integrator):
    r"""
    Static Euler integrator for quasi-static loading, using OpenCL kernels.

    The Euler method is a first-order numerical integration method. The
    integration is given by,

    .. math::
        u(t + \delta t) = u(t) + \delta t f(t) d

    where :math:`u(t)` is the displacement at time :math:`t`, :math:`f(t)` is
    the force at time :math:`t`, :math:`\delta t` is the time step and
    :math:`d` is a dampening factor.
    """
    def __init__(self, model):
        """ Initialise the integration scheme
        """
        
        def output_device_info(device_id):
            sys.stdout.write("Device is ")
            sys.stdout.write(device_id.name)
            if device_id.type == cl.device_type.GPU:
                sys.stdout.write("GPU from ")
            elif device_id.type == cl.device_type.CPU:
                sys.stdout.write("CPU from ")
            else:
                sys.stdout.write("non CPU of GPU processor from ")
            sys.stdout.write(device_id.vendor)
            sys.stdout.write(" with a max of ")
            sys.stdout.write(str(device_id.max_compute_units))
            sys.stdout.write(" compute units\n")
            sys.stdout.flush()

        # Initializing OpenCL
        self.context = cl.create_some_context()
        self.queue = cl.CommandQueue(self.context)   

        # Print out device info
        output_device_info(self.context.devices[0])

        # Build the OpenCL program from file
        kernelsource = open(pathlib.Path(__file__).parent.absolute() / "kernels/opencl_RK4.cl").read()
        SEP = " "

        options_string = (
            "-cl-fast-relaxed-math" + SEP
            + "-DPD_DPN_NODE_NO=" + str(model.degrees_freedom * model.nnodes) + SEP
            + "-DPD_NODE_NO=" + str(model.nnodes) + SEP
            + "-DMAX_HORIZON_LENGTH=" + str(model.max_horizon_length) + SEP
            + "-DPD_DT=" + str(model.dt) + SEP
            + "-DPD_RHO=" + str(model.density) + SEP
            + "-DPD_ETA=" + str(model.damping) + SEP)

        program = cl.Program(self.context, kernelsource).build([options_string])
        self.cl_kernel_calc_bond_force = program.CalcBondForce
        self.cl_kernel_displacement_update = program.UpdateDisplacement
        self.cl_kernel_partial_displacement_update = program.PartialUpdateDisplacement
        self.cl_kernel_partial_displacement_update2 = program.PartialUpdateDisplacement2
        self.cl_kernel_check_bonds = program.CheckBonds
        self.cl_kernel_calculate_damage = program.CalculateDamage

        # Set initial values in host memory

        # horizons and horizons lengths
        self.h_horizons = model.horizons
        self.h_horizons_lengths = model.horizons_lengths
        print(self.h_horizons_lengths)
        print(self.h_horizons)
        print("shape horizons lengths", self.h_horizons_lengths.shape)
        print("shape horizons lengths", self.h_horizons.shape)
        print(self.h_horizons_lengths.dtype, "dtype")

        # Nodal coordinates
        self.h_coords = np.ascontiguousarray(model.coords, dtype=np.float64)

        # Displacement boundary conditions types and delta values
        self.h_bc_types = model.bc_types
        self.h_bc_values = model.bc_values

        self.h_tip_types = model.tip_types

        # Force boundary conditions types and values
        self.h_force_bc_types = model.force_bc_types
        self.h_force_bc_values = model.force_bc_values

        # Nodal volumes
        self.h_vols = model.V

        # Bond stiffnesses
        self.h_bond_stiffness =  np.ascontiguousarray(model.bond_stiffness, dtype=np.float64)
        self.h_bond_critical_stretch = np.ascontiguousarray(model.bond_critical_stretch, dtype=np.float64)

        # Displacements
        self.h_un = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)
        self.h_un1 = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)
        self.h_un2 = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)
        self.h_un3 = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)

        # Forces
        self.h_k1dn = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)
        self.h_k2dn = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)
        self.h_k3dn = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)
        self.h_k4dn = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)

        # Damage vector
        self.h_damage = np.empty(model.nnodes).astype(np.float64)

        if model.v == True:
            # Print the dtypes
            print("horizons", self.h_horizons.dtype)
            print("horizons_length", self.h_horizons_lengths.dtype)
            print("force_bc_types", self.h_bc_types.dtype)
            print("force_bc_values", self.h_bc_values.dtype)
            print("bc_types", self.h_bc_types.dtype)
            print("bc_values", self.h_bc_values.dtype)
            print("coords", self.h_coords.dtype)
            print("vols", self.h_vols.dtype)
            print("un", self.h_un.dtype)
            print("damage", self.h_damage.dtype)

        # Build OpenCL data structures

        # Read only
        self.d_coords = cl.Buffer(self.context,
                             cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                             hostbuf=self.h_coords)
        self.d_bc_types = cl.Buffer(self.context,
                              cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                              hostbuf=self.h_bc_types)
        self.d_bc_values = cl.Buffer(self.context,
                               cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                               hostbuf=self.h_bc_values)
        self.d_force_bc_types = cl.Buffer(self.context,
                              cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                              hostbuf=self.h_force_bc_types)
        self.d_force_bc_values = cl.Buffer(self.context,
                               cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                               hostbuf=self.h_force_bc_values)
        self.d_vols = cl.Buffer(self.context,
                           cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                           hostbuf=self.h_vols)
        self.d_bond_stiffness = cl.Buffer(self.context,
                           cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                           hostbuf=self.h_bond_stiffness)
        self.d_bond_critical_stretch = cl.Buffer(self.context,
                           cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                           hostbuf=self.h_bond_critical_stretch)
        self.d_horizons_lengths = cl.Buffer(
                self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                hostbuf=self.h_horizons_lengths)

        # Read and write
        self.d_horizons = cl.Buffer(
                self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
                hostbuf=self.h_horizons)
        self.d_un = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_un.nbytes)
        self.d_un1 = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_k1dn.nbytes)
        self.d_un2 = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_k1dn.nbytes)
        self.d_un3 = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_k1dn.nbytes)
        self.d_k1dn = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_k1dn.nbytes)
        self.d_k2dn = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_k2dn.nbytes)
        self.d_k3dn = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_k3dn.nbytes)
        self.d_k4dn = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_k3dn.nbytes)
        # Write only
        self.d_damage = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, self.h_damage.nbytes)
        # Initialize kernel parameters
        self.cl_kernel_calc_bond_force.set_scalar_arg_dtypes(
            [None, None, None, None, None, None, None, None])
        self.cl_kernel_displacement_update.set_scalar_arg_dtypes(
            [None, None, None, None, None, None, None])
        self.cl_kernel_partial_displacement_update.set_scalar_arg_dtypes(
            [None, None, None])
        self.cl_kernel_partial_displacement_update2.set_scalar_arg_dtypes(
            [None, None, None])
        self.cl_kernel_check_bonds.set_scalar_arg_dtypes([None, None, None, None])
        self.cl_kernel_calculate_damage.set_scalar_arg_dtypes([None, None, None])
    def __call__(self):
        """
        Conduct one iteration of the integrator.

        :arg u: A (`nnodes`, 3) array containing the displacements of all
            nodes.
        :type u: :class:`numpy.ndarray`
        :arg f: A (`nnodes`, 3) array containing the components of the force
            acting on each node.
        :type f: :class:`numpy.ndarray`

        :returns: The new displacements after integration.
        :rtype: :class:`numpy.ndarray`
        """

    def runtime(self, model):
        # Find k1dn (forces)
        self.cl_kernel_calc_bond_force(self.queue, 
                                          (model.nnodes,), 
                                          None, self.d_k1dn, self.d_un, self.d_vols, self.d_horizons, self.d_coords, self.d_bond_stiffness, self.d_force_bc_types, self.d_force_bc_values)
        # Partial update of displacements
        self.cl_kernel_partial_displacement_update(self.queue, (model.nnodes * model.degrees_freedom,),
                                                     None, self.d_k1dn, self.d_un, self.d_un1)
        # Find k2dn (forces)
        self.cl_kernel_calc_bond_force(self.queue, 
                                          (model.nnodes,), 
                                          None, self.d_k2dn, self.d_un1, self.d_vols, self.d_horizons, self.d_coords, self.d_bond_stiffness, self.d_force_bc_types, self.d_force_bc_values)
        # Partial update of displacements
        self.cl_kernel_partial_displacement_update(self.queue, (model.nnodes * model.degrees_freedom,),
                                                      None, self.d_k2dn, self.d_un, self.d_un2)
        # Find k3dn (forces)
        self.cl_kernel_calc_bond_force(self.queue, 
                                          (model.nnodes,),
                                          None, self.d_k3dn, self.d_un2, self.d_vols, self.d_horizons, self.d_coords, self.d_bond_stiffness, self.d_force_bc_types, self.d_force_bc_values)
        # Partial update of displacements
        self.cl_kernel_partial_displacement_update2(self.queue, 
                                                   (model.nnodes * model.degrees_freedom,),
                                                   None, self.d_k3dn, self.d_un, self.d_un3)
        # Find k3dn (forces)
        self.cl_kernel_calc_bond_force(self.queue, 
                                          (model.nnodes,),
                                          None, self.d_k4dn, self.d_un3, self.d_vols, self.d_horizons, self.d_coords, self.d_bond_stiffness, self.d_force_bc_types, self.d_force_bc_values)
        # Finally update the displacements using weighted average of 4 incriments
        self.cl_kernel_displacement_update(self.queue, 
                                              (model.nnodes * model.degrees_freedom,), 
                                              None, self.d_k1dn, self.d_k2dn, self.d_k3dn, self.d_k4dn, self.d_bc_types, self.d_bc_values, self.d_un)
        # Check for broken bonds
        self.cl_kernel_check_bonds(self.queue,
                              (model.nnodes, model.max_horizon_length),
                              None, self.d_horizons, self.d_un, self.d_coords, self.d_bond_critical_stretch)

    def write(self, model, t, sample):
        """ Write a mesh file for the current timestep
        """
        self.cl_kernel_calculate_damage(self.queue, (model.nnodes,), None, 
                                           self.d_damage, self.d_horizons,
                                           self.d_horizons_lengths)
        cl.enqueue_copy(self.queue, self.h_damage, self.d_damage)
        cl.enqueue_copy(self.queue, self.h_un, self.d_un)
        # TODO define a failure criterion, idea: rate of change of damage goes to 0 after it has started increasing
        tip_displacement = 0
        tmp = 0
        for i in range(model.nnodes):
            if self.h_tip_types[i] == 1:
                tmp +=1
                tip_displacement += self.h_un[i][2]
        if tmp != 0:
            tip_displacement /= tmp
        else:
            tip_displacement = None
        vtk.write("output/U_"+"sample" + str(sample) +"t"+str(t) + ".vtk", "Solution time step = "+str(t),
                  model.coords, self.h_damage, self.h_un)
        #vtk.writeDamage("output/damage_" + str(t)+ "sample" + str(sample) + ".vtk", "Title", self.h_damage)
        return self.h_damage, tip_displacement

    def incrementLoad(self, model, load_scale):
        if model.num_force_bc_nodes != 0:
            tmp = -1. * model.max_reaction * load_scale / (model.num_force_bc_nodes)
            # update the host force_bc_values
            self.h_force_bc_values = tmp * np.ones((model.nnodes, model.degrees_freedom), dtype=np.float64)
            #print(h_force_bc_values)
            # update the GPU force_bc_values
            self.d_force_bc_values = cl.Buffer(self.context,
                               cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                               hostbuf=self.h_force_bc_values)
class RK4AT(Integrator):
    r"""
    4th order Runge-Kutta with adaptive time step

    .. math::
        u(t + \delta t) = u(t) + \delta t f(t) d

    where :math:`u(t)` is the displacement at time :math:`t`, :math:`f(t)` is
    the force at time :math:`t`, :math:`\delta t` is the time step and
    :math:`d` is a dampening factor.
    """
    def __init__(self, model):
        """ Initialise the integration scheme
        """
        
        def output_device_info(device_id):
            sys.stdout.write("Device is ")
            sys.stdout.write(device_id.name)
            if device_id.type == cl.device_type.GPU:
                sys.stdout.write("GPU from ")
            elif device_id.type == cl.device_type.CPU:
                sys.stdout.write("CPU from ")
            else:
                sys.stdout.write("non CPU of GPU processor from ")
            sys.stdout.write(device_id.vendor)
            sys.stdout.write(" with a max of ")
            sys.stdout.write(str(device_id.max_compute_units))
            sys.stdout.write(" compute units\n")
            sys.stdout.flush()

        # Initializing OpenCL
        self.context = cl.create_some_context()
        self.queue = cl.CommandQueue(self.context)   

        # Print out device info
        output_device_info(self.context.devices[0])

        # Build the OpenCL program from file
        kernelsource = open(pathlib.Path(__file__).parent.absolute() / "kernels/opencl_RK4AT.cl").read()
        SEP = " "

        options_string = (
            "-cl-fast-relaxed-math" + SEP
            + "-DPD_DPN_NODE_NO=" + str(model.degrees_freedom * model.nnodes) + SEP
            + "-DPD_NODE_NO=" + str(model.nnodes) + SEP
            + "-DMAX_HORIZON_LENGTH=" + str(model.max_horizon_length) + SEP
            + "-DPD_RHO=" + str(model.density) + SEP
            + "-DPD_ETA=" + str(model.damping) + SEP)

        program = cl.Program(self.context, kernelsource).build([options_string])
        self.cl_kernel_calc_bond_force = program.CalcBondForce
        self.cl_kernel_displacement_update = program.UpdateDisplacement
        self.cl_kernel_partial_displacement_update = program.PartialUpdateDisplacement
        self.cl_kernel_partial_displacement_update2 = program.PartialUpdateDisplacement2
        self.cl_kernel_check_bonds = program.CheckBonds
        self.cl_kernel_calculate_damage = program.CalculateDamage

        # Set initial values in host memory

        # horizons and horizons lengths
        self.h_horizons = model.horizons
        self.h_horizons_lengths = model.horizons_lengths
        print(self.h_horizons_lengths)
        print(self.h_horizons)
        print("shape horizons lengths", self.h_horizons_lengths.shape)
        print("shape horizons lengths", self.h_horizons.shape)
        print(self.h_horizons_lengths.dtype, "dtype")

        # Nodal coordinates
        self.h_coords = np.ascontiguousarray(model.coords, dtype=np.float64)

        # Displacement boundary conditions types and delta values
        self.h_bc_types = model.bc_types
        self.h_bc_values = model.bc_values

        self.h_tip_types = model.tip_types

        # Force boundary conditions types and values
        self.h_force_bc_types = model.force_bc_types
        self.h_force_bc_values = model.force_bc_values

        # Nodal volumes
        self.h_vols = model.V

        # Bond stiffnesses
        self.h_bond_stiffness =  np.ascontiguousarray(model.bond_stiffness, dtype=np.float64)
        self.h_bond_critical_stretch = np.ascontiguousarray(model.bond_critical_stretch, dtype=np.float64)

        # Displacements
        self.h_un = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)
        self.h_un1 = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)
        self.h_un2 = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)
        self.h_un3 = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)

        # Forces
        self.h_k1dn = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)
        self.h_k2dn = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)
        self.h_k3dn = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)
        self.h_k4dn = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)

        # Damage vector
        self.h_damage = np.empty(model.nnodes).astype(np.float64)

        # Time step size
        self.h_dt = np.float64(model.dt)

        # Build OpenCL data structures

        # Read only
        self.d_coords = cl.Buffer(self.context,
                             cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                             hostbuf=self.h_coords)
        self.d_bc_types = cl.Buffer(self.context,
                              cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                              hostbuf=self.h_bc_types)
        self.d_bc_values = cl.Buffer(self.context,
                               cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                               hostbuf=self.h_bc_values)
        self.d_force_bc_types = cl.Buffer(self.context,
                              cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                              hostbuf=self.h_force_bc_types)
        self.d_force_bc_values = cl.Buffer(self.context,
                               cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                               hostbuf=self.h_force_bc_values)
        self.d_vols = cl.Buffer(self.context,
                           cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                           hostbuf=self.h_vols)
        self.d_bond_stiffness = cl.Buffer(self.context,
                           cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                           hostbuf=self.h_bond_stiffness)
        self.d_bond_critical_stretch = cl.Buffer(self.context,
                           cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                           hostbuf=self.h_bond_critical_stretch)
        self.d_horizons_lengths = cl.Buffer(
                self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                hostbuf=self.h_horizons_lengths)

        # Read and write
        self.d_horizons = cl.Buffer(
                self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
                hostbuf=self.h_horizons)
        self.d_un = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_un.nbytes)
        self.d_un1 = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_k1dn.nbytes)
        self.d_un2 = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_k1dn.nbytes)
        self.d_un3 = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_k1dn.nbytes)
        self.d_k1dn = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_k1dn.nbytes)
        self.d_k2dn = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_k2dn.nbytes)
        self.d_k3dn = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_k3dn.nbytes)
        self.d_k4dn = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_k3dn.nbytes)
        #self.d_dt = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_dt)
        # Write only
        self.d_damage = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, self.h_damage.nbytes)
        # Initialize kernel parameters
        self.cl_kernel_calc_bond_force.set_scalar_arg_dtypes(
            [None, None, None, None, None, None, None, None])
        self.cl_kernel_displacement_update.set_scalar_arg_dtypes(
            [None, None, None, None, None, None, None, None])
        self.cl_kernel_partial_displacement_update.set_scalar_arg_dtypes(
            [None, None, None, None])
        self.cl_kernel_partial_displacement_update2.set_scalar_arg_dtypes(
            [None, None, None, None])
        self.cl_kernel_check_bonds.set_scalar_arg_dtypes([None, None, None, None])
        self.cl_kernel_calculate_damage.set_scalar_arg_dtypes([None, None, None])
    def __call__(self):
        """
        Conduct one iteration of the integrator.

        :arg u: A (`nnodes`, 3) array containing the displacements of all
            nodes.
        :type u: :class:`numpy.ndarray`
        :arg f: A (`nnodes`, 3) array containing the components of the force
            acting on each node.
        :type f: :class:`numpy.ndarray`

        :returns: The new displacements after integration.
        :rtype: :class:`numpy.ndarray`
        """

    def runtime(self, model):
        # Find k1dn (forces)
        self.cl_kernel_calc_bond_force(self.queue, 
                                          (model.nnodes,), 
                                          None, self.d_k1dn, self.d_un, self.d_vols, self.d_horizons, self.d_coords, self.d_bond_stiffness, self.d_force_bc_types, self.d_force_bc_values)
        # Partial update of displacements
        self.cl_kernel_partial_displacement_update(self.queue, (model.nnodes * model.degrees_freedom,),
                                                     None, self.d_k1dn, self.d_un, self.d_un1, self.h_dt)
        # Find k2dn (forces)
        self.cl_kernel_calc_bond_force(self.queue, 
                                          (model.nnodes,), 
                                          None, self.d_k2dn, self.d_un1, self.d_vols, self.d_horizons, self.d_coords, self.d_bond_stiffness, self.d_force_bc_types, self.d_force_bc_values)
        # Partial update of displacements
        self.cl_kernel_partial_displacement_update(self.queue, (model.nnodes * model.degrees_freedom,),
                                                      None, self.d_k2dn, self.d_un, self.d_un2, self.h_dt)
        # Find k3dn (forces)
        self.cl_kernel_calc_bond_force(self.queue, 
                                          (model.nnodes,),
                                          None, self.d_k3dn, self.d_un2, self.d_vols, self.d_horizons, self.d_coords, self.d_bond_stiffness, self.d_force_bc_types, self.d_force_bc_values)
        # Partial update of displacements
        self.cl_kernel_partial_displacement_update2(self.queue, 
                                                   (model.nnodes * model.degrees_freedom,),
                                                   None, self.d_k3dn, self.d_un, self.d_un3, self.h_dt)
        # Find k3dn (forces)
        self.cl_kernel_calc_bond_force(self.queue, 
                                          (model.nnodes,),
                                          None, self.d_k4dn, self.d_un3, self.d_vols, self.d_horizons, self.d_coords, self.d_bond_stiffness, self.d_force_bc_types, self.d_force_bc_values)
        # Finally update the displacements using weighted average of 4 incriments
        self.cl_kernel_displacement_update(self.queue, 
                                              (model.nnodes * model.degrees_freedom,), 
                                              None, self.d_k1dn, self.d_k2dn, self.d_k3dn, self.d_k4dn, self.d_bc_types, self.d_bc_values, self.d_un, self.h_dt)
        # Check for broken bonds
        self.cl_kernel_check_bonds(self.queue,
                              (model.nnodes, model.max_horizon_length),
                              None, self.d_horizons, self.d_un, self.d_coords, self.d_bond_critical_stretch)
    def write(self, model, t, sample):
        """ Write a mesh file for the current timestep
        """
        self.cl_kernel_calculate_damage(self.queue, (model.nnodes,), None, 
                                           self.d_damage, self.d_horizons,
                                           self.d_horizons_lengths)
        cl.enqueue_copy(self.queue, self.h_damage, self.d_damage)
        cl.enqueue_copy(self.queue, self.h_un, self.d_un)
        # TODO define a failure criterion, idea: rate of change of damage goes to 0 after it has started increasing
        tip_displacement = 0
        tmp = 0
        for i in range(model.nnodes):
            if self.h_tip_types[i] == 1:
                tmp +=1
                tip_displacement += self.h_un[i][2]
        if tmp != 0:
            tip_displacement /= tmp
        else:
            tip_displacement = None
        vtk.write("output/U_"+"sample" + str(sample) +"t"+str(t) + ".vtk", "Solution time step = "+str(t),
                  model.coords, self.h_damage, self.h_un)
        #vtk.writeDamage("output/damage_" + str(t)+ "sample" + str(sample) + ".vtk", "Title", self.h_damage)
        return self.h_damage, tip_displacement
    def incrementLoad(self, model, load_scale):
        if model.num_force_bc_nodes != 0:
            tmp = -1. * model.max_reaction * load_scale / (model.num_force_bc_nodes)
            # update the host force_bc_values
            self.h_force_bc_values = tmp * np.ones((model.nnodes, model.degrees_freedom), dtype=np.float64)
            #print(h_force_bc_values)
            # update the GPU force_bc_values
            self.d_force_bc_values = cl.Buffer(self.context,
                               cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                               hostbuf=self.h_force_bc_values)
class HeunEuler(Integrator):
    r"""
    Static Euler integrator for quasi-static loading, using OpenCL kernels.

    The Euler method is a first-order numerical integration method. The
    integration is given by,

    .. math::
        u(t + \delta t) = u(t) + \delta t f(t) d

    where :math:`u(t)` is the displacement at time :math:`t`, :math:`f(t)` is
    the force at time :math:`t`, :math:`\delta t` is the time step and
    :math:`d` is a dampening factor.
    """
    def __init__(self, model):
        """ Initialise the integration scheme
        """
        
        def output_device_info(device_id):
            sys.stdout.write("Device is ")
            sys.stdout.write(device_id.name)
            if device_id.type == cl.device_type.GPU:
                sys.stdout.write("GPU from ")
            elif device_id.type == cl.device_type.CPU:
                sys.stdout.write("CPU from ")
            else:
                sys.stdout.write("non CPU of GPU processor from ")
            sys.stdout.write(device_id.vendor)
            sys.stdout.write(" with a max of ")
            sys.stdout.write(str(device_id.max_compute_units))
            sys.stdout.write(" compute units\n")
            sys.stdout.flush()

        # Initializing OpenCL
        self.context = cl.create_some_context()
        self.queue = cl.CommandQueue(self.context)   

        # Print out device info
        output_device_info(self.context.devices[0])

        # Build the OpenCL program from file
        kernelsource = open(pathlib.Path(__file__).parent.absolute() / "kernels/opencl_heun_euler.cl").read()
        SEP = " "

        options_string = (
            "-cl-fast-relaxed-math" + SEP
            + "-DPD_DPN_NODE_NO=" + str(model.degrees_freedom * model.nnodes) + SEP
            + "-DPD_NODE_NO=" + str(model.nnodes) + SEP
            + "-DMAX_HORIZON_LENGTH=" + str(model.max_horizon_length) + SEP
            + "-DPD_RHO=" + str(model.density) + SEP
            + "-DPD_ETA=" + str(model.damping) + SEP)

        program = cl.Program(self.context, kernelsource).build([options_string])
        self.cl_kernel_calc_bond_force = program.CalcBondForce
        self.cl_kernel_displacement_update = program.UpdateDisplacement
        self.cl_kernel_partial_displacement_update = program.PartialUpdateDisplacement
        self.cl_kernel_check_bonds = program.CheckBonds
        self.cl_kernel_calculate_damage = program.CalculateDamage
        self.cl_kernel_check_error = program.CheckError

        # Set initial values in host memory

        # horizons and horizons lengths
        self.h_horizons = model.horizons
        self.h_horizons_lengths = model.horizons_lengths
        print(self.h_horizons_lengths)
        print(self.h_horizons)
        print("shape horizons lengths", self.h_horizons_lengths.shape)
        print("shape horizons lengths", self.h_horizons.shape)
        print(self.h_horizons_lengths.dtype, "dtype")

        # Nodal coordinates
        self.h_coords = np.ascontiguousarray(model.coords, dtype=np.float64)

        # Displacement boundary conditions types and delta values
        self.h_bc_types = model.bc_types
        self.h_bc_values = model.bc_values

        self.h_tip_types = model.tip_types

        # Force boundary conditions types and values
        self.h_force_bc_types = model.force_bc_types
        self.h_force_bc_values = model.force_bc_values

        # Nodal volumes
        self.h_vols = model.V

        # Bond stiffnesses
        self.h_bond_stiffness =  np.ascontiguousarray(model.bond_stiffness, dtype=np.float64)
        self.h_bond_critical_stretch = np.ascontiguousarray(model.bond_critical_stretch, dtype=np.float64)

        # Displacements
        # First order accurate displacement
        self.h_un1 = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)
        # Second order accurate displacement
        self.h_un2 = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)

        # Forces
        self.h_k1dn = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)
        self.h_k2dn = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)

        # Damage vector
        self.h_damage = np.empty(model.nnodes).astype(np.float64)

        # Time step size
        self.h_dt = np.float64(model.dt)

        # Errors
        self.h_errorn = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)

        # Build OpenCL data structures

        # Read only
        self.d_coords = cl.Buffer(self.context,
                             cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                             hostbuf=self.h_coords)
        self.d_bc_types = cl.Buffer(self.context,
                              cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                              hostbuf=self.h_bc_types)
        self.d_bc_values = cl.Buffer(self.context,
                               cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                               hostbuf=self.h_bc_values)
        self.d_force_bc_types = cl.Buffer(self.context,
                              cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                              hostbuf=self.h_force_bc_types)
        self.d_force_bc_values = cl.Buffer(self.context,
                               cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                               hostbuf=self.h_force_bc_values)
        self.d_vols = cl.Buffer(self.context,
                           cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                           hostbuf=self.h_vols)
        self.d_bond_stiffness = cl.Buffer(self.context,
                           cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                           hostbuf=self.h_bond_stiffness)
        self.d_bond_critical_stretch = cl.Buffer(self.context,
                           cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                           hostbuf=self.h_bond_critical_stretch)
        self.d_horizons_lengths = cl.Buffer(
                self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                hostbuf=self.h_horizons_lengths)

        # Read and write
        self.d_horizons = cl.Buffer(
                self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
                hostbuf=self.h_horizons)
        self.d_un1 = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_un1.nbytes)
        self.d_un2 = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_un2.nbytes)
        self.d_k1dn = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_k1dn.nbytes)
        self.d_k2dn = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_k2dn.nbytes)
        # Write only
        self.d_damage = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, self.h_damage.nbytes)
        self.d_errorn = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, self.h_errorn.nbytes)
        # Initialize kernel parameters
        self.cl_kernel_calc_bond_force.set_scalar_arg_dtypes(
            [None, None, None, None, None, None, None, None])
        self.cl_kernel_displacement_update.set_scalar_arg_dtypes(
            [None, None, None, None, None, None])
        self.cl_kernel_partial_displacement_update.set_scalar_arg_dtypes(
            [None, None, None, None])
        self.cl_kernel_check_bonds.set_scalar_arg_dtypes([None, None, None, None])
        self.cl_kernel_calculate_damage.set_scalar_arg_dtypes([None, None, None])
        self.cl_kernel_check_error.set_scalar_arg_dtypes([None, None, None])
    def __call__(self):
        """
        Conduct one iteration of the integrator.

        :arg u: A (`nnodes`, 3) array containing the displacements of all
            nodes.
        :type u: :class:`numpy.ndarray`
        :arg f: A (`nnodes`, 3) array containing the components of the force
            acting on each node.
        :type f: :class:`numpy.ndarray`

        :returns: The new displacements after integration.
        :rtype: :class:`numpy.ndarray`
        """

    def runtime(self, model):
        # store the last stable displacements and horizons
        self.d_un = self.d_un2
        self.d_horizons_stable = self.d_horizons
        # Find k1dn (forces)
        self.cl_kernel_calc_bond_force(self.queue, 
                                          (model.nnodes,), 
                                          None, self.d_k1dn, self.d_un2, self.d_vols, self.d_horizons, self.d_coords, self.d_bond_stiffness, self.d_force_bc_types, self.d_force_bc_values)
        # Find first order accurate displacements
        # Scalars like self.h_dt can live on the host memory
        self.cl_kernel_partial_displacement_update(self.queue, (model.nnodes * model.degrees_freedom,),
                                                     None, self.d_k1dn, self.d_un2, self.d_un1, self.h_dt)
        # Find k2dn (forces)
        self.cl_kernel_calc_bond_force(self.queue, 
                                          (model.nnodes,), 
                                          None, self.d_k2dn, self.d_un1, self.d_vols, self.d_horizons, self.d_coords, self.d_bond_stiffness, self.d_force_bc_types, self.d_force_bc_values)
        # Find second order accurate displacements
        self.cl_kernel_displacement_update(self.queue, 
                                              (model.nnodes * model.degrees_freedom,), 
                                              None, self.d_k1dn, self.d_k2dn, self.d_bc_types, self.d_bc_values, self.d_un2, self.h_dt)
        if self.adapt_time_step(model) == 1:
            # Reset displacements to last stable time-step
            self.d_un2 = self.d_un
        else:
            # Check for broken bonds
            self.cl_kernel_check_bonds(self.queue, 
                                       (model.nnodes, model.max_horizon_length),
                                       None, self.d_horizons, self.d_un2, self.d_coords, self.d_bond_critical_stretch)
    def write(self, model, t, sample):
        """ Write a mesh file for the current timestep
        """
        self.cl_kernel_calculate_damage(self.queue, (model.nnodes,), None, 
                                           self.d_damage, self.d_horizons,
                                           self.d_horizons_lengths)
        cl.enqueue_copy(self.queue, self.h_damage, self.d_damage)
        cl.enqueue_copy(self.queue, self.h_un2, self.d_un2)
        cl.enqueue_copy(self.queue, self.h_k1dn, self.d_k1dn)
        # TODO define a failure criterion, idea: rate of change of damage goes to 0 after it has started increasing
        tip_displacement = 0
        tip_shear_force = 0
        tmp = 0
        for i in range(model.nnodes):
            if self.h_tip_types[i] == 1:
                tmp +=1
                tip_displacement += self.h_un2[i][2]
                tip_shear_force += self.h_k1dn[i][2]
        if tmp != 0:
            tip_displacement /= tmp
        else:
            tip_displacement = None
        vtk.write("output/U_"+"sample" + str(sample) +"t"+str(t) + ".vtk", "Solution time step = "+str(t),
                  model.coords, self.h_damage, self.h_un2)
        #vtk.writeDamage("output/damage_" + str(t)+ "sample" + str(sample) + ".vtk", "Title", self.h_damage)
        return self.h_damage, tip_displacement, tip_shear_force
    def adapt_time_step(self, model, error_size_max=1e-10, error_size_min=3e-20):
        adapt = 0
        # Check for error size
        self.cl_kernel_check_error(self.queue,
                              (model.nnodes * model.degrees_freedom,),
                              None, self.d_un1, self.d_un2, self.d_errorn)
        cl.enqueue_copy(self.queue, self.h_errorn, self.d_errorn)
        error = np.linalg.norm(self.h_errorn, axis=1)
        error = np.mean(error)
        if error > error_size_max:
            self.h_dt /= 2.0
            print('Time step size reduced')
            print('time step is {}s, error size is {}'.format(self.h_dt, error))
            adapt = 1
        elif error < error_size_min:
            self.h_dt *= 1.1
            print('Time step size increased')
            print('time step is {}s, error size is {}'.format(self.h_dt, error))
        return adapt
    def incrementLoad(self, model, load_scale):
        if model.num_force_bc_nodes != 0:
            tmp = -1. * model.max_reaction * load_scale / (model.num_force_bc_nodes)
            # update the host force_bc_values
            self.h_force_bc_values = tmp * np.ones((model.nnodes, model.degrees_freedom), dtype=np.float64)
            #print(h_force_bc_values)
            # update the GPU force_bc_values
            self.d_force_bc_values = cl.Buffer(self.context,
                               cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                               hostbuf=self.h_force_bc_values)

class DormandPrince(Integrator):
    r"""
    4th order Runge-Kutta with adaptive time step

    .. math::
        u(t + \delta t) = u(t) + \delta t f(t) d

    where :math:`u(t)` is the displacement at time :math:`t`, :math:`f(t)` is
    the force at time :math:`t`, :math:`\delta t` is the time step and
    :math:`d` is a dampening factor.
    """
    def __init__(self, model):
        """ Initialise the integration scheme
        """
        
        def output_device_info(device_id):
            sys.stdout.write("Device is ")
            sys.stdout.write(device_id.name)
            if device_id.type == cl.device_type.GPU:
                sys.stdout.write("GPU from ")
            elif device_id.type == cl.device_type.CPU:
                sys.stdout.write("CPU from ")
            else:
                sys.stdout.write("non CPU of GPU processor from ")
            sys.stdout.write(device_id.vendor)
            sys.stdout.write(" with a max of ")
            sys.stdout.write(str(device_id.max_compute_units))
            sys.stdout.write(" compute units\n")
            sys.stdout.flush()

        # Initializing OpenCL
        self.context = cl.create_some_context()
        self.queue = cl.CommandQueue(self.context)   

        # Print out device info
        output_device_info(self.context.devices[0])

        # Build the OpenCL program from file
        kernelsource = open(pathlib.Path(__file__).parent.absolute() / "kernels/opencl_dormand_prince.cl").read()
        SEP = " "

        options_string = (
            "-cl-fast-relaxed-math" + SEP
            + "-DPD_DPN_NODE_NO=" + str(model.degrees_freedom * model.nnodes) + SEP
            + "-DPD_NODE_NO=" + str(model.nnodes) + SEP
            + "-DMAX_HORIZON_LENGTH=" + str(model.max_horizon_length) + SEP
            + "-DPD_RHO=" + str(model.density) + SEP
            + "-DPD_ETA=" + str(model.damping) + SEP)

        program = cl.Program(self.context, kernelsource).build([options_string])
        self.cl_kernel_calc_bond_force = program.CalcBondForce
        self.cl_kernel_displacement_update = program.UpdateDisplacement
        self.cl_kernel_partial_displacement_update = program.PartialUpdateDisplacement
        self.cl_kernel_partial_displacement_update2 = program.PartialUpdateDisplacement2
        self.cl_kernel_partial_displacement_update3 = program.PartialUpdateDisplacement3
        self.cl_kernel_partial_displacement_update4 = program.PartialUpdateDisplacement4
        self.cl_kernel_partial_displacement_update5 = program.PartialUpdateDisplacement5
        self.cl_kernel_partial_displacement_update6 = program.PartialUpdateDisplacement6
        self.cl_kernel_check_error = program.CheckError

        self.cl_kernel_check_bonds = program.CheckBonds
        self.cl_kernel_calculate_damage = program.CalculateDamage

        # Set initial values in host memory

        # horizons and horizons lengths
        self.h_horizons = model.horizons
        self.h_horizons_lengths = model.horizons_lengths
        print(self.h_horizons_lengths)
        print(self.h_horizons)
        print("shape horizons lengths", self.h_horizons_lengths.shape)
        print("shape horizons lengths", self.h_horizons.shape)
        print(self.h_horizons_lengths.dtype, "dtype")

        # Nodal coordinates
        self.h_coords = np.ascontiguousarray(model.coords, dtype=np.float64)

        # Displacement boundary conditions types and delta values
        self.h_bc_types = model.bc_types
        self.h_bc_values = model.bc_values

        self.h_tip_types = model.tip_types

        # Force boundary conditions types and values
        self.h_force_bc_types = model.force_bc_types
        self.h_force_bc_values = model.force_bc_values

        # Nodal volumes
        self.h_vols = model.V

        # Bond stiffnesses
        self.h_bond_stiffness =  np.ascontiguousarray(model.bond_stiffness, dtype=np.float64)
        self.h_bond_critical_stretch = np.ascontiguousarray(model.bond_critical_stretch, dtype=np.float64)

        # Displacements
        self.h_un4 = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)
        self.h_un5 = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)
        self.h_un5_1 = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)
        self.h_un_temp = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)

        # Forces
        self.h_k1dn = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)
        self.h_k2dn = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)
        self.h_k3dn = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)
        self.h_k4dn = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)
        self.h_k5dn = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)
        self.h_k6dn = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)
        self.h_k7dn = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)

        # Damage vector
        self.h_damage = np.empty(model.nnodes).astype(np.float64)

        # Time step size
        self.h_dt = np.float64(model.dt)

        # Errors
        self.h_errorn = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)

        # Build OpenCL data structures

        # Read only
        self.d_coords = cl.Buffer(self.context,
                             cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                             hostbuf=self.h_coords)
        self.d_bc_types = cl.Buffer(self.context,
                              cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                              hostbuf=self.h_bc_types)
        self.d_bc_values = cl.Buffer(self.context,
                               cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                               hostbuf=self.h_bc_values)
        self.d_force_bc_types = cl.Buffer(self.context,
                              cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                              hostbuf=self.h_force_bc_types)
        self.d_force_bc_values = cl.Buffer(self.context,
                               cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                               hostbuf=self.h_force_bc_values)
        self.d_vols = cl.Buffer(self.context,
                           cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                           hostbuf=self.h_vols)
        self.d_bond_stiffness = cl.Buffer(self.context,
                           cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                           hostbuf=self.h_bond_stiffness)
        self.d_bond_critical_stretch = cl.Buffer(self.context,
                           cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                           hostbuf=self.h_bond_critical_stretch)
        self.d_horizons_lengths = cl.Buffer(
                self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                hostbuf=self.h_horizons_lengths)

        # Read and write
        self.d_horizons = cl.Buffer(
                self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
                hostbuf=self.h_horizons)
        self.d_un4 = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_un4.nbytes)
        self.d_un5 = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_un5.nbytes)
        self.d_un5_1= cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_un5_1.nbytes)
        self.d_un_temp = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_un_temp.nbytes)
        self.d_k1dn = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_k1dn.nbytes)
        self.d_k2dn = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_k2dn.nbytes)
        self.d_k3dn = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_k3dn.nbytes)
        self.d_k4dn = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_k4dn.nbytes)
        self.d_k5dn = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_k5dn.nbytes)
        self.d_k6dn = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_k6dn.nbytes)
        self.d_k7dn = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_k7dn.nbytes)
        # Write only
        self.d_damage = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, self.h_damage.nbytes)
        self.d_errorn = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, self.h_errorn.nbytes)
        # Initialize kernel parameters
        self.cl_kernel_calc_bond_force.set_scalar_arg_dtypes(
            [None, None, None, None, None, None, None, None])
        self.cl_kernel_displacement_update.set_scalar_arg_dtypes(
            [None, None, None, None, None])
        self.cl_kernel_partial_displacement_update.set_scalar_arg_dtypes(
            [None, None, None, None])
        self.cl_kernel_partial_displacement_update2.set_scalar_arg_dtypes(
            [None, None, None, None, None])
        self.cl_kernel_partial_displacement_update3.set_scalar_arg_dtypes(
            [None, None, None, None, None, None])
        self.cl_kernel_partial_displacement_update4.set_scalar_arg_dtypes(
            [None, None, None, None, None, None, None])
        self.cl_kernel_partial_displacement_update5.set_scalar_arg_dtypes(
            [None, None, None, None, None, None, None, None])
        self.cl_kernel_partial_displacement_update6.set_scalar_arg_dtypes(
            [None, None, None, None, None, None, None, None])
        self.cl_kernel_check_error.set_scalar_arg_dtypes([None, None, None, None, None, None, None, None, None, None])
        self.cl_kernel_check_bonds.set_scalar_arg_dtypes([None, None, None, None])
        self.cl_kernel_calculate_damage.set_scalar_arg_dtypes([None, None, None])

    def __call__(self):
        """
        Conduct one iteration of the integrator.

        :arg u: A (`nnodes`, 3) array containing the displacements of all
            nodes.
        :type u: :class:`numpy.ndarray`
        :arg f: A (`nnodes`, 3) array containing the components of the force
            acting on each node.
        :type f: :class:`numpy.ndarray`

        :returns: The new displacements after integration.
        :rtype: :class:`numpy.ndarray`
        """

    def runtime(self, model):
        # Find k1dn (forces)
        self.cl_kernel_calc_bond_force(self.queue, 
                                          (model.nnodes,), 
                                          None, self.d_k1dn, self.d_un5, self.d_vols, self.d_horizons, self.d_coords, self.d_bond_stiffness, self.d_force_bc_types, self.d_force_bc_values)
        # Partial update of displacements
        self.cl_kernel_partial_displacement_update(self.queue, (model.nnodes * model.degrees_freedom,),
                                                     None, self.d_k1dn, self.d_un5, self.d_un_temp, self.h_dt)
        # Find k2dn (forces)
        self.cl_kernel_calc_bond_force(self.queue, 
                                          (model.nnodes,), 
                                          None, self.d_k2dn, self.d_un_temp, self.d_vols, self.d_horizons, self.d_coords, self.d_bond_stiffness, self.d_force_bc_types, self.d_force_bc_values)
        # Partial update of displacements 2
        self.cl_kernel_partial_displacement_update2(self.queue, (model.nnodes * model.degrees_freedom,),
                                                     None, self.d_k1dn, self.d_k2dn, self.d_un5, self.d_un_temp, self.h_dt)
        # Find k3dn (forces)
        self.cl_kernel_calc_bond_force(self.queue, 
                                          (model.nnodes,),
                                          None, self.d_k3dn, self.d_un_temp, self.d_vols, self.d_horizons, self.d_coords, self.d_bond_stiffness, self.d_force_bc_types, self.d_force_bc_values)
        # Partial update of displacements 3
        self.cl_kernel_partial_displacement_update3(self.queue, (model.nnodes * model.degrees_freedom,),
                                                     None, self.d_k1dn, self.d_k2dn, self.d_k3dn, self.d_un5, self.d_un_temp, self.h_dt)
        # Find k4dn (forces)
        self.cl_kernel_calc_bond_force(self.queue, 
                                          (model.nnodes,),
                                          None, self.d_k4dn, self.d_un_temp, self.d_vols, self.d_horizons, self.d_coords, self.d_bond_stiffness, self.d_force_bc_types, self.d_force_bc_values)
        # Partial update of displacements 4
        self.cl_kernel_partial_displacement_update4(self.queue, (model.nnodes * model.degrees_freedom,),
                                                     None, self.d_k1dn, self.d_k2dn, self.d_k3dn, self.d_k4dn, self.d_un5, self.d_un_temp, self.h_dt)
        # Find k5dn (forces)
        self.cl_kernel_calc_bond_force(self.queue, 
                                          (model.nnodes,),
                                          None, self.d_k5dn, self.d_un_temp, self.d_vols, self.d_horizons, self.d_coords, self.d_bond_stiffness, self.d_force_bc_types, self.d_force_bc_values)
        # Partial update of displacements 5
        self.cl_kernel_partial_displacement_update5(self.queue, (model.nnodes * model.degrees_freedom,),
                                                     None, self.d_k1dn, self.d_k2dn, self.d_k3dn, self.d_k4dn, self.d_k5dn, self.d_un5, self.d_un_temp, self.h_dt)
        # Find k6dn (forces)
        self.cl_kernel_calc_bond_force(self.queue, 
                                          (model.nnodes,),
                                          None, self.d_k6dn, self.d_un_temp, self.d_vols, self.d_horizons, self.d_coords, self.d_bond_stiffness, self.d_force_bc_types, self.d_force_bc_values)
        # Partial update of displacements 6
        self.cl_kernel_partial_displacement_update6(self.queue, (model.nnodes * model.degrees_freedom,),
                                                     None, self.d_k1dn, self.d_k3dn, self.d_k4dn, self.d_k5dn, self.d_k6dn, self.d_un5, self.d_un_temp, self.h_dt)
        # Find k7dn (forces)
        self.cl_kernel_calc_bond_force(self.queue, 
                                          (model.nnodes,),
                                          None, self.d_k7dn, self.d_un_temp, self.d_vols, self.d_horizons, self.d_coords, self.d_bond_stiffness, self.d_force_bc_types, self.d_force_bc_values)
        if self.adapt_time_step(model) == 1:
            pass
        else:
            # Full update of displacements
            self.cl_kernel_displacement_update(self.queue, 
                                              (model.nnodes * model.degrees_freedom,), 
                                              None, self.d_bc_types, self.d_bc_values, self.d_un_temp, self.d_un5, self.h_dt)
            # Check for broken bonds
            self.cl_kernel_check_bonds(self.queue, 
                                       (model.nnodes, model.max_horizon_length),
                                       None, self.d_horizons, self.d_un5, self.d_coords, self.d_bond_critical_stretch)
    def write(self, model, t, sample):
        """ Write a mesh file for the current timestep
        """
        self.cl_kernel_calculate_damage(self.queue, (model.nnodes,), None, 
                                           self.d_damage, self.d_horizons,
                                           self.d_horizons_lengths)
        cl.enqueue_copy(self.queue, self.h_damage, self.d_damage)
        cl.enqueue_copy(self.queue, self.h_un5, self.d_un5)
        # TODO define a failure criterion, idea: rate of change of damage goes to 0 after it has started increasing
        tip_displacement = 0
        tmp = 0
        for i in range(model.nnodes):
            if self.h_tip_types[i] == 1:
                tmp +=1
                tip_displacement += self.h_un5[i][2]
        if tmp != 0:
            tip_displacement /= tmp
        else:
            tip_displacement = None
        vtk.write("output/U_"+"sample" + str(sample) +"t"+str(t) + ".vtk", "Solution time step = "+str(t),
                  model.coords, self.h_damage, self.h_un5)
        #vtk.writeDamage("output/damage_" + str(t)+ "sample" + str(sample) + ".vtk", "Title", self.h_damage)
        return self.h_damage, tip_displacement
    def adapt_time_step(self, model, error_size_max=1e-7, error_size_min=1e-10):
        adapt = 0
        # Check for error size
        self.cl_kernel_check_error(self.queue,
                              (model.nnodes * model.degrees_freedom,),
                              None, self.d_k1dn, self.d_k3dn, self.d_k4dn, self.d_k5dn, self.d_k6dn, self.d_k7dn, self.d_un_temp, self.d_un5, self.d_errorn, self.h_dt)
        cl.enqueue_copy(self.queue, self.h_errorn, self.d_errorn)
        error = np.linalg.norm(self.h_errorn, axis=1)
        error = np.mean(error)
        if error > error_size_max:
            self.h_dt /= 1.1
            print('Time step size reduced')
            print('time step is {}s, error size is {}'.format(self.h_dt, error))
            adapt = 1
        elif error < error_size_min:
            self.h_dt *= 1.1
            print('Time step size increased')
            print('time step is {}s, error size is {}'.format(self.h_dt, error))
        return adapt
    def incrementLoad(self, model, load_scale):
        if model.num_force_bc_nodes != 0:
            tmp = -1. * model.max_reaction * load_scale / (model.num_force_bc_nodes)
            # update the host force_bc_values
            self.h_force_bc_values = tmp * np.ones((model.nnodes, model.degrees_freedom), dtype=np.float64)
            #print(h_force_bc_values)
            # update the GPU force_bc_values
            self.d_force_bc_values = cl.Buffer(self.context,
                               cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                               hostbuf=self.h_force_bc_values)
class EulerStochastic(Integrator):
    r"""
    Stochastic Euler integrator for quasi-static loading, using OpenCL kernels.

    The Euler method is a first-order numerical integration method. The
    integration is given by,

    .. math::
        u(t + \delta t) = u(t) + \delta t f(t) d

    where :math:`u(t)` is the displacement at time :math:`t`, :math:`f(t)` is
    the force at time :math:`t`, :math:`\delta t` is the time step and
    :math:`d` is a dampening factor.
    """
    def __init__(self, model):
        """ Initialise the integration scheme
        """
        
        def output_device_info(device_id):
            sys.stdout.write("Device is ")
            sys.stdout.write(device_id.name)
            if device_id.type == cl.device_type.GPU:
                sys.stdout.write("GPU from ")
            elif device_id.type == cl.device_type.CPU:
                sys.stdout.write("CPU from ")
            else:
                sys.stdout.write("non CPU of GPU processor from ")
            sys.stdout.write(device_id.vendor)
            sys.stdout.write(" with a max of ")
            sys.stdout.write(str(device_id.max_compute_units))
            sys.stdout.write(" compute units\n")
            sys.stdout.flush()

        # Initializing OpenCL
        self.context = cl.create_some_context()
        self.queue = cl.CommandQueue(self.context)   

        # Print out device info
        output_device_info(self.context.devices[0])

        # Build the OpenCL program from file
        kernelsource = open(pathlib.Path(__file__).parent.absolute() / "kernels/opencl_euler_stochastic.cl").read()
        SEP = " "

        options_string = (
            "-cl-fast-relaxed-math" + SEP
            + "-DPD_DPN_NODE_NO=" + str(model.degrees_freedom * model.nnodes) + SEP
            + "-DPD_NODE_NO=" + str(model.nnodes) + SEP
            + "-DMAX_HORIZON_LENGTH=" + str(model.max_horizon_length) + SEP
            + "-DPD_DT=" + str(model.dt) + SEP)

        program = cl.Program(self.context, kernelsource).build([options_string])
        self.cl_kernel_time_marching_1 = program.TimeMarching1
        self.cl_kernel_time_marching_2 = program.TimeMarching2
        self.cl_kernel_check_bonds = program.CheckBonds
        self.cl_kernel_calculate_damage = program.CalculateDamage
        self.cl_kernel_mmul = program.mmul

        # Set initial values in host memory

        # horizons and horizons lengths
        self.h_horizons = model.horizons
        self.h_horizons_lengths = model.horizons_lengths

        # Nodal coordinates
        self.h_coords = np.ascontiguousarray(model.coords, dtype=np.float64)

        # Displacement boundary conditions types and delta values
        self.h_bc_types = model.bc_types
        self.h_bc_values = model.bc_values

        self.h_tip_types = model.tip_types

        # Force boundary conditions types and values
        self.h_force_bc_types = model.force_bc_types
        self.h_force_bc_values = model.force_bc_values

        # Nodal volumes
        self.h_vols = model.V

        # Bond stiffnesses
        self.h_bond_stiffness =  np.ascontiguousarray(model.bond_stiffness, dtype=np.float64)
        self.h_bond_critical_stretch = np.ascontiguousarray(model.bond_critical_stretch, dtype=np.float64)

        # Displacements
        self.h_un = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)
        self.h_pn = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)

        # Forces
        self.h_udn = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)
        self.h_udn1 = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)

        # Damage vector
        self.h_damage = np.empty(model.nnodes).astype(np.float64)

        # Covariance matrix
        self.h_K = model.K

        # Build OpenCL data structures

        # Read only
        self.d_coords = cl.Buffer(self.context,
                             cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                             hostbuf=self.h_coords)
        self.d_bc_types = cl.Buffer(self.context,
                              cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                              hostbuf=self.h_bc_types)
        self.d_bc_values = cl.Buffer(self.context,
                               cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                               hostbuf=self.h_bc_values)
        self.d_force_bc_types = cl.Buffer(self.context,
                              cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                              hostbuf=self.h_force_bc_types)
        self.d_force_bc_values = cl.Buffer(self.context,
                               cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                               hostbuf=self.h_force_bc_values)
        self.d_vols = cl.Buffer(self.context,
                           cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                           hostbuf=self.h_vols)
        self.d_bond_stiffness = cl.Buffer(self.context,
                           cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                           hostbuf=self.h_bond_stiffness)
        self.d_bond_critical_stretch = cl.Buffer(self.context,
                           cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                           hostbuf=self.h_bond_critical_stretch)
        self.d_horizons_lengths = cl.Buffer(
                self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                hostbuf=self.h_horizons_lengths)
        self.d_K = cl.Buffer(
                self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                hostbuf=self.h_K)
        

        # Read and write
        self.d_horizons = cl.Buffer(
                self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
                hostbuf=self.h_horizons)
        self.d_un = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_un.nbytes)
        self.d_udn = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_udn1.nbytes)
        self.d_udn1 = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_udn1.nbytes)
        # Write only
        self.d_damage = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, self.h_damage.nbytes)
        # Initialize kernel parameters
        self.cl_kernel_time_marching_1.set_scalar_arg_dtypes(
            [None, None, None, None, None, None])
        self.cl_kernel_time_marching_2.set_scalar_arg_dtypes(
            [None, None, None, None, None, None, None, None])
        self.cl_kernel_check_bonds.set_scalar_arg_dtypes([None, None, None, None])
        self.cl_kernel_calculate_damage.set_scalar_arg_dtypes([None, None, None])
    def __call__(self):
        """
        Conduct one iteration of the integrator.

        :arg u: A (`nnodes`, 3) array containing the displacements of all
            nodes.
        :type u: :class:`numpy.ndarray`
        :arg f: A (`nnodes`, 3) array containing the components of the force
            acting on each node.
        :type f: :class:`numpy.ndarray`

        :returns: The new displacements after integration.
        :rtype: :class:`numpy.ndarray`
        """
    def reset(self, model, steps):
        # Displacements
        self.h_un = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)
        self.h_pn = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)

        # Forces
        self.h_udn = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)
        self.h_udn1 = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)

        # Damage vector
        self.h_damage = np.empty(model.nnodes).astype(np.float64)

        # Sample random noise vector
        self.h_pn = noise(model.C, model.K, model.nnodes, steps)

        # Covariance matrix
        self.h_K = model.K

        # Build OpenCL data structures

        # Read only
# =============================================================================
#         self.d_coords = cl.Buffer(self.context,
#                              cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
#                              hostbuf=self.h_coords)
#         self.d_bc_types = cl.Buffer(self.context,
#                               cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
#                               hostbuf=self.h_bc_types)
#         self.d_bc_values = cl.Buffer(self.context,
#                                cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
#                                hostbuf=self.h_bc_values)
#         self.d_force_bc_types = cl.Buffer(self.context,
#                               cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
#                               hostbuf=self.h_force_bc_types)
#         self.d_force_bc_values = cl.Buffer(self.context,
#                                cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
#                                hostbuf=self.h_force_bc_values)
#         self.d_vols = cl.Buffer(self.context,
#                            cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
#                            hostbuf=self.h_vols)
#         self.d_bond_stiffness = cl.Buffer(self.context,
#                            cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
#                            hostbuf=self.h_bond_stiffness)
#         self.d_bond_critical_stretch = cl.Buffer(self.context,
#                            cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
#                            hostbuf=self.h_bond_critical_stretch)
#         self.d_horizons_lengths = cl.Buffer(
#                 self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
#                 hostbuf=self.h_horizons_lengths)
# =============================================================================
        self.d_pn = cl.Buffer(self.context,
                             cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                             hostbuf=self.h_pn)
        self.d_K = cl.Buffer(
                self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                hostbuf=self.h_K)

        # Read and write
        self.d_horizons = cl.Buffer(
                self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
                hostbuf=self.h_horizons)
        self.d_un = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_un.nbytes)
        self.d_udn = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_udn1.nbytes)
        self.d_udn1 = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_udn1.nbytes)
        # Write only
        self.d_damage = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, self.h_damage.nbytes)
        # Initialize kernel parameters
    def runtime(self, model, step):
        # Time marching Part 1
        self.cl_kernel_time_marching_1(self.queue, (model.degrees_freedom * model.nnodes,),
                                  None, self.d_udn, self.d_un, self.d_pn, self.d_bc_types,
                                  self.d_bc_values, np.intc(step))
        # Time marching Part 2: Calc forces
        self.cl_kernel_time_marching_2(self.queue, (model.nnodes,), None, self.d_udn,
                                  self.d_un, self.d_vols, self.d_horizons, self.d_coords, self.d_bond_stiffness, self.d_force_bc_types, self.d_force_bc_values)
        # Matrix multiplication of forces
        self.cl_kernel_mmul(self.queue, (model.nnodes, model.degrees_freedom), None, self.d_K, self.d_udn, self.d_udn1)
        #cl.enqueue_copy(self.queue, self.h_udn, self.d_udn)
        #cl.enqueue_copy(self.queue, self.h_udn1, self.d_udn1)
        #print(self.h_udn1, 'udn1')
        #print(self.h_udn - self.h_udn1, 'here')
        # Check for broken bonds
        self.cl_kernel_check_bonds(self.queue,
                              (model.nnodes, model.max_horizon_length),
                              None, self.d_horizons, self.d_un, self.d_coords, self.d_bond_critical_stretch)
    def write(self, model, t, sample):
        """ Write a mesh file for the current timestep
        """
        self.cl_kernel_calculate_damage(self.queue, (model.nnodes,), None, 
                                           self.d_damage, self.d_horizons,
                                           self.d_horizons_lengths)
        cl.enqueue_copy(self.queue, self.h_damage, self.d_damage)
        #vtk.write("output/U_"+"sample" + str(sample) +"t"+str(t) + ".vtk", "Solution time step = "+str(t),
                  #model.coords, self.h_damage, self.h_un)
        #vtk.writeDamage("output/damage_" + str(t)+ "sample" + str(sample) + ".vtk", "Title", self.h_damage)
        return self.h_damage

    def incrementLoad(self, model, load_scale):
        if model.num_force_bc_nodes != 0:
            tmp = -1. * model.max_reaction * load_scale / (model.num_force_bc_nodes)
            # update the host force_bc_values
            self.h_force_bc_values = tmp * np.ones((model.nnodes, model.degrees_freedom), dtype=np.float64)
            #print(h_force_bc_values)
            # update the GPU force_bc_values
            self.d_force_bc_values = cl.Buffer(self.context,
                               cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                               hostbuf=self.h_force_bc_values)
class EulerOpenCLFEM(Integrator):
    r"""
    Static Euler integrator for quasi-static loading, using OpenCL kernels.

    The Euler method is a first-order numerical integration method. The
    integration is given by,

    .. math::
        u(t + \delta t) = u(t) + \delta t f(t) d

    where :math:`u(t)` is the displacement at time :math:`t`, :math:`f(t)` is
    the force at time :math:`t`, :math:`\delta t` is the time step and
    :math:`d` is a dampening factor.
    """
    def __init__(self, model):
        """ Initialise the integration scheme
        """
        def output_device_info(device_id):
            sys.stdout.write("Device is ")
            sys.stdout.write(device_id.name)
            if device_id.type == cl.device_type.GPU:
                sys.stdout.write("GPU from ")
            elif device_id.type == cl.device_type.CPU:
                sys.stdout.write("CPU from ")
            else:
                sys.stdout.write("non CPU of GPU processor from ")
            sys.stdout.write(device_id.vendor)
            sys.stdout.write(" with a max of ")
            sys.stdout.write(str(device_id.max_compute_units))
            sys.stdout.write(" compute units\n")
            sys.stdout.flush()

        # Initializing OpenCL
        self.context = cl.create_some_context()
        self.queue = cl.CommandQueue(self.context)   

        # Print out device info
        output_device_info(self.context.devices[0])

        # Build the OpenCL program from file
        kernelsource = open(pathlib.Path(__file__).parent.absolute() / "kernels/opencl_euler.cl").read()
        SEP = " "

        options_string = (
            "-cl-fast-relaxed-math" + SEP
            + "-DPD_DPN_NODE_NO=" + str(model.degrees_freedom * model.nnodes) + SEP
            + "-DPD_NODE_NO=" + str(model.nnodes) + SEP
            + "-DMAX_HORIZON_LENGTH=" + str(model.max_horizon_length) + SEP
            + "-DPD_DT=" + str(model.dt) + SEP)

        from pathlib import Path
        print(Path.cwd())
        program = cl.Program(self.context, kernelsource).build([options_string])
        self.cl_kernel_time_marching_1 = program.TimeMarching1
        self.cl_kernel_time_marching_2 = program.TimeMarching2
        self.cl_kernel_check_bonds = program.CheckBonds
        self.cl_kernel_calculate_damage = program.CalculateDamage

        # Set initial values in host memory

        # horizons and horizons lengths
        self.h_horizons = model.horizons
        self.h_horizons_lengths = model.horizons_lengths
        print(self.h_horizons_lengths)
        print(self.h_horizons)
        print("shape horizons lengths", self.h_horizons_lengths.shape)
        print("shape horizons lengths", self.h_horizons.shape)
        print(self.h_horizons_lengths.dtype, "dtype")

        # Nodal coordinates
        self.h_coords = np.ascontiguousarray(model.coords, dtype=np.float64)

        # Displacement boundary conditions types and delta values
        self.h_bc_types = model.bc_types
        self.h_bc_values = model.bc_values

        self.h_tip_types = model.tip_types

        # Force boundary conditions types and values
        self.h_force_bc_types = model.force_bc_types
        self.h_force_bc_values = model.force_bc_values

        # Nodal volumes
        self.h_vols = model.V

        # Bond stiffnesses
        self.h_bond_stiffness =  np.ascontiguousarray(model.bond_stiffness, dtype=np.float64)
        self.h_bond_critical_stretch = np.ascontiguousarray(model.bond_critical_stretch, dtype=np.float64)

        # Displacements
        self.h_un = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)

        # Forces
        self.h_udn1 = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)

        # Damage vector
        self.h_damage = np.empty(model.nnodes).astype(np.float64)

        if model.v == True:
            # Print the dtypes
            print("horizons", self.h_horizons.dtype)
            print("horizons_length", self.h_horizons_lengths.dtype)
            print("force_bc_types", self.h_bc_types.dtype)
            print("force_bc_values", self.h_bc_values.dtype)
            print("bc_types", self.h_bc_types.dtype)
            print("bc_values", self.h_bc_values.dtype)
            print("coords", self.h_coords.dtype)
            print("vols", self.h_vols.dtype)
            print("un", self.h_un.dtype)
            print("udn1", self.h_udn1.dtype)
            print("damage", self.h_damage.dtype)

        # Build OpenCL data structures
        # Read only
        self.d_coords = cl.Buffer(self.context,
                             cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                             hostbuf=self.h_coords)
        self.d_bc_types = cl.Buffer(self.context,
                              cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                              hostbuf=self.h_bc_types)
        self.d_bc_values = cl.Buffer(self.context,
                               cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                               hostbuf=self.h_bc_values)
        self.d_force_bc_types = cl.Buffer(self.context,
                              cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                              hostbuf=self.h_force_bc_types)
        self.d_force_bc_values = cl.Buffer(self.context,
                               cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                               hostbuf=self.h_force_bc_values)
        self.d_vols = cl.Buffer(self.context,
                           cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                           hostbuf=self.h_vols)
        self.d_bond_stiffness = cl.Buffer(self.context,
                           cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                           hostbuf=self.h_bond_stiffness)
        self.d_bond_critical_stretch = cl.Buffer(self.context,
                           cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                           hostbuf=self.h_bond_critical_stretch)
        self.d_horizons_lengths = cl.Buffer(
                self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                hostbuf=self.h_horizons_lengths)

        # Read and write
        self.d_horizons = cl.Buffer(
                self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
                hostbuf=self.h_horizons)
        self.d_un = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_un.nbytes)
        self.d_udn1 = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_udn1.nbytes)

        # Write only
        self.d_damage = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, self.h_damage.nbytes)
        # Initialize kernel parameters
        self.cl_kernel_time_marching_1.set_scalar_arg_dtypes(
            [None, None, None, None])
        self.cl_kernel_time_marching_2.set_scalar_arg_dtypes(
            [None, None, None, None, None, None, None, None])
        self.cl_kernel_check_bonds.set_scalar_arg_dtypes([None, None, None, None])
        self.cl_kernel_calculate_damage.set_scalar_arg_dtypes([None, None, None])

class HeunEulerNew(Integrator):
    r"""
    Static Euler integrator for quasi-static loading, using OpenCL kernels.

    The Euler method is a first-order numerical integration method. The
    integration is given by,

    .. math::
        u(t + \delta t) = u(t) + \delta t f(t) d

    where :math:`u(t)` is the displacement at time :math:`t`, :math:`f(t)` is
    the force at time :math:`t`, :math:`\delta t` is the time step and
    :math:`d` is a dampening factor.
    """
    def __init__(self, model):
        """ Initialise the integration scheme
        """
        
        def output_device_info(device_id):
            sys.stdout.write("Device is ")
            sys.stdout.write(device_id.name)
            if device_id.type == cl.device_type.GPU:
                sys.stdout.write("GPU from ")
            elif device_id.type == cl.device_type.CPU:
                sys.stdout.write("CPU from ")
            else:
                sys.stdout.write("non CPU of GPU processor from ")
            sys.stdout.write(device_id.vendor)
            sys.stdout.write(" with a max of ")
            sys.stdout.write(str(device_id.max_compute_units))
            sys.stdout.write(" compute units\n")
            sys.stdout.flush()

        # Initializing OpenCL
        self.context = cl.create_some_context()
        self.queue = cl.CommandQueue(self.context)   

        # Print out device info
        output_device_info(self.context.devices[0])

        # Build the OpenCL program from file
        kernelsource = open(pathlib.Path(__file__).parent.absolute() / "kernels/opencl_heun_euler_new.cl").read()
        SEP = " "

        options_string = (
            "-cl-fast-relaxed-math" + SEP
            + "-DPD_DPN_NODE_NO=" + str(model.degrees_freedom * model.nnodes) + SEP
            + "-DPD_NODE_NO=" + str(model.nnodes) + SEP
            + "-DMAX_HORIZON_LENGTH=" + str(model.max_horizon_length) + SEP
            + "-DPD_RHO=" + str(model.density) + SEP
            + "-DPD_ETA=" + str(model.damping) + SEP)

        program = cl.Program(self.context, kernelsource).build([options_string])
        self.cl_kernel_calc_bond_force = program.CalcBondForce
        self.cl_kernel_displacement_update = program.UpdateDisplacement
        self.cl_kernel_partial_displacement_update = program.PartialUpdateDisplacement
        self.cl_kernel_check_bonds = program.CheckBonds
        self.cl_kernel_calculate_damage = program.CalculateDamage
        self.cl_kernel_check_error = program.CheckError

        # Set initial values in host memory

        # horizons and horizons lengths
        self.h_horizons = model.horizons
        self.h_horizons_lengths = model.horizons_lengths
        print(self.h_horizons_lengths)
        print(self.h_horizons)
        print("shape horizons lengths", self.h_horizons_lengths.shape)
        print("shape horizons lengths", self.h_horizons.shape)
        print(self.h_horizons_lengths.dtype, "dtype")

        # Nodal coordinates
        self.h_coords = np.ascontiguousarray(model.coords, dtype=np.float64)

        # Displacement boundary conditions types and delta values
        self.h_bc_types = model.bc_types
        self.h_bc_values = model.bc_values

        self.h_tip_types = model.tip_types

        # Force boundary conditions types and values
        self.h_force_bc_types = model.force_bc_types
        self.h_force_bc_values = model.force_bc_values

        # Nodal volumes
        self.h_vols = model.V

        # Bond stiffnesses
        self.h_bond_stiffness =  np.ascontiguousarray(model.bond_stiffness, dtype=np.float64)
        self.h_bond_critical_stretch = np.ascontiguousarray(model.bond_critical_stretch, dtype=np.float64)

        # Displacements
        # First order accurate displacement
        self.h_un1 = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)
        # Second order accurate displacement
        self.h_un2 = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)
        # Second order accurate displacement update delta
        self.h_un2_1 = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)

        # Forces
        self.h_k1dn = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)
        self.h_k2dn = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)

        # Damage vector
        self.h_damage = np.empty(model.nnodes).astype(np.float64)

        # Time step size
        self.h_dt = np.float64(model.dt)

        # Errors
        self.h_errorn = np.empty((model.nnodes, model.degrees_freedom), dtype=np.float64)

        # Build OpenCL data structures

        # Read only
        self.d_coords = cl.Buffer(self.context,
                             cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                             hostbuf=self.h_coords)
        self.d_bc_types = cl.Buffer(self.context,
                              cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                              hostbuf=self.h_bc_types)
        self.d_bc_values = cl.Buffer(self.context,
                               cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                               hostbuf=self.h_bc_values)
        self.d_force_bc_types = cl.Buffer(self.context,
                              cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                              hostbuf=self.h_force_bc_types)
        self.d_force_bc_values = cl.Buffer(self.context,
                               cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                               hostbuf=self.h_force_bc_values)
        self.d_vols = cl.Buffer(self.context,
                           cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                           hostbuf=self.h_vols)
        self.d_bond_stiffness = cl.Buffer(self.context,
                           cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                           hostbuf=self.h_bond_stiffness)
        self.d_bond_critical_stretch = cl.Buffer(self.context,
                           cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                           hostbuf=self.h_bond_critical_stretch)
        self.d_horizons_lengths = cl.Buffer(
                self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                hostbuf=self.h_horizons_lengths)

        # Read and write
        self.d_horizons = cl.Buffer(
                self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
                hostbuf=self.h_horizons)
        self.d_un1 = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_un1.nbytes)
        self.d_un2 = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_un2.nbytes)
        self.d_un2_1 = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_un2.nbytes)
        self.d_k1dn = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_k1dn.nbytes)
        self.d_k2dn = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, self.h_k2dn.nbytes)
        # Write only
        self.d_damage = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, self.h_damage.nbytes)
        self.d_errorn = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, self.h_errorn.nbytes)
        # Initialize kernel parameters
        self.cl_kernel_calc_bond_force.set_scalar_arg_dtypes(
            [None, None, None, None, None, None, None, None])
        self.cl_kernel_displacement_update.set_scalar_arg_dtypes(
            [None, None, None, None, None])
        self.cl_kernel_partial_displacement_update.set_scalar_arg_dtypes(
            [None, None, None, None])
        self.cl_kernel_check_bonds.set_scalar_arg_dtypes([None, None, None, None])
        self.cl_kernel_calculate_damage.set_scalar_arg_dtypes([None, None, None])
        self.cl_kernel_check_error.set_scalar_arg_dtypes([None, None, None, None, None, None])
    def __call__(self):
        """
        Conduct one iteration of the integrator.

        :arg u: A (`nnodes`, 3) array containing the displacements of all
            nodes.
        :type u: :class:`numpy.ndarray`
        :arg f: A (`nnodes`, 3) array containing the components of the force
            acting on each node.
        :type f: :class:`numpy.ndarray`

        :returns: The new displacements after integration.
        :rtype: :class:`numpy.ndarray`
        """

    def runtime(self, model):
        # Find k1dn (forces)
        self.cl_kernel_calc_bond_force(self.queue, 
                                          (model.nnodes,), 
                                          None, self.d_k1dn, self.d_un2, self.d_vols, self.d_horizons, self.d_coords, self.d_bond_stiffness, self.d_force_bc_types, self.d_force_bc_values)
        # Find first order accurate displacements
        # Scalars like self.h_dt can live on the host memory
        self.cl_kernel_partial_displacement_update(self.queue, (model.nnodes * model.degrees_freedom,),
                                                     None, self.d_k1dn, self.d_un2, self.d_un1, self.h_dt)
        # Find k2dn (forces)
        self.cl_kernel_calc_bond_force(self.queue, 
                                          (model.nnodes,), 
                                          None, self.d_k2dn, self.d_un1, self.d_vols, self.d_horizons, self.d_coords, self.d_bond_stiffness, self.d_force_bc_types, self.d_force_bc_values)
        if self.adapt_time_step(model) == 1:
            # Do not update second order displacements (i.e. repeat time step)
            pass
        else:
            # Update second order displacements
            self.cl_kernel_displacement_update(self.queue,
                                               (model.nnodes * model.degrees_freedom,), 
                                               None, self.d_bc_types, self.d_bc_values, self.d_un2_1, self.d_un2, self.h_dt)
            # Check for broken bonds
            self.cl_kernel_check_bonds(self.queue, 
                                       (model.nnodes, model.max_horizon_length),
                                       None, self.d_horizons, self.d_un2, self.d_coords, self.d_bond_critical_stretch)
    def write(self, model, t, sample):
        """ Write a mesh file for the current timestep
        """
        self.cl_kernel_calculate_damage(self.queue, (model.nnodes,), None, 
                                           self.d_damage, self.d_horizons,
                                           self.d_horizons_lengths)
        cl.enqueue_copy(self.queue, self.h_damage, self.d_damage)
        cl.enqueue_copy(self.queue, self.h_un2, self.d_un2)
        # TODO define a failure criterion, idea: rate of change of damage goes to 0 after it has started increasing
        tip_displacement = 0
        tmp = 0
        for i in range(model.nnodes):
            if self.h_tip_types[i] == 1:
                tmp +=1
                tip_displacement += self.h_un2[i][2]
        if tmp != 0:
            tip_displacement /= tmp
        else:
            tip_displacement = None
        vtk.write("output/U_"+"sample" + str(sample) +"t"+str(t) + ".vtk", "Solution time step = "+str(t),
                  model.coords, self.h_damage, self.h_un2)
        #vtk.writeDamage("output/damage_" + str(t)+ "sample" + str(sample) + ".vtk", "Title", self.h_damage)
        return self.h_damage, tip_displacement
    def adapt_time_step(self, model, error_size_max=1e-7, error_size_min=1e-10):
        adapt = 0
        # Check for error size
        self.cl_kernel_check_error(self.queue,
                              (model.nnodes * model.degrees_freedom,),
                              None, self.d_k1dn, self.d_k2dn, self.d_un1, self.d_un2_1, self.d_errorn, self.h_dt)
        cl.enqueue_copy(self.queue, self.h_errorn, self.d_errorn)
        error = np.linalg.norm(self.h_errorn, axis=1)
        error = np.mean(error)
        if error > error_size_max:
            self.h_dt /= 1.1
            print('Time step size reduced')
            print('time step is {}s, error size is {}'.format(self.h_dt, error))
            adapt = 1
        elif error < error_size_min:
            self.h_dt *= 1.1
            print('Time step size increased')
            print('time step is {}s, error size is {}'.format(self.h_dt, error))
        return adapt
    def incrementLoad(self, model, load_scale):
        if model.num_force_bc_nodes != 0:
            tmp = -1. * model.max_reaction * load_scale / (model.num_force_bc_nodes)
            # update the host force_bc_values
            self.h_force_bc_values = tmp * np.ones((model.nnodes, model.degrees_freedom), dtype=np.float64)
            #print(h_force_bc_values)
            # update the GPU force_bc_values
            self.d_force_bc_values = cl.Buffer(self.context,
                               cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                               hostbuf=self.h_force_bc_values)