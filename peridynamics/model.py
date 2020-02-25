import time
from .integrators import Integrator
from collections import namedtuple
from itertools import combinations
import meshio
import numpy as np
import sys
sys.path.insert(1, './peridynamics')
sys.path.insert(1, './peridynamics/post_processing')
# bb515 I have added these imports here
import vtk as vtk
import periFunctions as func


_MeshElements = namedtuple("MeshElements", ["connectivity", "boundary"])
_mesh_elements_2d = _MeshElements(connectivity="triangle",
                                  boundary="line")
_mesh_elements_3d = _MeshElements(connectivity="tetra",
                                  boundary="triangle")

# bb515 NOTE: _MeshElements(connectivity="tetra") works whereas "tetrahedron" raises a KeyError: 'tetrahedron'

sys.path.insert(1, '../examples/example1')
sys.path.insert(1, './kernels')
sys.path.insert(1, './post_processing')


class Model:
    """
    A peridynamics model using OpenCL.

    This class allows users to define a peridynamics system from parameters and
    a set of initial conditions (coordinates and connectivity).

    :Example: ::

        >>> from peridynamics import Model
        >>>
        >>> model = Model(
        >>>     mesh_file="./example.msh",
        >>>     horizon=0.1,
        >>>     critical_strain=0.005,
        >>>     elastic_modulus=0.05
        >>>     )

    To define a crack in the inital configuration, you may supply a list of
    pairs of particles between which the crack is.

    :Example: ::

        >>> from peridynamics import Model, initial_crack_helper
        >>>
        >>> initial_crack = [(1,2), (5,7), (3,9)]
        >>> model = Model(mesh_file, horizon=0.1, critical_strain=0.005,
        >>>               elastic_modulus=0.05, initial_crack=initial_crack)

    If it is more convenient to define the crack as a function you may also
    pass a function to the constructor which takes the array of coordinates as
    its only argument and returns a list of tuples as described above. The
    :func:`peridynamics.model.initial_crack_helper` decorator has been provided
    to easily create a function of the correct form from one which tests a
    single pair of node coordinates and returns `True` or `False`.

    :Example: ::

        >>> from peridynamics import Model, initial_crack_helper
        >>>
        >>> @initial_crack_helper
        >>> def initial_crack(x, y):
        >>>     ...
        >>>     if crack:
        >>>         return True
        >>>     else:
        >>>         return False
        >>>
        >>> model = Model(mesh_file, horizon=0.1, critical_strain=0.005,
        >>>               elastic_modulus=0.05, initial_crack=initial_crack)

    The :meth:`peridynamics.model.Model.simulate` method can be used to conduct
    a peridynamics simulation. For this an
    :class:`peridynamics.integrators.Integrator` is required, and optionally a
    function implementing the boundary conditions.

    :Example: ::

        >>> from peridynamics import Model, initial_crack_helper
        >>> from peridynamics.integrators import Euler
        >>>
        >>> model = Model(...)
        >>>
        >>> euler = Euler(dt=1e-3)
        >>>
        >>> indices = np.arange(model.nnodes)
        >>> model.lhs = indices[model.coords[:, 0] < 1.5*model.horizon]
        >>> model.rhs = indices[model.coords[:, 0] > 1.0 - 1.5*model.horizon]
        >>>
        >>> def boundary_function(model, u, step):
        >>>     u[model.lhs] = 0
        >>>     u[model.rhs] = 0
        >>>     u[model.lhs, 0] = -1.0 * step
        >>>     u[model.rhs, 0] = 1.0 * step
        >>>
        >>>     return u
        >>>
        >>> u, damage = model.simulate(steps=1000, integrator=euler,
        >>>                            boundary_function=boundary_function)
    """
    def __init__(self, bond_type, initial_crack=[], dimensions=3):
        """
        Construct a :class:`Model` object.

        :arg str mesh_file: Path of the mesh file defining the systems nodes
            and connectivity.
        :arg float horizon: The horizon radius. Nodes within `horizon` of
            another interact with that node and are said to be within its
            neighbourhood.
        :arg float critical_strain: The critical strain of the model. Bonds
            which exceed this strain are permanently broken.
        :arg float elastic_modulus: The appropriate elastic modulus of the
            material.
        :arg initial_crack: The initial crack of the system. The argument may
            be a list of tuples where each tuple is a pair of integers
            representing nodes between which to create a crack. Alternatively,
            the arugment may be a function which takes the (nnodes, 3)
            :class:`numpy.ndarray` of coordinates as an argument, and returns a
            list of tuples defining the initial crack. Default is []
        :type initial_crack: list(tuple(int, int)) or function
        :arg int dimensions: The dimensionality of the model. The
            default is 2.

        :returns: A new :class:`Model` object.
        :rtype: Model

        :raises DimensionalityError: when an invalid `dimensions` argument is
            provided.
        """
        ## bb515 I have wrongly defined all the PD parameters here, but unsure how to restructure this as there are too many arguments for the command line. We
        ## should probably make an input parameters file.
        
        # verbose
        self.v = True

        self.token_problems = ['test.msh', 'debug3D.msh', 'debug3D2.msh']
        self.verification_problems = ['1000beam2D.msh', '1000beam3D.msh', '1000beam3DT.msh']
        self.benchmark_problems = ['3300beam.msh']
        
        self.mesh_file = '3300beam.msh'
        self.network_file = 'Network.vtk'
        
        if dimensions == 2:
            self.mesh_elements = _mesh_elements_2d
        elif dimensions == 3:
            self.mesh_elements = _mesh_elements_3d
        else:
            raise DimensionalityError(dimensions)
            
        self.dimensions = dimensions
        
        
        # Are the stiffness correction factors calculated using mesh element volumes (default 'precise', 1) or average nodal volume of a transfinite mesh (0)      
        self.precise_stiffness_correction = 1
        
        # Is the mesh transfinite mesh (support regular grid spacing with cuboidal (not tetra) elements, look up "gmsh transfinite") (default 0)
        # This mode will likely not be compatible with FEM, so shouldn't be used apart from the validation I'm doing
        self.transfinite = 0

        # Classical material model parameters
        self.YOUNGSM_CONCRETE = 1.*22e9
        self.DENSITY_CONCRETE = 2400.0
        self.YOUNGSM_STEEL = 1.*210e9
        self.DENSITY_STEEL = 8000.0
        self.COMPRESSIVE_STRENGTH_CONCRETE = 1.*25e6
        self.TENSILE_STRENGTH_CONCRETE = 2.6e6
        self.CONCRETE_FRACTURE_ENERGY = 100
        self.YIELD_STRENGTH_STEEL = 1.*250e6
# =============================================================================
#         # These classical material model parameters may be used in post processing to plot stress strain fields
#         # (not yet implemented)
#         self.POISSON_STEEL = 0.3                                    # Poisson ratio
#         self.POISSON_CONCRETE = 0.2
#         self.G_STEEL = 78e9                                         # Shear modulus
#         self.G_CONCRETE = self.YOUNGSM_CONCRETE/(2*(1+self.POISSON_CONCRETE))  # TODO need to verify this value
#         self.EFFECTIVEMODULUS_CONCRETE = self.YOUNGSM_CONCRETE/((1-2*self.POISSON_CONCRETE)(1+self.POISSON_CONCRETE))   # Effective modulus
#         self.EFFECTIVEMODULUS_STEEL = self.YOUNGSM_STEEL/((1-2*self.POISSON_STEEL)*(1+self.POISSON_STEEL))
# =============================================================================
        
        # Peridynamics parameters. These parameters will be passed to openCL kernels by command line argument
        # Bond-based peridynamics, known in PDLAMMPS as Prototype Microelastic Brittle (PMB) Model requires
        # a poisson ratio of v = 0.25, but this makes little to no difference in quasi-brittle materials
        self.PD_POISSON = 0.25
        self.PD_K_CONCRETE = self.YOUNGSM_CONCRETE/ (3* (1 - 2*self.PD_POISSON))
        self.PD_K_STEEL = self.YOUNGSM_STEEL / (3* (1- 2*self.PD_POISSON))
        # Density and damping of governing ODE
        self.PD_DENSITY = np.double(self.DENSITY_CONCRETE)       # force density term
        if self.mesh_file in self.benchmark_problems:
            # Problem specific parameters
            self.volume_total = 3.3 * 0.6 * 0.25
            self.dx = np.power(1.*self.volume_total/4625,1./3)
            self.PD_HORIZON = self.dx * np.pi 
            self.PD_FAMILY_VOLUME =(4./3)*np.pi*np.power(self.PD_HORIZON, 3)
            self.PD_DAMPING = 2.0e6                         # damping term
            # Peridynamic bond stiffness, c
            self.PD_C_CONCRETE = np.double(
                    (18.00 * self.PD_K_CONCRETE) / (np.pi * np.power(self.PD_HORIZON, 4)))
            self.PD_C_STEEL = np.double(
                    (18.00 * self.PD_K_STEEL) / (np.pi * np.power(self.PD_HORIZON, 4)))
            # Peridynamic critical stretch, s00
            self.PD_S0_CONCRETE = np.double(0.000533) # check this value
            self.PD_S0_STEEL = np.double(0.01)
            # User input parameters
            #self.loadRate = np.double(1e-5)
            self.crackLength = np.double(0.3)
            self.saf_fac = 0.2 # Typical values 0.70 to 0.95 (Sandia PeridynamicSoftwareRoadmap)
            #self.dt = (0.8 * np.power( 2.0 * self.DENSITY_CONCRETE * self.dx / (np.pi * np.power(self.PD_HORIZON, 2.0) * self.dx * self.PD_C_CONCRETE), 0.5)) * self.saf_fac
            self.dt = 1e-13
            self.self_weight = 1.*self.DENSITY_CONCRETE * self.volume_total * 9.81
            self.max_reaction = 1.* self.self_weight # in newtons, about 85 times self weight
            self.load_scale_rate = 1/1000
        elif self.mesh_file == '1000beam3DT.msh':
            # Problem specific parameters
            self.volume_total = 1.0 * 0.2 * 0.1
            self.PD_NODE_VOLUME_AVERAGE = 1.* self.volume_total / 67500 # volume_total / nnodes
            self.dx = 1./150
            self.PD_HORIZON = self.dx * np.pi 
            self.PD_FAMILY_VOLUME = np.pi * np.power(self.PD_HORIZON, 2)
            self.PD_DAMPING = 2.8e6                           # damping term
            # Peridynamic bond stiffness, c
            self.PD_C_CONCRETE = np.double(
                    (18.00 * self.PD_K_CONCRETE) / (np.pi * np.power(self.PD_HORIZON, 4)))
            self.PD_C_STEEL = np.double(
                    (18.00 * self.PD_K_STEEL) / (np.pi * np.power(self.PD_HORIZON, 4)))
            # Peridynamic critical stretch, s00
            self.PD_S0_CONCRETE = np.double(self.TENSILE_STRENGTH_CONCRETE / self.YOUNGSM_CONCRETE)
            self.PD_S0_STEEL = np.double(0.01)
            # User input parameters
            self.loadRate = np.double(1e-4)
            self.crackLength = np.double(0)
            self.saf_fac = 0.70 # Typical values 0.70 to 0.95 (Sandia PeridynamicSoftwareRoadmap)
            #self.dt = (0.8 * np.power( 2.0 * self.DENSITY_CONCRETE * self.dx / (np.pi * np.power(self.PD_HORIZON, 2.0) * self.dx * self.PD_C_CONCRETE), 0.5)) * self.saf_fac
            self.dt = 1e-8
            #self.max_reaction = 10000.0 # in newtons
            self.self_weight = 1.*self.DENSITY_CONCRETE * self.volume_total * 9.81
            self.max_reaction = 1.* self.self_weight # in newtons, about 85 times self weight
            self.load_scale_rate = 1/1000
        elif self.mesh_file == 'debug3D.msh':
            # Problem specific parameters
            self.volume_total = 1.0 * 1.0 * 1.0
            self.PD_NODE_VOLUME_AVERAGE = 1.* self.volume_total / 67500 # volume_total / nnodes
            self.dx = 1./9
            self.PD_HORIZON = self.dx * np.pi 
            self.PD_FAMILY_VOLUME = np.pi * np.power(self.PD_HORIZON, 2)
            self.PD_DAMPING = 2.5e6                           # damping term
            # Peridynamic bond stiffness, c
            self.PD_C_CONCRETE = np.double(
                    (18.00 * self.PD_K_CONCRETE) / (np.pi * np.power(self.PD_HORIZON, 4)))
            self.PD_C_STEEL = np.double(
                    (18.00 * self.PD_K_STEEL) / (np.pi * np.power(self.PD_HORIZON, 4)))
            # Peridynamic critical stretch, s00
            self.PD_S0_CONCRETE = np.double(self.TENSILE_STRENGTH_CONCRETE / self.YOUNGSM_CONCRETE)
            self.PD_S0_STEEL = np.double(0.01)
            # User input parameters
            self.loadRate = np.double(1e-4)
            self.crackLength = np.double(0)
            self.saf_fac = 0.70 # Typical values 0.70 to 0.95 (Sandia PeridynamicSoftwareRoadmap)
            self.dt = (0.8 * np.power( 2.0 * self.DENSITY_CONCRETE * self.dx / (np.pi * np.power(self.PD_HORIZON, 2.0) * self.dx * self.PD_C_CONCRETE), 0.5)) * self.saf_fac
            self.self_weight = 1.*self.DENSITY_CONCRETE * self.volume_total * 9.81
            self.max_reaction = 1.* self.self_weight # in newtons, about 85 times self weight
            #self.max_reaction = 10000.0 # in newtons
            self.load_scale_rate = 1/1000
        elif self.mesh_file in self.token_problems:
            # Problem specific parameters
            self.volume_total = 3.3 * 0.6 * 0.25
            self.dx = np.power(1.*self.volume_total/4625,1./3)
            self.PD_HORIZON = self.dx * np.pi 
            self.PD_FAMILY_VOLUME =np.pi * np.power(self.PD_HORIZON, 2)
            self.PD_DAMPING = 2.5e6                           # damping term
            # Peridynamic bond stiffness, c
            self.PD_C_CONCRETE = np.double(
                    (18.00 * self.PD_K_CONCRETE) / (np.pi * np.power(self.PD_HORIZON, 4)))
            self.PD_C_STEEL = np.double(
                    (18.00 * self.PD_K_STEEL) / (np.pi * np.power(self.PD_HORIZON, 4)))
            # Peridynamic critical stretch, s00
            self.PD_S0_CONCRETE = 0.000533 # check this value
            self.PD_S0_STEEL = np.double(0.01)
            # User input parameters
            self.loadRate = np.double(1e-4)
            self.crackLength = np.double(0.3)
            self.saf_fac = 0.70 # Typical values 0.70 to 0.95 (Sandia PeridynamicSoftwareRoadmap)
            #self.dt = (0.8 * np.power( 2.0 * self.DENSITY_CONCRETE * self.dx / (np.pi * np.power(self.PD_HORIZON, 2.0) * self.dx * self.PD_C_CONCRETE), 0.5)) * self.saf_fac
            self.dt = np.double(1e-6)
            self.self_weight = 1.*self.DENSITY_CONCRETE * self.volume_total * 9.81
            self.max_reaction = 1.* self.self_weight # in newtons, about 85 times self weight
            self.load_scale_rate = 1/1000
        # No. coordinate dimensions
        self._read_mesh(self.mesh_file)
        self.DPN = np.intc(3)
        self.PD_DPN_NODE_NO = np.intc(self.DPN * self.nnodes)
        
        st = time.time()
        
        self._set_volume(self.volume_total)
        
        # If the network has already been written to file, then read, if not, setNetwork
        try:
            self._read_network(self.network_file)
        except:
            self._set_network(self.PD_HORIZON, bond_type)
        
        # bb515 Initating crack is done when we _read_network or _set_network
        self._set_connectivity(initial_crack)
        
        print(
            "Building horizons took {} seconds. Horizon length: {}".format(
                (time.time() - st), self.MAX_HORIZON_LENGTH))
        # bb515 self.setH() # Will further optimise the code, TODO Set the node distance and store in OpenCL device memory

        # initiate
        self.bctypes = np.zeros((self.nnodes, self.DPN), dtype=np.intc)
        self.bcvalues = np.zeros((self.nnodes, self.DPN), dtype=np.float64)
        self.tiptypes = np.zeros(self.nnodes, dtype=np.intc)
        
        if self.v == True:
            print("total volume", self.sum_total_volume)
            print("volume total", self.volume_total)
            print("Horizon distance,", self.PD_HORIZON)
            print("Max reaction", self.max_reaction)
            print("Time step", self.dt)
            
    def _read_network(self, network_file):
        """ For reading a network file if it has been written to file yet. Quicker than building horizons from scratch."""
        
        f = open(network_file, "r")
        
        if f.mode == "r":
            iline = 0
            
            # Read the Max horizons length first
            find_MHL = 0
            while (find_MHL == 0):
                iline += 1
                line = f.readline()
                row = line.strip()
                rowAsList = row.split()
                
                find_MHL = 1 if 'MAX_HORIZON_LENGTH' in rowAsList else 0
            
            
            MAX_HORIZON_LENGTH = int(rowAsList[1])
            
            # Now read nnodes
            find_nnodes = 0
            while (find_nnodes == 0):
                iline += 1
                line = f.readline()
                row = line.strip()
                rowAsList = row.split()
                
                find_nnodes = 1 if 'NNODES' in rowAsList else 0
            
            nnodes = int(rowAsList[1])

            # Now read horizons lengths
            find_horizons_lengths = 0
            while (find_horizons_lengths == 0):
                iline += 1
                line = f.readline()
                row = line.strip()
                rowAsList = row.split()
                
                find_horizons_lengths = 1 if 'HORIZONS_LENGTHS' in rowAsList else 0
            
            horizons_lengths = np.zeros(nnodes, dtype=int)
            for i in range(0, nnodes):
                iline += 1
                line = f.readline()
                horizons_lengths[i] = np.intc(line.split())
                
            print('Building family matrix from file')
            # Now read family matrix
            find_family = 0
            while (find_family == 0):
                iline += 1
                line = f.readline()
                row = line.strip()
                rowAsList = row.split()
                
                find_family = 1 if 'FAMILY' in rowAsList else 0
            
            family = []
            for i in range(nnodes):
                iline += 1
                line = f.readline()
                row = line.strip()
                rowAsList = line.split()
                family.append(np.zeros(len(rowAsList), dtype=np.intc))
                for j in range(0, len(rowAsList)):
                    family[i][j] = np.intc(rowAsList[j])
            
            print('Finding stiffness values')
            # Now read stiffness values
            find_stiffness = 0
            while (find_stiffness == 0):
                iline += 1
                line = f.readline()
                row = line.strip()
                rowAsList = row.split()
                
                find_stiffness = 1 if 'STIFFNESS' in rowAsList else 0
            print('Building stiffnesses from file')
            
            bond_stiffness_family = []
            for i in range(nnodes):
                iline += 1
                line = f.readline()
                row = line.strip()
                rowAsList = line.split()
                bond_stiffness_family.append(np.zeros(len(rowAsList), dtype=np.float64))
                for j in range(0, len(rowAsList)):
                    bond_stiffness_family[i][j] = (rowAsList[j])
            
            print('Finding critical stretch values')
            # Now read critcal stretch values
            find_stretch = 0
            while (find_stretch == 0):
                iline += 1
                line = f.readline()
                row = line.strip()
                rowAsList = row.split()
                
                find_stretch = 1 if 'STRETCH' in rowAsList else 0
            
            print('Building critical stretch values from file')
            bond_critical_stretch_family = []
            for i in range(nnodes):
                iline += 1
                line = f.readline()
                row = line.strip()
                rowAsList = line.split()
                bond_critical_stretch_family.append(np.zeros(len(rowAsList), dtype=np.float64))
                for j in range(0, len(rowAsList)):
                    bond_critical_stretch_family[i][j] = rowAsList[j]
            
            # Maximum number of nodes that any one of the nodes is connected to
            MAX_HORIZON_LENGTH_CHECK = np.intc(
                len(max(family, key=lambda x: len(x)))
                )
            
            assert MAX_HORIZON_LENGTH == MAX_HORIZON_LENGTH_CHECK, 'Read failed on MAX_HORIZON_LENGTH check'
            
            horizons = -1 * np.ones([nnodes, MAX_HORIZON_LENGTH])
            for i, j in enumerate(family):
                horizons[i][0:len(j)] = j
                
            bond_stiffness = -1. * np.ones([nnodes, MAX_HORIZON_LENGTH])
            for i, j in enumerate(bond_stiffness_family):
                bond_stiffness[i][0:len(j)] = j
            
            bond_critical_stretch = -1. * np.ones([nnodes, MAX_HORIZON_LENGTH])
            for i, j in enumerate(bond_critical_stretch_family):
                bond_critical_stretch[i][0:len(j)] = j

            # Make sure it is in a datatype that C can handle
            self.horizons = horizons.astype(np.intc)
            self.bond_stiffness = bond_stiffness
            self.bond_critical_stretch = bond_critical_stretch
            
            self.horizons_lengths = horizons_lengths
            self.family = family
            self.MAX_HORIZON_LENGTH = MAX_HORIZON_LENGTH
            self.nnodes = nnodes
            f.close()

    def _read_mesh(self, filename):
        """
        Read the model's nodes, connectivity and boundary from a mesh file.

        :arg str filename: Path of the mesh file to read

        :returns: None
        :rtype: NoneType
        """
        mesh = meshio.read(filename)
        
        if self.transfinite == 1:
            # In this case, only need coordinates, encoded as mesh points
            self.coords = mesh.points
            self.nnodes = self.coords.shape[0]
            
        else:
            
            # Get coordinates, encoded as mesh points
            self.coords = mesh.points
            self.nnodes = self.coords.shape[0]

            # Get connectivity, mesh triangle cells
            self.connectivity = mesh.cells['tetra']

            # Get boundary connectivity, mesh lines
            self.connectivity_bnd = mesh.cells['triangle']
            
            # bb515 this has been removed?
            self.nelem_bnd = self.connectivity_bnd.shape[0]
        
    
    def write_mesh(self, filename, damage=None, displacements=None,
                   file_format=None):
        """
        Write the model's nodes, connectivity and boundary to a mesh file.
        Optionally, write damage and displacements as points data.

        :arg str filename: Path of the file to write the mesh to.
        :arg damage: The damage of each node. Default is None.
        :type damage: :class:`numpy.ndarray`
        :arg displacements: An array with shape (nnodes, dim) where each row is
            the displacement of a node. Default is None.
        :type displacements: :class:`numpy.ndarray`
        :arg str file_format: The file format of the mesh file to
            write. Inferred from `filename` if None. Default is None.

        :returns: None
        :rtype: NoneType
        """
        meshio.write_points_cells(
            filename,
            points=self.coords,
            cells={
                self.mesh_elements.connectivity: self.connectivity,
                self.mesh_elements.boundary: self.connectivity_bnd
                },
            point_data={
                "damage": damage,
                "displacements": displacements
                },
            file_format=file_format
            )

    def _set_volume(self, volume_total):
        """
        Calculate the value of each node.
        
        :arg volume_total: User input for the total volume of the mesh, for checking sum total of elemental volumes is equal to user input volume for simple prismatic problems.
        In the case of non-prismatic problems when the user does not know what the volume is, we should do something else as an assertion

        :returns: None
        :rtype: NoneType
        """
        
        # bb515 this has changed significantly,
        # OpenCL (or rather C) requires that we are careful with
        # types so that they are compatible with the specifed C types in the
        # OpenCL kernels
        self.V = np.zeros(self.nnodes, dtype=np.float64)
        
        # this is the sum total of the elemental volumes, initiated at 0.
        self.sum_total_volume = 0
        
        if self.transfinite == 1:
            """ Tranfinite mode is when we have approximated the volumes of the nodes
            as the average volume of nodes on a rectangular grid.
            The transfinite grid (search on youtube for "transfinite mesh gmsh") is not
            neccessarily made up of tetrahedra, but may be made up of cuboids.
            """
            tmp = volume_total / self.nnodes
            for i in range(0, self.nnodes):
                self.V[i] = tmp
                self.sum_total_volume += tmp
        else:
            for element in self.connectivity:
                
                # Compute Area or Volume
                val = 1. / len(element)
                
                # Define area of element
                if self.dimensions == 2:
                    
                    xi, yi, *_ = self.coords[element[0]]
                    xj, yj, *_ = self.coords[element[1]]
                    xk, yk, *_ = self.coords[element[2]]
                    
                    element_area = 0.5 * np.absolute(((xj - xi) * (yk - yi) - (xk - xi) * (yj - yi)))
                    val *= element_area
                    self.sum_total_volume += element_area
                    
                elif self.dimensions == 3:
        
                    a = self.coords[element[0]]
                    b = self.coords[element[1]]
                    c = self.coords[element[2]]
                    d = self.coords[element[3]]
                    
                    # Volume of a tetrahedron
                    i = np.subtract(a,d)
                    j = np.subtract(b,d)
                    k = np.subtract(c,d)
                    
                    element_volume = (1./6) * np.absolute(np.dot(i, np.cross(j,k)))
                    val*= element_volume
                    self.sum_total_volume += element_volume
                else:
                    raise ValueError('dim', 'dimension size can only take values 2 or 3')
                    
                for j in range(0, len(element)):
                    self.V[element[j]] += val
        # For non prismatic problems where the user does not know the volume_total, do another test?
        assert self.sum_total_volume - volume_total < volume_total/1e5, "Sum total of elemental volumes was {}, but geometry had total volume {}".format(self.sum_total_volume, volume_total)
        self.V = self.V.astype(np.float64)

    def _set_network(self, horizon, bond_type):
        """
        Sets the family matrix, and converts this to a horizons matrix (a fixed size data structure compatible with OpenCL).
        Calculates horizons_lengths
        
        Also initiate crack here if there is one
        

        :arg horizon: Peridynamic horizon distance

        :returns: None
        :rtype: NoneType
        """
        
        # Container for nodal family
        family = []
        bond_stiffness_family = []
        bond_critical_stretch_family = []

        # Container for number of nodes (including self) that each of the nodes
        # is connected to
        self.horizons_lengths = np.zeros(self.nnodes, dtype=np.intc)

        for i in range(0, self.nnodes):
            print('node', i, 'networking...')
            # Container for family nodes
            tmp = []
            # Container for bond stiffnesses
            tmp2 = []
            # Container for bond critical stretches
            tmp3 = []
            for j in range(0, self.nnodes):
                if i != j:
                    l2_sqr = func.l2_sqr(self.coords[i, :], self.coords[j, :])
                    if np.sqrt(l2_sqr) < horizon:
                        tmp.append(j)
                        # Determine the material properties for that bond
                        material_flag = bond_type(self.coords[i, :], self.coords[j, :])
                        if material_flag == 'steel':
                            tmp2.append(self.PD_C_STEEL)
                            tmp3.append(self.PD_S0_STEEL)
                        elif material_flag == 'interface':
                            tmp2.append(self.PD_C_CONCRETE) # choose the weakest stiffness of the two bond types
                            tmp3.append(self.PD_S0_CONCRETE * 3.0) # 3.0 is used for interface bonds in the literature
                        elif material_flag == 'concrete':
                            tmp2.append(self.PD_C_CONCRETE)
                            tmp3.append(self.PD_S0_CONCRETE)
             
            family.append(np.zeros(len(tmp), dtype=np.intc))
            bond_stiffness_family.append(np.zeros(len(tmp2), dtype=np.float64))
            bond_critical_stretch_family.append(np.zeros(len(tmp3), dtype=np.float64))
            
            
            self.horizons_lengths[i] = np.intc((len(tmp)))
            for j in range(0, len(tmp)):
                family[i][j] = np.intc(tmp[j])
                bond_stiffness_family[i][j] = np.float64(tmp2[j])
                bond_critical_stretch_family[i][j] = np.float64(tmp3[j])
            
        
        assert len(family) == self.nnodes
        # As numpy array
        self.family = np.array(family)
        
        # Do the bond critical ste
        self.bond_critical_stretch_family = np.array(bond_critical_stretch_family)
        self.bond_stiffness_family = np.array(bond_stiffness_family)
        
        self.family_v = np.zeros(self.nnodes)
        for i in range(0, self.nnodes):
            tmp = 0 # tmp family volume
            family_list = family[i]
            for j in range(0, len(family_list)):
                tmp += self.V[family_list[j]]
            self.family_v[i] = tmp
        
        
        if self.precise_stiffness_correction == 1:
            # Calculate stiffening factor nore accurately using actual nodal volumes
            for i in range(0, self.nnodes):
                family_list = family[i]
                nodei_family_volume = self.family_v[i] # Possible to calculate more exactly, we have the volumes for free
                for j in range(len(family_list)):
                    nodej_family_volume = self.family_v[j]
                    stiffening_factor = 2.* self.PD_FAMILY_VOLUME /  (nodej_family_volume + nodei_family_volume)
                    print('Stiffening factor {}'.format(stiffening_factor))
                    bond_stiffness_family[i][j] *= stiffening_factor
        elif self.precise_stiffness_correction == 0:
            # Calculate stiffening factor - surface corrections for 3D problem, for this we need family matrix
            for i in range(0, self.nnodes):
                nnodes_i_family = len(family[i])
                nodei_family_volume = nnodes_i_family * self.PD_NODE_VOLUME_AVERAGE # Possible to calculate more exactly, we have the volumes for free
                for j in range(len(family[i])):
                    nnodes_j_family = len(family[j])
                    nodej_family_volume = nnodes_j_family* self.PD_NODE_VOLUME_AVERAGE # Possible to calculate more exactly, we have the volumes for free
                    
                    stiffening_factor = 2.* self.PD_FAMILY_VOLUME /  (nodej_family_volume + nodei_family_volume)
                    
                    bond_stiffness_family[i][j] *= stiffening_factor
            
        
        # Maximum number of nodes that any one of the nodes is connected to
        self.MAX_HORIZON_LENGTH = np.intc(
            len(max(self.family, key=lambda x: len(x)))
            )

        self.horizons = -1 * np.ones([self.nnodes, self.MAX_HORIZON_LENGTH])
        for i, j in enumerate(self.family):
            self.horizons[i][0:len(j)] = j
            
        self.bond_stiffness = -1. * np.ones([self.nnodes, self.MAX_HORIZON_LENGTH])
        for i, j in enumerate(self.bond_stiffness_family):
            self.bond_stiffness[i][0:len(j)] = j
            
        self.bond_critical_stretch = -1. * np.ones([self.nnodes, self.MAX_HORIZON_LENGTH])
        for i, j in enumerate(self.bond_critical_stretch_family):
            self.bond_critical_stretch[i][0:len(j)] = j

        # Make sure it is in a datatype that C can handle
        self.horizons = self.horizons.astype(np.intc)
        
        vtk.writeNetwork("Network"+".vtk", "Network",
                      self.MAX_HORIZON_LENGTH, self.horizons_lengths, self.family, self.bond_stiffness_family, self.bond_critical_stretch_family)
                    
    def _set_connectivity(self, initial_crack):
        """
        Sets the intial crack.

        :arg initial_crack: The initial crack of the system. The argument may
            be a list of tuples where each tuple is a pair of integers
            representing nodes between which to create a crack. Alternatively,
            the arugment may be a function which takes the (nnodes, 3)
            :class:`numpy.ndarray` of coordinates as an argument, and returns a
            list of tuples defining the initial crack.
        :type initial_crack: list(tuple(int, int)) or function

        :returns: None
        :rtype: NoneType
        
        
        bb515 connectivity matrix is replaced by self.horizons and self.horizons_lengths for OpenCL
        
        also see self.family, which is a verlet list:
            self.horizons and self.horizons_lengths are neccessary OpenCL cannot deal with non fixed length arrays
        """
        if callable(initial_crack):
            initial_crack = initial_crack(self.coords)
        
# =============================================================================
#         initiate_crack = 0
#         # Initiate crack
#         if initiate_crack == 1:
#             for i in range(0, self.nnodes):
#     
#                 for k in range(0, self.MAX_HORIZON_LENGTH):
#                     j = self.horizons[i][k]
#                     if self.isCrack(self.coords[i, :], self.coords[j, :]):
#                         self.horizons[i][k] = np.intc(-1)
#             
# =============================================================================
        # bb515 this code is meant to replace the code commented out above, and now uses your "initial_crack" list. I haven't tested this code, it might be buggy.
        for i,j in initial_crack:
            for l, m in enumerate(self.horizons[i]):
                if m == j:
                    self.horizons[i][l] = -1
    
    def _set_H(self):
        """
        Constructs the failure strains matrix and H matrix, which is a sparse
        matrix containing distances.

        :returns: None
        :rtype: NoneType
        
        bb515 failure strains matrix is replaced by self.bond_critical_stretch
        for OpenCL
        """
        # TODO add this back in?

    def bond_stretch(self, u):
        """
        Calculates the strain (bond stretch) of all nodes for a given
        displacement.

        :arg u: The displacement array with shape
            (`nnodes`, `dimension`).
        :type u: :class:`numpy.ndarray`

        :returns: None
        :rtype: NoneType
        
        
        bb515 this is replaced by '__kernel void CheckBonds' OpenCL kernel.
        But has recently been merged into a TimeMarching kernel as this is more optimal (don't calculate norms twice)
        """

    def damage(self):
        """
        Calculates bond damage.

        :returns: A (`nnodes`, ) array containing the damage
            for each node.
        :rtype: :class:`numpy.ndarray`
        
        
        bb515 this has been replaced by an OpenCL kernel '__kernel void CalculateDamage' NOTE this kernel function works on my laptop, but for some reason
        it calculates damages wrong, but only when its ran on the GPU ?? 20/02/20
        """

    def bond_force(self):
        """
        Calculate the force due to bonds acting on each node.

        :returns: A (`nnodes`, 3) array of the component of the force in each
            dimension for each node.
        :rtype: :class:`numpy.ndarray`
        
        bb515 replaced by opencl kernel "__kernel void TimeMarching2" or similar name for different integrators, e.g. "__kernel void TimeMarching<n>" where n is an integer
        """

    def simulate(self, model, steps, integrator, write=None, toolbar=0):
        """
        Simulate the peridynamics model.

        :arg int steps: The number of simulation steps to conduct.
        :arg  integrator: The integrator to use, see
            :mod:`peridynamics.integrators` for options.
        :type integrator: :class:`peridynamics.integrators.Integrator`
        :arg boundary_function: A function to apply the boundary conditions for
            the simlation. It has the form
            boundary_function(:class:`peridynamics.model.Model`,
            :class:`numpy.ndarray`, `int`). The arguments are the model being
            simulated, the current displacements, and the current step number
            (beginning from 1). `boundary_function` returns a (nnodes, 3)
            :class:`numpy.ndarray` of the updated displacements
            after applying the boundary conditions. Default `None`.
        :type boundary_function: function
        :arg u: The initial displacements for the simulation. If `None` the
            displacements will be initialised to zero. Default `None`.
        :type u: :class:`numpy.ndarray`
        :arg int write: The frequency, in number of steps, to write the system
            to a mesh file by calling
            :meth:`peridynamics.model.Model.write_mesh`. If `None` then no
            output is written. Default `None`.
        """
        if not isinstance(integrator, Integrator):
            raise InvalidIntegrator(integrator)

        # Container for plotting data
        damage_data = []
        tip_displacement_data = []
        
        #Progress bar
        toolbar_width = 40
        if toolbar:    
            sys.stdout.write("[%s]" % (" " * toolbar_width))
            sys.stdout.flush()
            sys.stdout.write("\b" * (toolbar_width+1)) # return to start of line, after '['
        st = time.time()
        for step in range(1, steps+1):
            # Conduct one integration step
            integrator.runtime(model) 
            if write:
                if step % write == 0:
                    damage_sum, tip_displacement = integrator.write(model, step)
                    damage_data.append(damage_sum)
                    tip_displacement_data.append(tip_displacement)
                    
                    if damage_sum > 3000.0:
                        print('Failure criterion reached, Peridynamic Simulation -- Stopping')
                        break
                    if toolbar == 0:
                        print('Print number {}/{} complete in {} s '.format(int(step/write), int(steps/write), time.time() - st))
                        st = time.time()

            # Increase load in linear increments
            load_scale = min(1.0, model.load_scale_rate * step)
            if load_scale != 1.0:
                integrator.incrementLoad(model, load_scale)
                    
            # Loading bar update
            if step%(steps/toolbar_width)<1 & toolbar:
                sys.stdout.write("\u2588")
                sys.stdout.flush()
        
        if toolbar:
            sys.stdout.write("]\n")

        return damage_data, tip_displacement_data

def initial_crack_helper(crack_function):
    """
    A decorator to help with the construction of an initial crack function.

    crack_function has the form crack_function(icoord, jcoord) where icoord and
    jcoord are :class:`numpy.ndarray` s representing two node coordinates.
    crack_function returns a truthy value if there is a crack between the two
    nodes and a falsy value otherwise.

    This decorator returns a function which takes all node coordinates and
    returns a list of tuples of the indices pair of nodes which define the
    crack. This function can therefore be used as the `initial_crack` argument
    of the :class:`Model`

    :arg function crack_function: The function which determine whether there is
        a crack between a pair of node coordinates.

    :returns: A function which determines all pairs of nodes with a crack
        between them.
    :rtype: function
    """
    def initial_crack(coords):
        crack = []
        # Iterate over all unique pairs of coordinates with their indicies
        for (i, icoord), (j, jcoord) in combinations(enumerate(coords), 2):
            if crack_function(icoord, jcoord):
                crack.append((i, j))
        return crack
    return initial_crack

class DimensionalityError(Exception):
    """
    Raised when an invalid dimensionality argument used to construct a model
    """
    def __init__(self, dimensions):
        message = (
            f"The number of dimensions must be 2 or 3,"
            " {dimensions} was given."
            )

        super().__init__(message)


class InvalidIntegrator(Exception):
    """
    Raised when the integrator passed to
    :meth:`peridynamics.model.Model.simulate` is not an instance of
    :class:`peridynamics.integrators.Integrator`.
    """
    def __init__(self, integrator):
        message = (
            f"{integrator} is not an instance of"
            "peridynamics.integrators.Integrator"
            )

        super().__init__(message)