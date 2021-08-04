
# ----------------------------------------------
#      Beam 4 (175 mm x 50 mm x 50 mm)
# ----------------------------------------------


import argparse
import cProfile
from io import StringIO
import numpy as np
import pathlib
from peripy import Model
from peripy.integrators import VelocityVerletCL  # Euler_jit
from peripy.utilities import read_array as read_model
from peripy.utilities import write_array
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from pstats import SortKey, Stats
from bc_utilities import (
    calc_boundary_conditions_magnitudes, is_tip_5mm, is_tip_2mm, is_tip_1mm,
    is_bond_type_5mm, is_bond_type_2mm, is_bond_type_1mm,
    is_displacement_boundary_5mm, is_displacement_boundary_2mm,
    is_displacement_boundary_1mm, smooth_step_data)
import os
import scipy.interpolate as inter
import h5py
import warnings
# import matplotlib.pyplot as plt     # edit MH
# import scipy.io                     # edit MH

os.environ['PYOPENCL_CTX'] = '0:0'

# TODO: why are these variables not part of the input file?

dxs = {'175beam5.0mmUN4T.msh': 5.0e-3,
       '175beam2.0mmUN4T.msh': 2.0e-3,
       '175beam1.0mmUN4T.msh': 1.0e-3,
       }

nnodes = {
    '175beam5.0mmUN4T.msh': 3645,
    '175beam2.0mmUN4T.msh': 55311,
    '175beam1.0mmUN4T.msh': 438100,
    }

safety_factor = {
    '175beam5.0mmUN4T.msh': 0.25,
    '175beam2.0mmUN4T.msh': 0.10,
    '175beam1.0mmUN4T.msh': 0.15,
    }

dampings = {
    '175beam5.0mmUN4T.msh': 2.5e6,
    '175beam2.0mmUN4T.msh': 3.0e6,
    '175beam1.0mmUN4T.msh': 3.0e6,
    }

# We found these values from example_s0.py
s_0s = {
    '175beam5.0mmUN4T.msh': np.float64(1.05e-4),  # Yang
    #'175beam5.0mmUN4T.msh': np.float64(2.2e-4),
    #'175beam5.0mmUN4T.msh': np.float64(2.3e-4),
    '175beam2.0mmUN4T.msh': np.float64(1.80e-4),
    '175beam1.0mmUN4T.msh': np.float64(1.87e-4),
    }

s_cs = {
    '175beam5.0mmUN4T.msh': np.float64(5.56e-3),  # Yang
    # '175beam5.0mmUN4T.msh': np.float64(1.75e-3),
    # '175beam5.0mmUN4T.msh': np.float64(1.45e-3),
    '175beam2.0mmUN4T.msh': np.float64(3.0e-3),
    '175beam1.0mmUN4T.msh': np.float64(1.75e-3),
    }

s_1s = {
    '175beam5.0mmUN4T.msh': np.float64(6.90e-4),  # Yang
    # '175beam5.0mmUN4T.msh': np.float64(2.25e-4),
    # '175beam5.0mmUN4T.msh': np.float64(2.42e-4),
    '175beam2.0mmUN4T.msh': np.float64(2.0e-4),
    '175beam1.0mmUN4T.msh': np.float64(4.5e-4),
    }

cs = {
    '175beam5.0mmUN4T.msh': np.float64(2.32e+18),  # Yang
    # '175beam5.0mmUN4T.msh': np.float64(1.7917711297325573e+18),
    '175beam2.0mmUN4T.msh': np.float64(7.0795e19),
    '175beam1.0mmUN4T.msh': np.float64(1.009890e21),
    # '175beam1.0mmUN4T.msh': np.float64(1.00754e21),
    }


read_path = pathlib.Path() / "EpUN4.h5"
read_path_restart = pathlib.Path() / "state.npz"


# TODO: move this to utilities
def read_array(read_path, dataset):
    """
    Read a :class numpy.ndarray: from a HDF5 file.

    :arg read_path: The path to which the HDF5 file is written.
    :type read_path: path-like or str
    :arg dataset: The name of the dataset stored in the HDF5 file.
    :type dataset: str

    :return: An array which was stored on disk.
    :rtype: :class numpy.ndarray:
    """
    try:
        with h5py.File(read_path, 'r') as hf:
            try:
                array = hf[dataset][:]
                return array
            except KeyError:
                warnings.warn(
                    "The {} array does not appear to exist in the file {}. "
                    "Please set a write_path keyword argument in `Model` "
                    "and the {} array will be created and then written to "
                    "that file path.".format(dataset, read_path, dataset))
                return None
    except IOError:
        warnings.warn(
            "The {} file does not appear to exist yet.".format(
                read_path))
        return None

# TODO: understand what this does and move to utilities
def read_npz_array(read_path, dataset):
    """
    Read a :class numpy.ndarray: from a HDF5 file.

    :arg read_path: The path to which the HDF5 file is written.
    :type read_path: path-like or str
    :arg dataset: The name of the dataset stored in the HDF5 file.
    :type dataset: str

    :return: An array which was stored on disk.
    :rtype: :class numpy.ndarray:
    """
    try:
        with np.load(read_path) as nz:
            try:
                array = nz[dataset]
                return array
            except KeyError:
                warnings.warn(
                    "The {} array does not appear to exist in the file {}. "
                    "Please set a write_path keyword argument in `Model` "
                    "and the {} array will be created and then written to "
                    "that file path.".format(dataset, read_path, dataset))
    except IOError:
        warnings.warn(
            "The {} file does not appear to exist yet.".format(
                read_path))
        return None

# TODO: delete this
def rbf_regression(x, y, rbfi):
    """Return the mse of the rbf interpolation of x values and y."""
    error = y - rbfi(x)
    mse = np.linalg.norm(error)
    return mse

# TODO: delete this
def linear_regression(x, y, split):
    """Plot a straight line using linear regression."""
    # Reshape the data
    x = np.array(x).reshape(-1, 1)
    y = np.array(y).reshape(-1, 1)
    # Split the data into training/testing sets
    x_train = x[split:]
    x_test = x[:split]

    # Split the targets into training/testing sets
    y_train = y[split:]
    y_test = y[:split]

    # Create linear regression object
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(x_train, y_train)

    # Make predictions using the testing set
    y_pred = regr.predict(x_test)

    # The coefficients
    print('Coefficients: \n', regr.coef_)
    # The mean squared error
    print('Mean squared error: %.2f'
          % mean_squared_error(y_test, y_pred))
    # The coefficient of determination: 1 is perfect prediction
    print('Coefficient of determination: %.2f'
          % r2_score(y_test, y_pred))

    return regr.coef_[0][0], r2_score(y_test, y_pred)

# TODO: why is this here?
def is_density(x):
    """Determine the density of the node."""
    density_concrete = 2346.0
    return density_concrete


def main():
    """Conduct a peridynamics simulation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("mesh_file_name", help="run example on a given mesh file name")
    parser.add_argument('--profile', action='store_const', const=True)
    args = parser.parse_args()

    mesh_file = pathlib.Path(__file__).parent.absolute() / args.mesh_file_name
    write_path_solutions = (pathlib.Path(__file__).parent.absolute() /
                            args.mesh_file_name.replace('.msh', ''))
    write_path_model = (pathlib.Path(__file__).parent.absolute() / str(
        args.mesh_file_name.replace('.msh', '') + "_model.h5"))
    dx = dxs[args.mesh_file_name]   # TODO: what does this represent? Grid resolution?

    # TODO: is this only needed for linear regression?
    # Find the rbf interpolation of the data
    cmod = read_array(read_path, "CMOD")
    force = read_array(read_path, "force_mean")

    length = 1200
    # Get the interpolated function for the force-displacement
    # rbfi = inter.Rbf(cmod[:length], force[:length])

    # TODO: move to an input module
    # Constants
    horizon = dx * np.pi
    s_0 = s_0s[args.mesh_file_name]
    s_1 = s_1s[args.mesh_file_name]
    s_c = s_cs[args.mesh_file_name]
    # Calibrated value
    beta = 1./4  # 0.4
    c = cs[args.mesh_file_name]
    c_1 = (beta * c * s_0 - c * s_0) / (s_1 - s_0)
    c_2 = (- beta * c * s_0) / (s_c - s_1)
    critical_stretch_ = [np.float64(s_0), np.float64(s_1), np.float64(s_c)]
    critical_stretch_nf = [1000. * np.float64(s_0), 1000. * np.float64(s_1), 1000. * np.float64(s_c)]  # TODO: what is the purpose of this line?
    bond_stiffness_ = [np.float64(c), np.float64(c_1), np.float64(c_2)]
    bond_stiffness_nf = [np.float64(c), np.float64(c), np.float64(c)]
    bond_stiffness = [bond_stiffness_, bond_stiffness_nf]
    critical_stretch = [critical_stretch_, critical_stretch_nf]

    bond_stiffness = c
    critical_stretch = s_0

    if args.profile:
        profile = cProfile.Profile()
        profile.enable()

    # Example function for calculating the boundary conditions magnitudes
    # displacement_bc_array, *_ = calc_boundary_conditions_magnitudes(
    #     70000, 5e-9, 1.75e-4, 1.75e-4)
    # displacement_bc_array, *_ = calc_boundary_conditions_magnitudes(
    #     100000, 1.0e-9, 4.0e-5, 4.0e-5)
    # displacement_bc_array, *_ = calc_boundary_conditions_magnitudes(
    #     200000, 4.5e-10, 4.5e-5, 4.5e-5)
    # displacement_bc_array, *_ = calc_boundary_conditions_magnitudes(
    #     250000, 3.5e-10, 4.4e-5, 4.4e-5)  # (250000, 3.5e-10, 4.4e-5, 4.4e-5)
    displacement_bc_array = smooth_step_data(0, 100000, 0, 1.5e-4)

    saf_fac = safety_factor[args.mesh_file_name]
    dt = (np.power(
        2.0 * 2400.0 / (
            (4 / 3) * np.pi * np.power(horizon, 3.0) * 8 * c),
        0.5) * saf_fac)
    damping = dampings[args.mesh_file_name]
    print('dt=', dt, ', safety_fac=', saf_fac, 'damping=', damping)
    integrator = VelocityVerletCL(dt=dt, damping=damping)  # TODO: (dt=dt, damping=damping, context=1)
    # integrator = Euler_jit(dt=dt)

    # TODO: simplify. This functionality should be moved into a separate input module
    # Try reading volume, density, family and connectivity arrays from
    # the file ./..._model.h5
    # volume is an (nnodes, ) :class:`np.ndarray` of particle volumes, where
    # nnodes is the number of particles in the .msh file
    volume = np.power(dx, 3) * np.ones(nnodes[args.mesh_file_name], dtype=np.float64)
    # density is an (nnodes, ) :class:`np.ndarray` of particle densities
    density = read_model(write_path_model, "density")
    # family is an (nnodes, ) :class:`np.ndarray` of initial number of
    # neighbours for each particle
    family = read_model(write_path_model, "family")
    # nlist is an (nnodes, max_neigh) :class:`np.ndarray` of the neighbours
    # for each particle. Each neigbour is given an integer i.d. in the range
    # [0, nnodes). max_neigh is atleast as large as np.max(family)
    nlist = read_model(write_path_model, "nlist")
    # n_neigh is an (nnodes, ) :class:`np.ndarray` of current number of
    # neighbours for each particle.
    n_neigh = read_model(write_path_model, "n_neigh")
    # The connectivity of the model is the tuple (nlist, n_neigh)
    if (nlist is not None) and (n_neigh is not None):
        connectivity = (nlist, n_neigh)
    else:
        connectivity = None
    # Bond types
    bond_types = read_model(
        write_path_model, "bond_types")

    # TODO: simplify
    if ((volume is not None) and
            (density is not None) and
            (family is not None) and
            (connectivity is not None) and
            (bond_types is not None)):
        # Model has been initiated before, so to avoid calculating volume,
        # family and connectivity arrays again, we can pass them as arguments
        # to the Model class
        model = Model(
            mesh_file, integrator=integrator, horizon=horizon,
            critical_stretch=critical_stretch, bond_stiffness=bond_stiffness,
            dimensions=3, family=family,
            volume=volume, connectivity=connectivity,
            bond_types=bond_types,
            density=density,
            is_displacement_boundary=is_displacement_boundary_5mm,
            is_tip=is_tip_5mm,
            write_path=write_path_model)
    else:
        # This is the first time that Model has been initiated, so the volume,
        # family and connectivity = (nlist, n_neigh) arrays will be calculated
        # and written to the file at location "write_path_model"
        model = Model(
            mesh_file, integrator=integrator, horizon=horizon,
            critical_stretch=critical_stretch, bond_stiffness=bond_stiffness,
            dimensions=3,
            is_density=is_density,
            volume=volume,
            is_bond_type=is_bond_type_5mm,
            is_displacement_boundary=is_displacement_boundary_5mm,
            is_tip=is_tip_5mm,
            write_path=write_path_model)
        density = read_model(write_path_model, "density")
        # family is an (nnodes, ) :class:`np.ndarray` of initial number of
        # neighbours for each particle
        family = read_model(write_path_model, "family")
        # nlist is an (nnodes, max_neigh) :class:`np.ndarray` of the neighbours
        # for each particle. Each neigbour is given an integer i.d. in the range
        # [0, nnodes). max_neigh is atleast as large as np.max(family)
        nlist = read_model(write_path_model, "nlist")
        # n_neigh is an (nnodes, ) :class:`np.ndarray` of current number of
        # neighbours for each particle.
        n_neigh = read_model(write_path_model, "n_neigh")
        # The connectivity of the model is the tuple (nlist, n_neigh)
        if ((nlist is not None) and (n_neigh is not None)):
            connectivity = (nlist, n_neigh)
        else:
            connectivity = None
        # Bond types
        bond_types = read_model(write_path_model, "bond_types")

    # TODO: why define this so late in the script?
    first = 0
    last = -1
    steps = 100000   # 250,000
    write = 1000     # TODO: Ben had this set at 1 - why?

    # TODO: what is this code doing?
    # Try to read the restart simulation from the variables
    first_step = 120000  # TODO: what is this doing?
    nlist = read_npz_array(read_path_restart, "nlist")
    n_neigh = read_npz_array(read_path_restart, "n_neigh")
    u = read_npz_array(read_path_restart, "u")
    ud = read_npz_array(read_path_restart, "ud")
    connectivity = (nlist, n_neigh)
    print(nlist, n_neigh, u, ud)

    # TODO: how is nregimes set?
    if ((nlist is not None) and
            (n_neigh is not None) and
            (u is not None) and
            (ud is not None)):
        steps = 60000  # TODO: why is steps defined again?
        # 'unpause' a simulation given the state variables
        (u, coords, damage, connectivity, f, ud, data) = model.simulate(
            bond_stiffness=bond_stiffness,
            critical_stretch=critical_stretch,
            u=u,
            ud=ud,
            connectivity=connectivity,
            first_step=first_step,
            steps=steps,
            displacement_bc_magnitudes=displacement_bc_array,
            write=write
        )
    else:
        # Start from the first timestep
        # TODO: (u, coords, damage, connectivity, f, ud, data) expected 7, got 6
        (u, damage, connectivity, f, ud, data) = model.simulate(
            bond_stiffness=bond_stiffness,
            critical_stretch=critical_stretch,
            steps=steps,
            displacement_bc_magnitudes=displacement_bc_array,
            write=write
            )

    # TODO: outputting data needs to be massively simplified
    force = np.array(data['force']['body_force'][first:last]) / 1000.
    deflection = 1000. * np.array(data['deflection']['displacement'][first:last])
    left_displacement = 1000. * np.array(data['CMOD_left']['displacement'][first:last])
    right_displacement = 1000. * np.array(data['CMOD_right']['displacement'][first:last])
    CMOD = np.subtract(right_displacement, left_displacement)
    # mse_l = rbf_regression(CMOD, force, rbfi)
    mdic = {"CMOD": CMOD, "load": force}        # edit MH
    scipy.io.savemat('load_CMOD.mat', mdic)     # edit MH
    plt.plot(CMOD, force)                       # edit MH
    plt.show()                                  # edit MH
    print('mse = ', mse_l)

    np.savez(write_path_solutions / "state.npz",
             u=u,
             # coords=coords,
             damage=damage,
             nlist=connectivity[0],
             n_neigh=connectivity[1],
             f=f,
             ud=ud)

    np.savez(write_path_solutions / "model.npz",
             step=np.array(data['model']['step']),
             displacement=np.array(data['model']['displacement']),
             velocity=np.array(data['model']['velocity']),
             acceleration=np.array(data['model']['acceleration']),
             force=np.array(data['model']['force']),
             body_force=np.array(data['model']['body_force']),
             damage=np.array(data['model']['damage'])
             )
    np.savez(write_path_solutions / "force.npz",
             displacement=np.array(data['force']['displacement']),
             velocity=np.array(data['force']['velocity']),
             acceleration=np.array(data['force']['acceleration']),
             force=np.array(data['force']['force']),
             body_force=np.array(data['force']['body_force']),
             )
    np.savez(write_path_solutions / "deflection.npz",
             displacement=np.array(data['deflection']['displacement']),
             velocity=np.array(data['deflection']['velocity']),
             acceleration=np.array(data['deflection']['acceleration']),
             force=np.array(data['deflection']['force']),
             body_force=np.array(data['deflection']['body_force']),
             )
    np.savez(write_path_solutions / "CMOD_left.npz",
             displacement=np.array(data['CMOD_left']['displacement']),
             velocity=np.array(data['CMOD_left']['velocity']),
             acceleration=np.array(data['CMOD_left']['acceleration']),
             force=np.array(data['CMOD_left']['force']),
             body_force=np.array(data['CMOD_left']['body_force']),
             )
    np.savez(write_path_solutions / "CMOD_right.npz",
             displacement=np.array(data['CMOD_right']['displacement']),
             velocity=np.array(data['CMOD_right']['velocity']),
             acceleration=np.array(data['CMOD_right']['acceleration']),
             force=np.array(data['CMOD_right']['force']),
             body_force=np.array(data['CMOD_right']['body_force']),
             )

    try:
        # Write data to disk
        write_array(
            write_path_solutions /
            "data.h5", "force", np.array(force))
        write_array(
            write_path_solutions
            / "data.h5", "CMOD", np.array(CMOD))

    except:
        os.remove(write_path_solutions / "data.h5")
        # Write data to disk
        write_array(
            write_path_solutions /
            "data.h5", "force", np.array(force))
        write_array(
            write_path_solutions
            / "data.h5", "CMOD", np.array(CMOD))

    if args.profile:
        profile.disable()
        s = StringIO()
        stats = Stats(profile, stream=s).sort_stats(SortKey.CUMULATIVE)
        stats.print_stats(.05)
        print(s.getvalue())


if __name__ == "__main__":
    main()
