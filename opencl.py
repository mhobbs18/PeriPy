"""
Created on Sun Nov 10 16:25:58 2019

@author: Ben Boys
"""
import argparse
import cProfile
from io import StringIO
import numpy as np
import pathlib
from peridynamics import OpenCL
from peridynamics.model import initial_crack_helper
from peridynamics.integrators import HeunEulerNew
from pstats import SortKey, Stats
# TODO: add argument on command line that gives option to plot results or not,
# as some systems won't have matplotlib installed.
#import matplotlib.pyplot as plt
import time
import shutil
import os

mesh_file_name = '3300beam.msh'
mesh_file = pathlib.Path(__file__).parent.absolute() / mesh_file_name

token_problems = ['test.msh', 'debug3D.msh', 'debug3D2.msh']
verification_problems = ['1000beam2D.msh', '1000beam3D.msh', '1000beam3DT.msh']
benchmark_problems = ['3300beam.msh']

@initial_crack_helper
def is_crack(x, y):
    output = 0
    return output

def is_tip(horizon, x):
    output = 0
    if mesh_file_name in verification_problems:
        if x[0] > 1.0 - 1./3 * horizon:
            output = 1
    elif mesh_file_name in benchmark_problems:
        if x[0] > 3.3 - 0.5 * horizon:
            output = 1
    elif mesh_file_name in token_problems:
        if x[0] > 1.0 - 1. * horizon:
            output = 1
    return output

def is_rebar(p):
    """ Function to determine whether the node coordinate is rebar
    """
    p = p[1:] # y and z coordinates for this node
    if mesh_file_name == '3300beam.msh':
        bar_centers = [
            # Compressive bars 25mm of cover
            np.array((0.031, 0.031)),
            np.array((0.219, 0.031)),

            # Tensile bars 25mm of cover
            np.array((0.03825, 0.569)),
            np.array((0.21175, 0.569))]

        rad_c = 0.006
        rad_t = 0.01325

        radii = [
            rad_c,
            rad_c,
            rad_t,
            rad_t]

        costs = [ np.sum(np.square(cent - p) - (np.square(rad))) for cent, rad in zip(bar_centers, radii) ]
        if any( c <= 0 for c in costs ):
            return True
        else:
            return False
    elif mesh_file_name == '1000beam3DT.msh':
        # Beam type 1 for flexural failure beam
        # Beam type 2 for shear failure beam
        beam_type = 2
        if beam_type == 1:
            bar_centers = [
                    # Tensile bars 25mm of cover, WARNING: only gives 21.8mm inner spacing of bars
                    np.array((0.0321, 0.185)),
                    np.array((0.0679, 0.185))]
            rad_t = 0.00705236
            
            radii = [
                    rad_t,
                    rad_t]
            costs = [ np.sum(np.square(cent - p) - (np.square(rad))) for cent, rad in zip(bar_centers, radii) ]
            if any( c <= 0 for c in costs ):
                return True
            else:
                return False
        elif beam_type ==2:
            bar_centers = [
                    # Tensile bars 25mm of cover, WARNING: only gives 7.6mm inner spacing of bars
                    np.array((0.0356, 0.185)),
                    np.array((0.0644, 0.185))]
            rad_t = 0.0105786
            
            radii = [
                    rad_t,
                    rad_t]
            costs = [ np.sum(np.square(cent - p) - (np.square(rad))) for cent, rad in zip(bar_centers, radii) ]
            if any( c <= 0 for c in costs ):
                return True
            else:
                return False
    else:
        return False

def bond_type(x, y):
    """ 
    Determines bond type given pair of node coordinates.
    Usage:
        'plain = 1' will return a plain concrete bond for all bonds, an so a
    plain concrete beam.
        'plain = 0' will return a concrete beam with some rebar as specified
        in "is_rebar()"
    """
    plain = 0
    output = 0 # default to concrete
    bool1 = is_rebar(x)
    bool2 = is_rebar(y)
    if plain == 1:
        output = 'concrete'
    elif bool1 and bool2:
        output = 'steel'
    elif bool1 != bool2:
        output = 'interface'
    else:
        output = 'concrete'
    return output

def is_boundary(horizon, x):
    """
    Function which marks displacement boundary constrained particles
    2 is no boundary condition (the number here is an arbitrary choice)
    -1 is displacement loaded IN -ve direction
    1 is displacement loaded IN +ve direction
    0 is clamped boundary
    """
    if mesh_file_name == '3300beam.msh':
        bnd = [2, 2, 2]
        if x[0] < 1.5 * horizon:
            bnd[0] = 0
            bnd[1] = 0
            bnd[2] = 0
        if x[0] > 3.3 - 0.2* horizon:
            bnd[2] = 1
    return bnd

def is_forces_boundary(horizon, x):
    """
    Marks types of body force on the particles
    2 is no boundary condition (the number here is an arbitrary choice)
    -1 is force loaded IN -ve direction
    1 is force loaded IN +ve direction
    """
    if mesh_file_name in token_problems:
        bnd = 2
        if x[0] > 1.0 - 1.5 * horizon:
            bnd = 2
    elif mesh_file_name == '1000beam2D.msh':
        bnd = 2
        if x[1] > 0.2 - 1./3 * horizon:
            bnd = 2
    elif mesh_file_name == '1000beam3DT.msh':
        bnd = 2
        if x[2] > 0.2 - 1. * horizon:
            bnd = -1
    elif mesh_file_name in verification_problems:
        bnd = 2
        if x[0] > 1.0 - 1. * horizon:
            bnd = -1
    elif mesh_file_name == '3300beam.msh':
        bnd = 2
        if x[2] > 0.6 - 1. * horizon:
            bnd = 2
    return bnd

def boundary_function(model):
    """ 
    Initiates displacement boundary conditions,
    also define the 'tip' (for plotting displacements)
    """
    load_rate = 1e-8
    #theta = 18.75
    # initiate
    model.bc_types = np.zeros((model.nnodes, model.degrees_freedom), dtype=np.intc)
    model.bc_values = np.zeros((model.nnodes, model.degrees_freedom), dtype=np.float64)
    model.tip_types = np.zeros(model.nnodes, dtype=np.intc)

    # Find the boundary nodes and apply the displacement values
    for i in range(0, model.nnodes):
        # Define boundary types and values
        bnd = is_boundary(model.horizon, model.coords[i][:])
        model.bc_types[i, 0] = np.intc(bnd[0])
        model.bc_types[i, 1] = np.intc(bnd[1])
        model.bc_types[i, 2] = np.intc((bnd[2]))
        model.bc_values[i, 0] = np.float64(bnd[0] * 0.5 * load_rate)
        model.bc_values[i, 1] = np.float64(bnd[1] * 0.5 * load_rate)
        model.bc_values[i, 2] = np.float64(bnd[2] * 0.5 * load_rate)
        # Define tip here
        tip = is_tip(model.horizon, model.coords[i][:])
        model.tip_types[i] = np.intc(tip)
    print(np.max(model.tip_types), 'max_tip_types')

def boundary_forces_function(model):
    """ 
    Initiates boundary forces
    """
    model.force_bc_types = np.zeros((model.nnodes, model.degrees_freedom), dtype=np.intc)
    model.force_bc_values = np.zeros((model.nnodes, model.degrees_freedom), dtype=np.float64)

    # Find the force boundary nodes and find amount of boundary nodes
    num_force_bc_nodes = 0
    for i in range(0, model.nnodes):
        bnd = is_forces_boundary(model.horizon, model.coords[i][:])
        if bnd == -1:
            num_force_bc_nodes += 1
        elif bnd == 1:
            num_force_bc_nodes += 1
        model.force_bc_types[i, 0] = np.intc((bnd))
        model.force_bc_types[i, 1] = np.intc((bnd))
        model.force_bc_types[i, 2] = np.intc((bnd))

    model.num_force_bc_nodes = num_force_bc_nodes

    # Calculate initial forces
    model.force_bc_values = np.zeros((model.nnodes, model.degrees_freedom), dtype=np.float64)
    load_scale = 0.0
    for i in range(0, model.nnodes):
        bnd = is_forces_boundary(model.horizon, model.coords[i][:])
        if bnd == 1:
            pass
        elif bnd == -1:
            model.force_bc_values[i, 2] = np.float64(1.* bnd * model.max_reaction * load_scale / (model.num_force_bc_nodes))

def main():
    """
    3D canteliver beam peridynamics simulation
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', action='store_const', const=True)
    args = parser.parse_args()

    if args.profile:
        profile = cProfile.Profile()
        profile.enable()

    st = time.time()

    volume_total = 3.3 * 0.6 * 0.25
    density_concrete = 2400
    self_weight = 1.*density_concrete * volume_total * 9.81
    youngs_modulus_concrete = 1.*22e9
    youngs_modulus_steel = 1.*210e9
    tensile_strength_concrete = 2.6e6
# =============================================================================
#     # Sength scale for covariance matrix
#     l = 1e-2
#     # Vertical scale of the covariance matrix
#     nu = 9e-20
#     model = OpenCLProbabilistic(mesh_file_name, volume_total, nu, l, bond_type=bond_type, initial_crack=is_crack)
# =============================================================================
    model = OpenCL(mesh_file_name, volume_total, bond_type=bond_type, initial_crack=is_crack, dimensions=3)
    # Set simulation parameters
    # not a transfinite mesh
    model.transfinite = 0
    # do precise stiffness correction factors
    model.precise_stiffness_correction = 1
    # Only one material in this example, that is 'concrete'
    model.density = density_concrete
    dx = np.power(1.*volume_total/model.nnodes,1./(model.dimensions))
    model.horizon = dx * np.pi 
    model.family_volume =(4./3)*np.pi*np.power(model.horizon, 3)
    model.damping = 2.0e6 # damping term
    # Peridynamic bond stiffness, c
    bulk_modulus_concrete = youngs_modulus_concrete/ (3* (1 - 2*model.poisson_ratio))
    bulk_modulus_steel = youngs_modulus_steel / (3* (1- 2*model.poisson_ratio))
    model.bond_stiffness_concrete = (
    np.double((18.00 * bulk_modulus_concrete) /
    (np.pi * np.power(model.horizon, 4)))
    )
    model.bond_stiffness_steel = (
    np.double((18.00 * bulk_modulus_steel) /
    (np.pi * np.power(model.horizon, 4)))
    )
    model.critical_strain_concrete = (
    np.double(tensile_strength_concrete /
    youngs_modulus_concrete)
    )
    #model.critical_strain_concrete = np.double(0.000533) # check this value
    model.critical_strain_steel = np.double(0.01)
    model.crackLength = np.double(0.3)
    #saf_fac = 0.2 # Typical values 0.70 to 0.95 (Sandia PeridynamicSoftwareRoadmap)
    #model.dt = (
    # 0.8 * np.power( 2.0 * density_concrete * dx / 
    # (np.pi * np.power(model.horizon, 2.0) * dx * model.bond_stiffness_concrete), 0.5)
    # * saf_fac
    # )
    model.dt = 1.5e-8
    model.max_reaction = 1.* self_weight # in newtons, about 85 times self weight
    model.load_scale_rate = 1/1000

    # Set force and displacement boundary conditions
    boundary_function(model)
    boundary_forces_function(model)

    integrator = HeunEulerNew(model)

    # delete output directory contents, this is probably unsafe?
    shutil.rmtree('./output', ignore_errors=False)
    os.mkdir('./output')

    damage_data, damage_sum_data, tip_displacement_data = model.simulate(model, sample=1, steps=20000, integrator=integrator, write=20, toolbar=0)
# =============================================================================
#     plt.figure(1)
#     plt.title('damage over time')
#     plt.plot(damage_data)
#     plt.figure(2)
#     plt.title('tip displacement over time')
#     plt.plot(tip_displacement_data)
#     plt.show()
# =============================================================================
    print(damage_sum_data)
    print('TOTAL TIME REQUIRED {}'.format(time.time() - st))
    if args.profile:
        profile.disable()
        s = StringIO()
        stats = Stats(profile, stream=s).sort_stats(SortKey.CUMULATIVE)
        stats.print_stats()
        print(s.getvalue())


if __name__ == "__main__":
    main()