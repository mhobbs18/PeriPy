"""
Created on Sun Nov 10 16:25:58 2019

@author: Ben Boys
"""
#import vtk as vtk
#import time
#import pyopencl as cl
# TODO: add argument on command line that gives option to plot results or not,
# as some systems won't have matplotlib installed.
#import matplotlib.pyplot as plt
import argparse
import cProfile
from io import StringIO
import numpy as np
import pathlib
from peridynamics import Model
from peridynamics.model import initial_crack_helper
from peridynamics.integrators import RungeKuttaOpenCL
from pstats import SortKey, Stats
#import matplotlib.pyplot as plt
import time
import sys
sys.path.insert(1, './peridynamics/kernels')
import shutil
import os


mesh_file_name = '1000beam3DT.msh'
mesh_file = pathlib.Path(__file__).parent.absolute() / mesh_file_name


token_problems = ['test.msh', 'debug3D.msh', 'debug3D2.msh']
verification_problems = ['1000beam2D.msh', '1000beam3D.msh', '1000beam3DT.msh']
benchmark_problems = ['3300beam.msh']

plain = 1
        
@initial_crack_helper
def is_crack(x, y):
    output = 0
    crack_length = 0.3
    p1 = x
    p2 = y
    if x[0] > y[0]:
        p2 = x
        p1 = y
    # 1e-6 makes it fall one side of central line of particles
    if p1[0] < 0.5 + 1e-6 and p2[0] > 0.5 + 1e-6:
        # draw a straight line between them
        m = (p2[1] - p1[1]) / (p2[0] - p1[0])
        c = p1[1] - m * p1[0]
        # height a x = 0.5
        height = m * 0.5 + c
        if (height > 0.5 * (1 - crack_length)
                and height < 0.5 * (1 + crack_length)):
            output = 1
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
        if x[0] > 1.0 - 1./3 * horizon:
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
        if plain == 1:
            return False
        elif beam_type == 1:
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

def bond_type(x, y):
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
    # Function which marks constrain particles
    # 2 == NO BOUNDARY CONDITION (the number here is an arbitrary choice)
    # -1 == DISPLACEMENT LOADED IN -ve direction
    #  1 == DISPLACEMENT LOADED IN +ve direction
    #  0 == FIXED (DIRICHLET) BOUNDARY
    
    if mesh_file_name in token_problems:
        # Does not live on a boundary
        bnd = 2
        # Does live on boundary
        if x[0] < 1.5 * horizon:
            bnd = 0
            
        elif x[0] > 3.0 - 1.5 * horizon:
            bnd = 1
    elif mesh_file_name in verification_problems:
        # Does not live on a boundary
        bnd = 2
        # Does live on boundary
        if x[0] < 1.5* horizon:
            bnd = 0
        if x[0] > 1.0 - 1.* horizon:
            if x[2] > 0.2 - 1.* horizon:
                bnd = 1
    elif mesh_file_name == '3300beam.msh':
        bnd = 2
        if x[0] < 1.5 * horizon:
            bnd = 0
        if x[0] > 3.3 - 0.3* horizon:
            if x[2] > 0.6 - 0.3*horizon:
                bnd = 1
    return bnd

def is_forces_boundary(horizon, x):
    # Function which marks body force loaded particles
    # 2 == NO BOUNDARY CONDITION (the number here is an arbitrary choice)
    # -1 == FORCE LOADED IN -ve direction
    #  1 == FORCE LOADED IN +ve direction
    if mesh_file_name in token_problems:
        bnd = 2
        if x[0] > 1.0 - 1.5 * horizon:
            bnd = 1
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
    """ """
    load_rate = 2e-9
    theta = 18.75
    # initiate
    model.bctypes = np.zeros((model.nnodes, model.DPN), dtype=np.intc)
    model.bcvalues = np.zeros((model.nnodes, model.DPN), dtype=np.float64)
    model.tiptypes = np.zeros(model.nnodes, dtype=np.intc)
    
    # Find the boundary nodes and apply the displacement values
    for i in range(0, model.nnodes):
        # define boundary types and values
        bnd = is_boundary(model.PD_HORIZON, model.coords[i][:])
        model.bctypes[i, 0] = np.intc((bnd))
        model.bctypes[i, 1] = np.intc((bnd))
        model.bctypes[i, 2] = np.intc((bnd))
        model.bcvalues[i, 2] = np.float64(bnd * 0.5 * load_rate)
        model.bcvalues[i, 0] = np.float64(bnd * -0.5/theta * load_rate)
        
        # also define tip here
        tip = is_tip(model.PD_HORIZON, model.coords[i][:])
        model.tiptypes[i] = np.intc((tip))
    
def boundary_forces_function(model):
    """ Initiate boundary forces if constant
    """
    model.force_bctypes = np.zeros((model.nnodes, model.DPN), dtype=np.intc)
    model.force_bcvalues = np.zeros((model.nnodes, model.DPN), dtype=np.float64)
    
    # Find the force boundary nodes and find amount of boundary nodes
    num_force_bc_nodes = 0
    for i in range(0, model.nnodes):
        bnd = is_forces_boundary(model.PD_HORIZON, model.coords[i][:])
        if bnd == -1:
            num_force_bc_nodes += 1
        elif bnd == 1:
            num_force_bc_nodes += 1
        model.force_bctypes[i, 0] = np.intc((bnd))
        model.force_bctypes[i, 1] = np.intc((bnd))
        model.force_bctypes[i, 2] = np.intc((bnd))
        
    model.num_force_bc_nodes = num_force_bc_nodes
    
    # Calculate initial forces
    model.force_bcvalues = np.zeros((model.nnodes, model.DPN), dtype=np.float64)
    load_scale = 0.0
    for i in range(0, model.nnodes):
        bnd = is_forces_boundary(model.PD_HORIZON, model.coords[i][:])
        if bnd == 1:
            pass
        elif bnd == -1:
            model.force_bcvalues[i, 2] = np.float64(1.* bnd * model.max_reaction * load_scale / (model.num_force_bc_nodes))

   
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
    model = Model(bond_type=bond_type, initial_crack=is_crack)
    
    # Set force and displacement boundary conditions
    boundary_function(model)
    boundary_forces_function(model)
    
    integrator = RungeKuttaOpenCL(model)
    
    # delete output directory contents, this is probably unsafe?
    shutil.rmtree('./output', ignore_errors=False)
    os.mkdir('./output')
    
    damage_data, tip_displacement_data = model.simulate(model, steps=10, integrator=integrator, write=1, toolbar=0)
# =============================================================================
#     plt.figure(1)
#     plt.title('damage over time')
#     plt.plot(damage_data)
#     plt.figure(2)
#     plt.title('tip displacement over time')
#     plt.plot(tip_displacement_data)
#     plt.show()
# =============================================================================
    print('TOTAL TIME REQUIRED {}'.format(time.time() - st))
    print(damage_data)
    print(tip_displacement_data)
    if args.profile:
        profile.disable()
        s = StringIO()
        stats = Stats(profile, stream=s).sort_stats(SortKey.CUMULATIVE)
        stats.print_stats()
        print(s.getvalue())


if __name__ == "__main__":
    main()

