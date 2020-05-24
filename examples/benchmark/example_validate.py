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
from peridynamics import OpenCLProbabilistic
from peridynamics.model import initial_crack_helper
from peridynamics.integrators import EulerCromer
from peridynamics.integrators import EulerCromerOptimised
from peridynamics.integrators import EulerCromerOptimisedLumped2
from matplotlib import rc
from pstats import SortKey, Stats
import matplotlib.pyplot as plt
from scipy import interpolate
import time
import shutil
import os

mesh_file_name = 'validate.msh'
mesh_file = pathlib.Path(__file__).parent.absolute() / mesh_file_name

@initial_crack_helper
def is_crack(x, y):
    output = 0
    crack_length = 0.0
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
    if mesh_file_name == 'validate.msh':
        if x[0] > 0.350/2 - 0.2*horizon:
            if x[0] < 0.350/2 + 0.2*horizon:
                if x[1] > 0.350/2 - 0.2*horizon:
                    if x[1] < 0.350/2 + 0.2*horizon:
                        output = 1
    return output

def is_rebar(p):
    """ Function to determine whether the node coordinate is rebar
    """
    p = p[1:] # y and z coordinates for this node
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
    plain = 1
    output = 'concrete' # default to concrete
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
    bnd = [2, 2, 2]
    if x[0] > 0.350 - 0.2 * horizon:
        bnd[0] = 0
        if (x[1] < 0.350/2 + 0.2*horizon) and (x[1] >0.350/2 -0.2*horizon):
            bnd[1] = 0
    return bnd

def is_forces_boundary(horizon, x):
    """
    Marks types of body force on the particles
    2 is no boundary condition (the number here is an arbitrary choice)
    -1 is force loaded IN -ve direction
    1 is force loaded IN +ve direction
    """
    if mesh_file_name == 'validate.msh':
        bnd = [2, 2, 2]
        if x[0] < 0.2 * horizon:
            bnd[0] = -1
    return bnd

def boundary_function(model, displacement_rate):
    """ 
    Initiates displacement boundary conditions,
    also define the 'tip' (for plotting displacements)
    """
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
        model.bc_values[i, 0] = np.float64(bnd[0] * displacement_rate)
        model.bc_values[i, 1] = np.float64(bnd[1] * displacement_rate)
        model.bc_values[i, 2] = np.float64(bnd[2] * displacement_rate)
        # Define tip here
        tip = is_tip(model.horizon, model.coords[i][:])
        model.tip_types[i] = np.intc(tip)

   
def boundary_forces_function(model):
    """ 
    Initiates boundary forces. The units are force per unit volume.
    """
    model.force_bc_types = np.zeros((model.nnodes, model.degrees_freedom), dtype=np.intc)
    model.force_bc_values = np.zeros((model.nnodes, model.degrees_freedom), dtype=np.float64)

    # Find the force boundary nodes and find amount of boundary nodes
    num_force_bc_nodes = 0
    for i in range(0, model.nnodes):
        bnd = is_forces_boundary(model.horizon, model.coords[i][:])
        if -1 in bnd:
            num_force_bc_nodes += 1
        elif 1 in bnd:
            num_force_bc_nodes += 1
        model.force_bc_types[i, 0] = np.intc((bnd[0]))
        model.force_bc_types[i, 1] = np.intc((bnd[1]))
        model.force_bc_types[i, 2] = np.intc((bnd[2]))
    #print('number of force loaded nodes is ', num_force_bc_nodes)
    model.num_force_bc_nodes = num_force_bc_nodes
    for i in range(0, model.nnodes):
        for j in range(model.dimensions):
            bnd = model.force_bc_types[i,j]
            if bnd != 2:
                # apply the force bc value, which is total reaction force / (num loaded nodes * node volume)
                # units are force per unit volume
                model.force_bc_values[i, j] = np.float64(bnd * model.max_reaction / (model.num_force_bc_nodes * model.V[i]))
                
def airy_stress_function(x, u, F, E, W, T, nu):
    x_x = x[0]
    x_y = x[1]
    u_x = np.divide(1.*F, E*W*T) * (x_x - W)
    u_y = np.divide(-1.*nu * F, E * T) * (np.divide(x_y, W) - 0.5)
    airy_norm = u_x**2 + u_y**2
    u_x_pd = u[0]
    u_y_pd = u[1]
    pd_norm = u_x_pd**2 + u_y_pd**2
    error = np.abs(airy_norm - pd_norm)/airy_norm
    error_x = (u_x-u_x_pd)/u_x
    error_y = (u_y-u_y_pd)/u_y
    return error, error_x, error_y

def airy_stress_function_x(x, y, F, E, W, T, nu):

    u_x = np.divide(1.*F, E*W*T) * (x - W)
    #u_y = np.divide(-1.*nu * F, E * T) * (np.divide(y, W) - 0.5)

    return u_x

def airy_stress_function_y(x, y, F, E, W, T, nu):

    u_y = np.divide(-1.*nu * F, E * T) * (np.divide(y, W) - 0.5)

    return u_y



def gridspace(model, F, E, W, T, nu):
    
    
    
    data = [[],[],[]]
    for i in range(0, model.nnodes):
        x = model.coords[i][:]
        u = model.h_un[i][:]
        u_x = u[0]
        u_y = u[1]
        data[0].append(x[:2])
        data[1].append(u_x)
        data[2].append(u_y)
        

    displacements_y = np.array(data[2])
    displacements_x = np.array(data[1])
    nodes = np.array(data[0])
    
    x_lim = 1000*np.min(displacements_x)
    y_lim = 1000*np.max(displacements_y)
    y_lim_min = 1000*np.min(displacements_y)
    print(x_lim, y_lim, y_lim_min)
    breaks = np.linspace(x_lim, 0, 100)
    
    x = np.linspace(0, 0.350, 96)
    y = np.linspace(0, 0.350, 96)
    X, Y = np.meshgrid(x,y)
    font = 20
    
    plt.rc('xtick',labelsize=10)
    plt.rc('ytick',labelsize=10)
    
    # X DIRECTION PD
    Z1 = interpolate.griddata(nodes, displacements_x, (X, Y),method='nearest')
    zlimit_min = np.min(Z1)*1000
    zlimit_max = np.max(Z1)*1000
    breaks = np.linspace(zlimit_min, zlimit_max, 100)
    print('max PD', np.min(Z1))
    cnt = plt.contourf(x*1000,y*1000,Z1*1000, breaks, cmap = 'seismic')
    cb = plt.colorbar(ticks=[zlimit_min, zlimit_max, 1])
    cb.ax.tick_params(labelsize='large')
    plt.title(r'Peridynamics solution, $u_x$', fontdict = {'fontsize': font})
    plt.xlabel(r'$x$ [mm]', fontdict = {'fontsize': font})
    plt.ylabel(r'$y$ [mm]', fontdict = {'fontsize': font})
    for c in cnt.collections:
        c.set_edgecolor("face")
    plt.savefig('pd_x.pdf', bbox_inches='tight')
    plt.show()
    
    # AIRY
    Z2 = airy_stress_function_x(X, Y, F, E, W, T, nu)
    zlimit_min = np.min(Z2)*1000
    zlimit_max = np.max(Z2)*1000
    breaks = np.linspace(zlimit_min, zlimit_max, 100)
    print('max airy', np.min(Z2))
    cnt = plt.contourf(x*1000,y*1000,Z2*1000, breaks, cmap = 'seismic')
    plt.title('Classical continuum mechanics \n' + r'solution, $u_x$', fontdict = {'fontsize': font})
    plt.xlabel(r'$x$ [mm]', fontdict = {'fontsize': font})
    plt.ylabel(r'$y$ [mm]', fontdict = {'fontsize': font})
    plt.colorbar(ticks=[zlimit_min, zlimit_max , 1])
    for c in cnt.collections:
        c.set_edgecolor("face")
    plt.savefig('ccm_x.pdf', bbox_inches='tight')
    plt.show()
    
    #ERROR
    Error = (Z1-Z2)*1000*100/12.50
    zlimit_min = np.min((Error))
    zlimit_max = np.max((Error))
    breaks3 = np.linspace(zlimit_min, -zlimit_min, 100)
    cnt = plt.contourf(x*1000,y*1000,Error, breaks3, cmap = 'seismic')
    plt.title('Error as a percentage \n'+ r'of maximum displacement in $u_x$', fontdict = {'fontsize': font})
    plt.xlabel(r'$x$ [mm]', fontdict = {'fontsize': font})
    plt.ylabel(r'$y$ [mm]', fontdict = {'fontsize': font})
    plt.colorbar(ticks=[zlimit_min, 0, -zlimit_min])
    for c in cnt.collections:
        c.set_edgecolor("face")
    plt.savefig('error_x.pdf', bbox_inches='tight')
    plt.show()
    
    # Y DIRECTION PD
    Z1 = interpolate.griddata(nodes, displacements_y, (X, Y),method='nearest')
    zlimit_min = np.min(Z1)*1000
    zlimit_max = np.max(Z1)*1000
    breaks = np.linspace(zlimit_min, zlimit_max, 100)
    cnt = plt.contourf(x*1000,y*1000,Z1*1000, breaks, cmap = 'seismic')
    plt.colorbar(ticks=[zlimit_min, zlimit_max, 0])
    plt.title(r'Peridynamics solution, $u_y$', fontdict = {'fontsize': font})
    plt.xlabel(r'$x$ [mm]', fontdict = {'fontsize': font})
    plt.ylabel(r'$y$ [mm]', fontdict = {'fontsize': font})
    for c in cnt.collections:
        c.set_edgecolor("face")
    plt.savefig('pd_y.pdf', bbox_inches='tight')
    plt.show()
    
    # Y DIRECTION AIRY
    Z2 = airy_stress_function_y(X, Y, F, E, W, T, nu)
    zlimit_min = np.min(Z2)*1000
    zlimit_max = np.max(Z2)*1000
    breaks = np.linspace(zlimit_min, zlimit_max, 100)
    cnt = plt.contourf(x*1000,y*1000,Z2*1000, breaks, cmap = 'seismic')
    plt.title('Classical continuum mechanics \n' + r'solution, $u_y$', fontdict = {'fontsize': font})
    plt.xlabel(r'$x$ [mm]', fontdict = {'fontsize': font})
    plt.ylabel(r'$y$ [mm]', fontdict = {'fontsize': font})
    plt.colorbar(ticks=[zlimit_min, zlimit_max, 0])
    for c in cnt.collections:
        c.set_edgecolor("face")
    plt.savefig('ccm_y.pdf', bbox_inches='tight')
    plt.show()
    
    Error = (Z1-Z2)*1000*100/2.0625
    zlimit_max = np.max(np.abs(Error))
    breaks3 = np.linspace(-zlimit_max, zlimit_max, 100)
    cnt = plt.contourf(x*1000,y*1000,Error, breaks3, cmap = 'seismic')
    plt.title('Error as a percentage \n' + r'of maximum displacement in $u_y$', fontdict = {'fontsize': font})
    plt.xlabel(r'$x$ [mm]', fontdict = {'fontsize': font})
    plt.ylabel(r'$y$ [mm]', fontdict = {'fontsize': font})
    plt.colorbar(ticks=[-zlimit_max, 0, zlimit_max])
    for c in cnt.collections:
        c.set_edgecolor("face")
    plt.savefig('error_y.pdf', bbox_inches='tight')
    plt.show()
    

        
    

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
    poisson_ratio = 0.33
    youngs_modulus_concrete = 4000.0e6
    bulk_modulus_concrete = youngs_modulus_concrete/ (3* (1 - 2*poisson_ratio))
    
    horizon = 0.00390625*3.14
    t = 0.001
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})
    rc('text', usetex=True)
    # Set simulation parameters
# =============================================================================
#     bond_stiffness_concrete = (
#                        np.double((18.00 * bulk_modulus_concrete) /
#                                  (np.pi * np.power(horizon, 4)))
#                        ),
# =============================================================================
    model = OpenCL(mesh_file_name, 
               density = 100.0, 
               horizon = horizon,
               damping = 3000000,
               dx = 0.00390625,
               bond_stiffness_concrete = (
                       np.double((9.00 * youngs_modulus_concrete) /
                                 (np.pi * t* np.power(horizon, 3)))
                       ),
               critical_strain_concrete = 100.0,
               crack_length = 0.0,
               volume_total=0.35365 * 0.35365 * 0.001,
               bond_type=bond_type,
               network_file_name = 'Network_validate.vtk',
               initial_crack=[],
               dimensions=2,
               transfinite=1,
               precise_stiffness_correction=0)
    model.dt = np.double(1.2e-6)
    displacement_rate = 0
    model.max_reaction = 50000 # in newtons, about 85 times self weight
    model.load_scale_rate = 1/500
    # Set force and displacement boundary conditions
    boundary_function(model, displacement_rate)
    boundary_forces_function(model)

    # delete output directory contents, this is probably unsafe?
    shutil.rmtree('./output', ignore_errors=False)
    os.mkdir('./output')
    integrator = EulerCromerOptimisedLumped2(model)#, error_size_max=1e-6, error_size_min=1e-20)
    damage_sum_data, tip_displacement_data, tip_acceleration_data, tip_shear_force_data = model.simulate(model, sample=1, steps=1000, integrator=integrator, write=100, toolbar=0)
    gridspace(model, model.max_reaction, youngs_modulus_concrete, 0.350, 0.001, 0.333)
    print('damage_sum_data', damage_sum_data)
    print('displacement_data', tip_displacement_data)
    print('TOTAL TIME REQUIRED {}'.format(time.time() - st))
    plt.figure(1)
    plt.title('damage over time')
    plt.plot(damage_sum_data)
    plt.figure(2)
    plt.title('tip displacement over time')
    plt.plot(tip_displacement_data)
    plt.show()
    if args.profile:
        profile.disable()
        s = StringIO()
        stats = Stats(profile, stream=s).sort_stats(SortKey.CUMULATIVE)
        stats.print_stats()
        print(s.getvalue())

if __name__ == "__main__":
    main()