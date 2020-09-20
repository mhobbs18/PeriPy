__kernel void
	update_displacement(
        __global float const* force,
        __global float* u,
        __global float* ud,
        __global float* udd,
        __global int const* bc_types,
		__global float const* bc_values,
        __global float const* densities,
        float bc_scale,
        float damping,
        float dt
	){
    /* Calculate the dispalcement and velocity of each node using an
     * Euler Cromer integrator.
     *
     * force - An (n,3) array of the forces of each node.
     * u - An (n,3) array of the current displacements of each node.
     * ud - An (n,3) array of the current velocities of each node.
     * udd - An (n,3) array of the accelerations of each node.
     * bc_types - An (n,3) array of the boundary condition types.
     * bc_values - An (n,3) array of the boundary condition values applied to the nodes.
     * densties - An (n,3) array of the density values of the nodes.
     * bc_scale - The scalar value applied to the displacement BCs.
     * damping - The dynamics relaxation damping constant in [kg/(m^3 s)].
     * dt - The time step in [s]. */
	const int i = get_global_id(0);

    float uddi = (force[i] - damping * ud[i]) / densities[i];
    udd[i] = uddi;
    ud[i] += uddi * dt;
    u[i] = (bc_types[i] == 0 ? (u[i] + dt * ud[i]) : (bc_scale * bc_values[i]));
}
