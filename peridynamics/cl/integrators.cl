#pragma OPENCL EXTENSION cl_khr_fp64 : enable


__kernel void euler(__global double* r, __global const double* f, double dt,
                    double dampening) {
    /* Conduct one iteration of the Euler integrator.
     *
     * r - An (n,3) array of the coordinates of each node.
     * f - An (n,3) array of the force one each node.
     * dt - The time step.
     * dampening - The dampening factor. */
    int i = get_global_id(0);

    #pragma unroll
    for (int dim=0; dim<3; dim++) {
        r[i*3 + dim] = r[i*3 + dim] + dt * f[i*3 + dim] * dampening;
    }
}
