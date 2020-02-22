////////////////////////////////////////////////////////////////////////////////
//
// opencl_peridynamics.cl
//
// OpenCL Peridynamics kernels
//
// Based on code from Copyright (c) Farshid Mossaiby, 2016, 2017. Adapted for python.
//
////////////////////////////////////////////////////////////////////////////////

// Includes, project
//#include <stdio.h>
#include "opencl_enable_fp64.cl"

// Macros

#define DPN 3
// MAX_HORIZON_LENGTH, PD_DT, PD_E, PD_S0, PD_NODE_NO, PD_DPN_NODE_NO will be defined on JIT compiler's command line

// A horizon by horizon approach is chosen to proceed with the solution, in which
// no assembly of the system of equations is required.

__kernel void
	TimeMarching1(
        __global double const *Udn,
        __global double *Un,
		__global int const *BCTypes,
		__global double const *BCValues
	)
{
	const int i = get_global_id(0);

	if (i < PD_DPN_NODE_NO)
	{	// Update displacements
		Un[i] = BCTypes[i] == 2 ? Un[i] + PD_DT * (Udn[i]) : Un[i] + BCValues[i] ;
	}
}


// Calculate force using Un
__kernel void
	TimeMarching2(
        __global double *Un,
		__global double *Forces,
        __global double const *Vols,
		__global int *Horizons,
		__global double const *Nodes,
		__global double const *Stiffnesses,
        __global double const *FailStretches
	)
{
	const int i = get_global_id(0);
	const int j = get_global_id(1);

	if ((i < PD_NODE_NO) && (j > 0) && (j < MAX_HORIZON_LENGTH))
	{

		const int n = Horizons[MAX_HORIZON_LENGTH * i + j];

		if (n != -1)
		{
			const double xi_x = Nodes[DPN * n + 0] - Nodes[DPN * i + 0];  // Optimize later, doesn't need to be done every time
			const double xi_y = Nodes[DPN * n + 1] - Nodes[DPN * i + 1];
			const double xi_z = Nodes[DPN * n + 2] - Nodes[DPN * i + 2];


			const double xi_eta_x = Un[DPN * n + 0] - Un[DPN * i + 0] + xi_x;
			const double xi_eta_y = Un[DPN * n + 1] - Un[DPN * i + 1] + xi_y;
			const double xi_eta_z = Un[DPN * n + 2] - Un[DPN * i + 2] + xi_z;

			const double xi = sqrt(xi_x * xi_x + xi_y * xi_y + xi_z * xi_z);
			const double y = sqrt(xi_eta_x * xi_eta_x + xi_eta_y * xi_eta_y + xi_eta_z * xi_eta_z);
			const double y_xi = (y - xi);

			const double cx = xi_eta_x / y;
			const double cy = xi_eta_y / y;
			const double cz = xi_eta_z / y;

			const double _E = Stiffnesses[MAX_HORIZON_LENGTH * i + j];
			const double _A = Vols[i];
			const double _L = xi;

			const double _EAL = _E * _A / _L;

			Forces[MAX_HORIZON_LENGTH * i + j + 0] = _EAL * cx * y_xi;
			Forces[MAX_HORIZON_LENGTH * i + j + 1] = _EAL * cy * y_xi;
			Forces[MAX_HORIZON_LENGTH * i + j + 2] = _EAL * cz * y_xi;


			// Bond stretch calculations

			const double PD_S0 = FailStretches[i * MAX_HORIZON_LENGTH + j];

			const double s = y_xi / xi; //

			// Check for state of the bond

			if (s > PD_S0)
			{
				Horizons[i * MAX_HORIZON_LENGTH + j] = -1;  // Break the bond
			}
		}
	}
	
	barrier(CLK_LOCAL_MEM_FENCE);
}

// Reduction

__kernel void
	reduce(
		__global double* Forces,
		__global int *Horizons,
		__global double const *Udn,
		__global double *Uddn,
		__global int const *FCTypes,
		__global double const *FCValues
	)
{
	const int i = get_global_id(0);
	
	if(i < PD_NODE_NO) 
	{
		double sum0 = 0.00;
		double sum1 = 0.00;
		double sum2 = 0.00;

		for (int j =1; j < MAX_HORIZON_LENGTH; j++) {

			const int n = Horizons[MAX_HORIZON_LENGTH * i + j];

			if (n != -1)
			{
				sum0 += Forces[MAX_HORIZON_LENGTH * i + j + 0];
				sum1 += Forces[MAX_HORIZON_LENGTH * i + j + 1];
				sum2 += Forces[MAX_HORIZON_LENGTH * i + j + 2];
			}
		}

		const double f0 = (FCTypes[DPN*i + 0] == 2 ? sum0 : sum0); //+ FCValues[DPN * i + 0]);
		const double f1 = (FCTypes[DPN*i + 1] == 2 ? sum1 : sum1); //+ FCValues[DPN * i + 1]);
		const double f2 = (FCTypes[DPN*i + 2] == 2 ? sum2 : sum2 + FCValues[DPN * i + 2]);

		// Update accelerations
		Uddn[DPN * i + 0] = (f0 + PD_ETA * Udn[DPN * i + 0]) / PD_RHO;
		Uddn[DPN * i + 1] = (f1 + PD_ETA * Udn[DPN * i + 1]) / PD_RHO;
		Uddn[DPN * i + 2] = (f2 + PD_ETA * Udn[DPN * i + 2]) / PD_RHO;
	}
}

// Update velocities
__kernel void
	TimeMarching4(
        __global double *Udn,
        __global double const *Uddn
	)
{
	const int i = get_global_id(0);

	if (i < PD_DPN_NODE_NO)
	{   
        Udn[i] += Uddn[i] * PD_DT;
	}
}


__kernel void
	CheckBonds(
		__global int *Horizons,
		__global double const *Un,
		__global double const *Nodes,
		__global double const *FailStretches
	)
{
	const int i = get_global_id(0);
	const int j = get_global_id(1);

	if ((i < PD_NODE_NO) && (j > 0) && (j < MAX_HORIZON_LENGTH))
	{
		const int n = Horizons[i * MAX_HORIZON_LENGTH + j];

		if (n != -1)
		{
			const double xi_x = Nodes[DPN * n + 0] - Nodes[DPN * i + 0];  // Optimize later
			const double xi_y = Nodes[DPN * n + 1] - Nodes[DPN * i + 1];
			const double xi_z = Nodes[DPN * n + 2] - Nodes[DPN * i + 2];

			const double xi_eta_x = Un[DPN * n + 0] - Un[DPN * i + 0] + xi_x;
			const double xi_eta_y = Un[DPN * n + 1] - Un[DPN * i + 1] + xi_y;
			const double xi_eta_z = Un[DPN * n + 2] - Un[DPN * i + 2] + xi_z;

			const double xi = sqrt(xi_x * xi_x + xi_y * xi_y + xi_z * xi_z);
			const double y = sqrt(xi_eta_x * xi_eta_x + xi_eta_y * xi_eta_y + xi_eta_z * xi_eta_z);

			const double PD_S0 = FailStretches[i * MAX_HORIZON_LENGTH + j];

			const double s = (y - xi) / xi;

			// Check for state of the bond

			if (s > PD_S0)
			{
				Horizons[i * MAX_HORIZON_LENGTH + j] = -1;  // Break the bond
			}
		}
	}
}

__kernel void
	CalculateDamage(
		__global float *Phi,
		__global int const *Horizons,
		__global int const *HorizonLengths
	)
{
	const int i = get_global_id(0);

	if (i < PD_NODE_NO)
	{
		int active_bonds = 0;

		for (int j = 1; j < MAX_HORIZON_LENGTH; j++)
		{
			if (Horizons[MAX_HORIZON_LENGTH * i + j] != -1)
			{
				active_bonds++;
			}
		}

		Phi[i] = 1.00f - (float) active_bonds / (float) (HorizonLengths[i] - 1);
	}
}