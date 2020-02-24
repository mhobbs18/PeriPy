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
		Un[i] = BCTypes[i] == 2 ? Un[i] + PD_DT * (Udn[i]) : Un[i] + BCValues[i];
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

			Forces[MAX_HORIZON_LENGTH * (i + 0) + j] = _EAL * cx * y_xi;
			Forces[MAX_HORIZON_LENGTH * (i + 1) + j] = _EAL * cy * y_xi;
			Forces[MAX_HORIZON_LENGTH * (i + 2) + j] = _EAL * cz * y_xi;


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

//------------------------------------------------------------------------------
//
// OpenCL function:  reduction    
//
// Purpose: reduce across all the work-items in a work-group
// 
// input: local float* an array to hold sums from each work item
//
// output: global float* partial_sums   float vector of partial sums
//

__kernel void reduce(                                          
   __global float* Forces,
   __global double *Udn,
   __global double *Uddn,
   __global int const *FCTypes,
   __global double const *FCValues
   
   )                        
{               
   //int local_id       = get_local_id(0);
   //int group_id       = get_group_id(0);                
   int global_id       = get_global_id(0);                   
   
   float sum;                            
   int i;
   barrier(CLK_LOCAL_MEM_FENCE);
   if (global_id < PD_DPN_NODE_NO) {
	   sum = 0.0f;
       for (i=0; i<MAX_HORIZON_LENGTH; i++) {
		   sum += Forces[global_id* MAX_HORIZON_LENGTH + i];         
      }
	  barrier(CLK_LOCAL_MEM_FENCE);
	  // Update accelerations
	  Uddn[global_id] = FCTypes[global_id] == 2 ? sum + PD_ETA * Udn[global_id] / PD_RHO : sum + PD_ETA * Udn[global_id] / PD_RHO;
	  // Update velocities
	  Udn[global_id] += Uddn[global_id] * PD_DT;
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