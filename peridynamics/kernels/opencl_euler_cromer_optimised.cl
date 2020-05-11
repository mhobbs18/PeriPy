////////////////////////////////////////////////////////////////////////////////
//
// opencl_euler_cromer_optimised.cl
//
// OpenCL Peridynamics kernels
//
// Based on code from Copyright (c) Farshid Mossaiby, 2016, 2017. Adapted for python.
//
////////////////////////////////////////////////////////////////////////////////

// Includes, project
#include "opencl_enable_fp64.cl"

// Macros
#define DPN 3
// MAX_HORIZON_LENGTH, PD_DT, PD_E, PD_S0, PD_NODE_NO, PD_DPN_NODE_NO will be defined on JIT compiler's command line

// Volume correction factor
double beta(const double xi)
{
	if (xi <= PD_R - PD_DX / 2.00)
	{
		return 1.00;
	}

	else if (xi <= PD_R + PD_DX / 2.00)
	{
		return (PD_R + PD_DX / 2.00 - xi) / (PD_DX);
	}

	else
	{
		return 0.00;
	}
}

// Update displacements
__kernel void
	UpdateDisplacement(
        __global double const *Udn,
        __global double *Un,
		__global int const *BCTypes,
		__global double const *BCValues,
		double DISPLACEMENT_LOAD_SCALE
	)
{
	const int i = get_global_id(0);

	if (i < PD_DPN_NODE_NO)
	{
		Un[i] = (BCTypes[i] == 2 ? (Un[i] + PD_DT * Udn[i]) : (Un[i] + DISPLACEMENT_LOAD_SCALE * BCValues[i]));
	}
}

// Calculate force using un
__kernel void
	CalcBondForce(
    __global double *Forces,
    __global double const *Un,
    __global double const *Vols,
	__global int *Horizons,
	__global double const *Nodes,
	__global double const *Stiffnesses,
	__global double const *FailStretches
	)
{
	const int i = get_global_id(0);
	const int j = get_global_id(1);

	if ((i < PD_NODE_NO) && (j >= 0) && (j < MAX_HORIZON_LENGTH))
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
			const double _A = beta(xi) * Vols[n];
			const double _L = xi;

			const double _EAL = _E * _A / _L;

			Forces[MAX_HORIZON_LENGTH * (DPN * i + 0) + j] = _EAL * cx * y_xi;
			Forces[MAX_HORIZON_LENGTH * (DPN * i + 1) + j] = _EAL * cy * y_xi;
			Forces[MAX_HORIZON_LENGTH * (DPN * i + 2) + j] = _EAL * cz * y_xi;

			const double PD_S0 = FailStretches[i * MAX_HORIZON_LENGTH + j];

			const double s = (y - xi) / xi;

			// Check for state of the bond

			if (s > PD_S0)
			{
			Horizons[i * MAX_HORIZON_LENGTH + j] = -1;  // Break the bond
			}
		}
		else 
		{
			Forces[MAX_HORIZON_LENGTH * (DPN * i + 0) + j] = 0.00;
			Forces[MAX_HORIZON_LENGTH * (DPN * i + 1) + j] = 0.00;
			Forces[MAX_HORIZON_LENGTH * (DPN * i + 2) + j] = 0.00;
		}
	}
}

__kernel void 
    ReduceForce(
        __global double const * Forces,
        __global double * Uddn,
		__global double const * Udn,
        __global int const * FCTypes,
        __global double const * FCValues,
        __local double * local_cache,
		double FORCE_LOAD_SCALE
   )
{
    
  	int global_id = get_global_id(0); 
  
 	int local_id = get_local_id(0); 
  
  	// local size is the MAX_HORIZONS_LENGTHS and must be a power of 2
  	int local_size = get_local_size(0); 
  
  	//Copy values into local memory 
  	local_cache[local_id] = Forces[global_id]; 

  	//Wait for all threads to catch up 
  	barrier(CLK_LOCAL_MEM_FENCE); 

	// Parallel reduction
	for (int i = local_size/2; i > 0; i /= 2){
		if(local_id < i){
		local_cache[local_id] += local_cache[local_id + i];
		} 
		//Wait for all threads to catch up 
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (!local_id) {
		//Get the reduced forces
		int index = global_id/local_size;
		// Update accelerations
		Uddn[index] = (FCTypes[index] == 2 ? ((local_cache[0] - PD_ETA * Udn[index]) / PD_RHO) : ((local_cache[0] + FORCE_LOAD_SCALE * FCValues[index] - PD_ETA * Udn[index]) / PD_RHO));
	}
}

// Update velocities
__kernel void
	UpdateVelocity(
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
    ReduceDamage(
        __global int const *Horizons,
		__global int const *HorizonLengths,
        __global double *Phi,
        __local double* local_cache
   )
{
    
  int global_id = get_global_id(0); 
  
  int local_id = get_local_id(0); 
  
  // local size is the MAX_HORIZONS_LENGTHS and must be a power of 2
  int local_size = get_local_size(0); 
  
  //Copy values into local memory 
  local_cache[local_id] = Horizons[global_id] != -1 ? 1.00 : 0.00; 

  //Wait for all threads to catch up 
  barrier(CLK_LOCAL_MEM_FENCE); 

  for (int i = local_size/2; i > 0; i /= 2){
    if(local_id < i){
      local_cache[local_id] += local_cache[local_id + i];
    } 
    //Wait for all threads to catch up 
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (!local_id) {
    //Get the reduced forces
    int index = global_id/local_size;
    // Update damage
    Phi[index] = 1.00 - (double) local_cache[0] / (double) (HorizonLengths[index]);
}
}