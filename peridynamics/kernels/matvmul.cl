////////////////////////////////////////////////////////////////////////////////
//
// matvul.cl
//
// Matrix vector multiplication kernels
//
//
//
////////////////////////////////////////////////////////////////////////////////

// Includes, project

#include "opencl_enable_fp64.cl"

// Macros
#define DPN 3

kernel void mvmul1(int M,
                    int N,
                    const global double *A,
                    const global double *x,
                    global double *y)
    {   
        int i = get_global_id(0);
        double acc = 0.0;

        for (int j=0; j<N; j++)
        {
            acc += A[M * j + i] * x[j];
        }

        y[i] = acc;
    }

kernel void mvmul2(int M,
                    int N,
                    int P,
                    const global double *A,
                    const global double *x,
                    global double *y)
    {   
        int i = get_global_id(0);
        int j = get_global_id(1);

        local double sums[$(P)];
        float sum = 0.0;

        for (int q=0; q<(N / P); q++)
        {
            sum += A[M * (j + P * q) + i] * x[j + P * q];
        }

        sums[j] = sum;
        barrier(CLK_LOCAL_MEM_FENCE);

        if (j == 0)
        {
            double sumtotal = 0.0;
            for (int p=0; p<P; p++)
            {
                sumtotal += sums[p];
            }

            y[i] = sumtotal;
        }
    }