
#pragma once

#include <hip/hip_runtime.h>

static inline void exit_if_cudaerror(int thr_id, const char *file, int line)
{
	hipError_t err = hipGetLastError();
	if(err != hipSuccess)
	{
		printf("\nGPU %d: %s\n%s line %d\n", thr_id, hipGetErrorString(err), file, line);
		exit(1);
	}
}
