#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <hip/hip_runtime.h>
#ifdef __HCC__
#include <hip/hcc_detail/device_functions.h>
#else
#include <vector_functions.h>
#endif

#ifdef _WIN32
#include <windows.h>
extern "C" void compat_usleep(uint64_t waitTime)
{
    if (waitTime > 0)
    {
        if (waitTime > 100)
        {
            // use a waitable timer for larger intervals > 0.1ms

            HANDLE timer;
            LARGE_INTEGER ft;

            ft.QuadPart = -(10*waitTime); // Convert to 100 nanosecond interval, negative value indicates relative time

            timer = CreateWaitableTimer(NULL, TRUE, NULL);
            SetWaitableTimer(timer, &ft, 0, NULL, NULL, 0);
            WaitForSingleObject(timer, INFINITE);
            CloseHandle(timer);
        }
        else
        {
            // use a polling loop for short intervals <= 100ms

            LARGE_INTEGER perfCnt, start, now;
            __int64 elapsed;

            QueryPerformanceFrequency(&perfCnt);
            QueryPerformanceCounter(&start);
            do {
		SwitchToThread();
                QueryPerformanceCounter((LARGE_INTEGER*) &now);
                elapsed = (__int64)((now.QuadPart - start.QuadPart) / (float)perfCnt.QuadPart * 1000 * 1000);
            } while ( elapsed < waitTime );
        }
    }
}
#else
#include <unistd.h>
extern "C" void compat_usleep(uint64_t waitTime)
{
	usleep(waitTime);
}
#endif

#include "cryptonight.h"
#include "cuda_extra.h"
#include "cuda_aes.hpp"
#include "cuda_device.hpp"

/* sm_2X is limited to 2GB due to the small TLB
 * therefore we never use 64bit indices
 */
#if 1 // defined(XMR_STAK_LARGEGRID) // && (__CUDA_ARCH__ >= 300)
typedef uint64_t IndexType;
#else
typedef int IndexType;
#endif

__device__ __forceinline__ uint64_t cuda_mul128( uint64_t multiplier, uint64_t multiplicand, uint64_t* product_hi )
{
	*product_hi = __umul64hi( multiplier, multiplicand );
	return (multiplier * multiplicand );
}

// ASM required to wedge operations on variables loaded from memory into the correct spot.
#ifdef __HCC__
#define RETIRE(x)  { \
        uint32_t * const w = reinterpret_cast<uint32_t*const>(&x); \
        asm volatile("v_mov_b32 %0, %1" : "=v" (w[0]), "=v" (w[1]) : "v" (w[0]), "v" (w[1]) : "memory"); }

#define FENCE(x) { \
	uint32_t * const w = reinterpret_cast<uint32_t*const>(&x); \
	asm volatile("v_mov_b32 %0, %2\n\tv_mov_b32 %1, %3" : "=v" (w[0]), "=v" (w[1]) : "v" (w[0]), "v" (w[1]) : "memory"); }

#define WAIT_FOR(x, n) asm volatile("s_waitcnt vmcnt(" #n ")\n\t": : "v" (x) : "memory"); FENCE(x)
#define PRIO(n) asm volatile ("s_setprio 0x" #n);
#else
#define WAIT_FOR(x, n) FENCE(x)
#define RETIRE(x) x;
#define FENCE(w) { \
	asm volatile("mov.u64 %0, %1;\n\t" : "=l" (w) : "l" (w) : "memory"); }
//	uint32_t * const w = reinterpret_cast<uint32_t*const>(&x); \
//	asm volatile("mov.u32 %0, %2;\n\tmov.u32 %1, %3;" : "=r" (w[0]), "=r" (w[1]) : "r" (w[0]), "r" (w[1]) : "memory"); }
#endif


#if !__HIP_ARCH_GFX803__
#define EMIT_LOAD(args) "global_load_dwordx2 " args ", off"
#define EMIT_STORE(args) "global_store_dwordx4 " args ", off"
#else
#define EMIT_LOAD(args) "flat_load_dwordx2 " args
#define EMIT_STORE(args) "flat_store_dwordx4 " args
#endif

template< typename T >
__device__ __forceinline__ void storeGlobal128AsyncGlc( T* adr, T const & val ) {
#ifdef __HCC__
	uint32_t * const val32 = (uint32_t*) &val;
	asm volatile (EMIT_STORE("%4, v[19:22]") " glc"
				  :
				  : "{v19}" (val32[0]), "{v20}" (val32[1]), "{v21}" (val32[2]), "{v22}" (val32[3]), "r" ( adr ));
#else
	*adr = val;
#endif
}

template<int N>
__device__ __forceinline__ void memc_(void * __restrict__ dst, void * __restrict__ src) {
	ulonglong2 * lDst = reinterpret_cast<ulonglong2*>(dst);
	ulonglong2 * lSrc = reinterpret_cast<ulonglong2*>(src);
	#pragma unroll
	for (int i = 0; i < N; i++) {
		lDst[i] = lSrc[i];
	}
}

// Number of threads per block to use in phase 1 and 3
#define P13T 64
#define ENABLE_LAUNCH_BOUNDS 1

#if ENABLE_LAUNCH_BOUNDS
__launch_bounds__( P13T )
#endif
__global__ void cryptonight_core_gpu_phase1( int threads, int bfactor, int partidx, uint64_t * __restrict__ long_state_64, uint32_t * __restrict__ ctx_state, uint32_t * __restrict__ ctx_key1 )
{
	__shared__ uint32_t sharedMemory[1024];

	cn_aes_gpu_init( sharedMemory );
	__syncthreads( );
	uint4 * const long_state = reinterpret_cast<uint4*>(long_state_64);

	const uint64_t thread = ( hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x ) >> 3;
	const int sub = ( hipThreadIdx_x & 7 ) << 2;

	const int batchsize = 0x80000 >> bfactor;
	const int start = partidx * batchsize;
	const int end = start + batchsize;

	if ( thread >= threads )
		return;

	uint32_t key[40];
	uint4 text;

	//memcpy( key, ctx_key1 + thread * 40, 160 );
	memc_<10>( key, ctx_key1 + thread * 40);

	if( partidx == 0 )
	{
		// first round
		//text = ctx_state[thread * 50 + sub + 16];
		text = *reinterpret_cast<uint4*>(ctx_state + thread * 50 + sub + 16);
	}
	else
	{
		// load previous text data
		MEMCPY8( &text, &long_state[( (uint64_t) thread << 19 ) + sub + start - 32], 2 );
	}
	__syncthreads( );
	for ( int i = start; i < end; i += 32 )
	{
		cn_aes_pseudo_round_mut( sharedMemory, (uint32_t*) &text, key );

		int offset = ((thread << 19) + (sub + i) ) / 4;
		storeGlobal128AsyncGlc(long_state + offset, text);
	}
}

#ifdef __HCC__
// HIP builtin is broken so we need to resort to manual impl.
#define UMUL64HI(m1, m2) _gpu_mul_hi_u64(m1, m2)
#else
#define UMUL64HI(m1, m2) __umul64hi(m1, m2)
#endif

__device__ __forceinline__ ulong
_gpu_mul_hi_u64(ulong x, ulong y)
{
    ulong x0 = x & 0xffffffffUL;
    ulong x1 = x >> 32;
    ulong y0 = y & 0xffffffffUL;
    ulong y1 = y >> 32;
    ulong z0 = x0*y0;
    ulong t = x1*y0 + (z0 >> 32);
    ulong z1 = t & 0xffffffffUL;
    ulong z2 = t >> 32;
    z1 = x0*y1 + z1;
    return x1*y1 + z2 + (z1 >> 32);
}

#if __HIP_ARCH_GFX900__
#define MFLAGS " slc"
#else
#define MFLAGS ""
#endif

#ifdef __HCC__
#define ASYNC_LOAD(dst0, dst1, adr)	{								\
		asm volatile(												\
			EMIT_LOAD("%0, %2") MFLAGS " \n\t"								\
			EMIT_LOAD("%1, %3") MFLAGS " \n\t"						\
			: "=v"(dst0), "=v" (dst1)								\
			: "r" (adr), "r"(adr+1) ); }

#define AL128(dst, adr) asm volatile("flat_load_dwordx4 %0, %1" : "=v"(dst) : "r"(adr));
#else
#define ASYNC_LOAD(dst0, dst1, adr)	{								\
		asm volatile( "prefetch.global.L1 [%0];" : : "l"(adr) );	\
		asm volatile( "prefetch.global.L1 [%0+8];" : : "l"(adr) );}
#endif

// Only available for HCC.
#define ASYNC_STORE(adr, src) {								\
		uint32_t * const s32 = reinterpret_cast<uint32_t*>(&src);		\
		asm volatile(EMIT_STORE("%0, v[20:23]")	MFLAGS				\
					 :						\
					 : "r" (adr),					\
					   "{v20}" (s32[0]), "{v21}" (s32[1]), "{v22}" (s32[2]), "{v23}" (s32[3])); }


#define EXTRACT_32(dst, src)					\
	dst.x = (uint32_t) src.x;					\
	dst.y = (uint32_t) (src.x >> 32);			\
	dst.z = (uint32_t) src.y;					\
	dst.w = (uint32_t) (src.y >> 32);

#ifdef __HCC__
__launch_bounds__( 8 )
#else
//__launch_bounds__( 64 )
#endif
__global__ void cryptonight_core_gpu_phase2( int threads, int bfactor, int partidx, uint64_t * __restrict__ d_long_state_64, uint32_t * __restrict__ d_ctx_a, uint32_t * __restrict__ d_ctx_b, uint32_t * __restrict__ d_ctx_state, uint32_t startNonce, uint32_t * __restrict__ d_input )
{
	__shared__ uint32_t sharedMemWritable[1024];

	cn_aes_gpu_init( sharedMemWritable );

	__syncthreads( );

	const uint32_t * const __restrict__ sharedMemory = (const uint32_t*) sharedMemWritable;


	const int thread = ( hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x );

	int i;
	uint32_t j0, j1;
	bool same_adr;
	const int batchsize = ITER >> ( 2 );
	const int start = 0;
	const int end = start + batchsize;
	ulonglong2 * __restrict__ long_state = reinterpret_cast<ulonglong2*>(d_long_state_64 + ((((IndexType) thread) << 19) / 2));
	ulonglong2 * ctx_a = reinterpret_cast<ulonglong2*>(d_ctx_a + thread * (16/sizeof(uint32_t)));
	uint32_t * ctx_b = d_ctx_b + thread * (16/sizeof(uint32_t));

	uint32_t * state = d_ctx_state + thread * 50;

	uint32_t tweak1_2[2];
	tweak1_2[0] = (d_input[8] >> 24) | (d_input[9] << 8);
	tweak1_2[0] ^= state[48];
	tweak1_2[1] = startNonce + thread;
	tweak1_2[1] ^= state[49];

	ulonglong2 d[2];

	// Do not do memcpy here: it somehow causes the main loop to buffer registers t_t
	ulonglong2 a = *reinterpret_cast<ulonglong2*>(ctx_a);
	d[1] = *reinterpret_cast<ulonglong2*>(ctx_b);

	j0 = ( ( a.x & 0x1FFFF0 ) >> 4 );

	__syncthreads();
	#pragma unroll 2
	for ( i = start; i < end; ++i )
	{
		#pragma unroll 2
		for ( int x = 0; x < 2; ++x )
		{

			ulonglong2 x64 = long_state[j0];

			uint4 c;

			uint32_t * x32 = reinterpret_cast<uint32_t*>(&x64);
			uint32_t * a32 = reinterpret_cast<uint32_t*>(&a);

			c.x = a32[0] ^ (t_fn0(x32[0] & 0xff) ^ t_fn1((x32[1] >> 8) & 0xff) ^ t_fn2((x32[2] >> 16) & 0xff) ^ t_fn3((x32[3] >> 24)));
			j1 = ( ( c.x & 0x1FFFF0 ) >> 4 );

			c.y = a32[1]  ^ (t_fn0(x32[1] & 0xff) ^ t_fn1((x32[2] >> 8) & 0xff) ^ t_fn2((x32[3] >> 16) & 0xff) ^ t_fn3((x32[0] >> 24)));
			c.z = a32[2]  ^ (t_fn0(x32[2] & 0xff) ^ t_fn1((x32[3] >> 8) & 0xff) ^ t_fn2((x32[0] >> 16) & 0xff) ^ t_fn3((x32[1] >> 24)));
			c.w = a32[3]  ^ (t_fn0(x32[3] & 0xff) ^ t_fn1((x32[0] >> 8) & 0xff) ^ t_fn2((x32[1] >> 16) & 0xff) ^ t_fn3((x32[2] >> 24)));

			d[x].x = c.x | ((uint64_t) c.y << 32);
			d[x].y = c.z | ((uint64_t) c.w << 32);

			ulonglong2 d_xored = d[0]; // ^d[1];
			d_xored ^= d[1];

			uint32_t fork_7 = d_xored.y;
			uint8_t xo = fork_7 >> 24;

			const uint16_t table = 0x7531;

			uint8_t index = (((xo >> 3) & 6) | (xo & 1)) << 1;
			fork_7 ^= ((table >> index) & 0x3) << 28;

			d_xored.y = (d_xored.y & 0xFFFFFFFF00000000) | fork_7;

			long_state[j0] = d_xored;

/*
 * 2
 */


			ulonglong2 y2 = long_state[j1];
			uint64_t t1_64 = d[x].x;

			a.x += UMUL64HI(t1_64, y2.x);

			ulonglong2 a_stor;
			a_stor.x = a.x;

///////
			a.x ^= y2.x;
			j0 = ( ( a.x & 0x1FFFF0 ) >> 4 );
//////
			a.y += (t1_64 * y2.x);

			a_stor.y = a.y;

			uint32_t * a_stor32 = reinterpret_cast<uint32_t*>(&a_stor);
			a_stor32[2] ^= tweak1_2[0];
			a_stor32[3] ^= tweak1_2[1];

			long_state[j1] = a_stor;
			a.y ^= y2.y;
		}
		// Voodoo
//		if (i % 32 == 0) asm volatile("s_barrier" ::);
	}
/*
	if ( bfactor > 0 )
	{
		*ctx_a = a;
		*ctx_b = d[1];
	}*/
}

#if ENABLE_LAUNCH_BOUNDS
__launch_bounds__( P13T )
#endif
__global__ void cryptonight_core_gpu_phase3( int threads, int bfactor, int partidx, const uint64_t * __restrict__ long_state_64, uint32_t * __restrict__ d_ctx_state, uint32_t * __restrict__ d_ctx_key2 )
{
	__shared__ uint32_t sharedMemory[1024];

	cn_aes_gpu_init( sharedMemory );
	__syncthreads( );

	int thread = ( hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x ) >> 3;
	int subRaw = ( hipThreadIdx_x & 7 );
	int sub = ( hipThreadIdx_x & 7 ) << 2;
	const uint4 * __restrict__ long_state = reinterpret_cast<const uint4*>(long_state_64 + ((((IndexType) thread) << 19) / 2));

	const int batchsize = (0x80000 >> 2) >> bfactor;
	const int start = partidx * batchsize;
	const int end = start + batchsize;

	if ( thread >= threads )
		return;

	uint32_t key[40];
	uint4 text;
	//memcpy( key, d_ctx_key2 + thread * 40, 160 );
	memc_<10>( key, d_ctx_key2 + thread * 40 );
	text = *reinterpret_cast<uint4*>(d_ctx_state + thread * 50 + sub + 16);

	__syncthreads( );
	#pragma unroll
	for ( int i = start; i < end; i += 8 )
	{
		uint4 l = long_state[subRaw+i];

		text.x ^= l.x;
		text.y ^= l.y;
		text.z ^= l.z;
		text.w ^= l.w;

		cn_aes_pseudo_round_mut( sharedMemory, (uint32_t*) &text, key );
	}

	//memcpy(d_ctx_state + thread * 50 + sub + 16, &text, sizeof(uint4));
	memc_<1>(d_ctx_state + thread * 50 + sub + 16, &text);
	__syncthreads( );
}

extern "C" void cryptonight_core_cpu_hash(nvid_ctx* ctx, uint32_t nonce)
{
	dim3 grid( ctx->device_blocks );
	dim3 grid_halved( ctx->device_blocks / 2 );
	dim3 block( ctx->device_threads );
	dim3 block2( ctx->device_threads * 2);
	dim3 block8( ctx->device_threads << 3 );
	dim3 block16( ctx->device_threads << 4 );

	dim3 p1_3_grid((ctx->device_blocks * ctx->device_threads * 8) / P13T);
	dim3 p1_3_block( P13T );
#if 0
	// Always use min. 64 threads in p1 and p3
	// So when threads < 8, take threads * 16 and halve the blocks.
	if (ctx->device_threads < 8) {
		p1_3_block = block16;
		p1_3_grid = grid_halved;
	} else {
		p1_3_block = block8;
		p1_3_grid = grid;
	}
#endif
	int partcount = 1 << ctx->device_bfactor;

	/* bfactor for phase 1 and 3
	 *
	 * phase 1 and 3 consume less time than phase 2, therefore we begin with the
	 * kernel splitting if the user defined a `bfactor >= 5`
	 */
	int bfactorOneThree = ctx->device_bfactor - 4;
	if( bfactorOneThree < 0 )
		bfactorOneThree = 0;

	int partcountOneThree = 1 << bfactorOneThree;

	for ( int i = 0; i < partcountOneThree; i++ )
	{
		hipLaunchKernelGGL(cryptonight_core_gpu_phase1, dim3(p1_3_grid), dim3(p1_3_block), 0, 0, ctx->device_blocks*ctx->device_threads,
			bfactorOneThree, i,
			ctx->d_long_state, ctx->d_ctx_state, ctx->d_ctx_key1);
		exit_if_cudaerror( ctx->device_id, __FILE__, __LINE__ );

		if ( partcount > 1 && ctx->device_bsleep > 0) compat_usleep( ctx->device_bsleep );
	}
	if ( partcount > 1 && ctx->device_bsleep > 0) compat_usleep( ctx->device_bsleep );

	for ( int i = 0; i < partcount; i++ )
	{
		printf("Starting run for nonce %d\n", nonce);
        hipLaunchKernelGGL(cryptonight_core_gpu_phase2, dim3(grid), dim3(block), 0, 0, ctx->device_blocks*ctx->device_threads,
            ctx->device_bfactor,
            i,
            ctx->d_long_state,
            ctx->d_ctx_a,
            ctx->d_ctx_b, ctx->d_ctx_state, nonce, ctx->d_input);
		exit_if_cudaerror( ctx->device_id, __FILE__, __LINE__ );

		if ( partcount > 1 && ctx->device_bsleep > 0) compat_usleep( ctx->device_bsleep );
	}

	for ( int i = 0; i < partcountOneThree; i++ )
	{
		hipLaunchKernelGGL(cryptonight_core_gpu_phase3, dim3(p1_3_grid), dim3(p1_3_block), 0, 0, ctx->device_blocks*ctx->device_threads,
			bfactorOneThree, i,
			ctx->d_long_state,
			ctx->d_ctx_state, ctx->d_ctx_key2);
		exit_if_cudaerror( ctx->device_id, __FILE__, __LINE__ );
	}
}
