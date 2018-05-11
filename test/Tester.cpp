#define __HIP_PLATFORM_HCC__ 1

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <hip/hip_runtime.h>
//#include <hip/device_functions.h>
//#include "hcc_code/cuda_extra.h"
#include "hip_code/cryptonight.h"
//#include "hcc_code/cuda_extra.cu"
#include "crypto/cryptonight_aesni.h"
#include "crypto/cryptonight_common.cpp"
#include "crypto/soft_aes.c"


/*
#include "c_groestl.h"
#include "c_blake256.h"
#include "c_jh.h"
#include "c_skein.h"


void do_blake_hash(const void* input, size_t len, char* output) {
	blake256_hash((uint8_t*)output, (const uint8_t*)input, len);
}

void do_groestl_hash(const void* input, size_t len, char* output) {
	groestl((const uint8_t*)input, len * 8, (uint8_t*)output);
}

void do_jh_hash(const void* input, size_t len, char* output) {
	jh_hash(32 * 8, (const uint8_t*)input, 8 * len, (uint8_t*)output);
}

void do_skein_hash(const void* input, size_t len, char* output) {
	skein_hash(8 * 32, (const uint8_t*)input, 8 * len, (uint8_t*)output);
}

void (* const extra_hashes[4])(const void *, size_t, char *) = {do_blake_hash, do_groestl_hash, do_jh_hash, do_skein_hash};
*/
const uint64_t keccakf_rndc_2[24] = 
{
	0x0000000000000001, 0x0000000000008082, 0x800000000000808a,
	0x8000000080008000, 0x000000000000808b, 0x0000000080000001,
	0x8000000080008081, 0x8000000000008009, 0x000000000000008a,
	0x0000000000000088, 0x0000000080008009, 0x000000008000000a,
	0x000000008000808b, 0x800000000000008b, 0x8000000000008089,
	0x8000000000008003, 0x8000000000008002, 0x8000000000000080, 
	0x000000000000800a, 0x800000008000000a, 0x8000000080008081,
	0x8000000000008080, 0x0000000080000001, 0x8000000080008008
};

extern void(*const extra_hashes[4])(const void *, size_t, char *);
extern int cuda_get_devicecount( int* deviceCount);
extern int cuda_get_deviceinfo(nvid_ctx *ctx);
extern int cryptonight_extra_cpu_init(nvid_ctx *ctx);
extern void cryptonight_extra_cpu_set_data( nvid_ctx* ctx, const void *data, uint32_t len);
extern void cryptonight_extra_cpu_prepare(nvid_ctx* ctx, uint32_t startNonce);
extern void cryptonight_core_cpu_hash(nvid_ctx* ctx, uint32_t startNonce);
extern void cryptonight_extra_cpu_final(nvid_ctx* ctx, uint32_t startNonce, uint64_t target, uint32_t* rescount, uint32_t *resnonce);

const int32_t start_nonce = 1060;
const int32_t keccak_input = 76;
const int32_t threads = 32;

void write(std::string filename, char * buf, int len) {
	std::ofstream outfile;
	outfile.open(filename, std::ios::out | std::ios::binary);
	outfile.write(buf, len);
}

int test_full_run() {
	printf("test\n");
	uint64_t state[25 * threads];
	memset(state, 0x00, 25 * threads * sizeof(int64_t));

	uint8_t inbuf[88];
	memset(inbuf, 0x05, 88);

	nvid_ctx ctx;
	ctx.device_id = 0;
	ctx.device_blocks = 4;
	ctx.device_threads = threads / 4;
	ctx.device_bfactor = 0;
	ctx.device_bsleep = 0;

	uint32_t rescount = 0;
	uint32_t resnonce[10];


	cryptonight_extra_cpu_init(&ctx);
	cryptonight_extra_cpu_set_data(&ctx, inbuf, keccak_input);
	cryptonight_extra_cpu_prepare(&ctx, start_nonce);
	cryptonight_core_cpu_hash(&ctx, start_nonce);
	cryptonight_extra_cpu_final(&ctx, start_nonce, 0xFFFFFFFl, &rescount, resnonce);

	hipMemcpy( state, ctx.d_ctx_state, 200 * threads, hipMemcpyDeviceToHost );

	for (int i = 0; i < threads; i++) {
		//*(uint32_t*)(inbuf + 39) = start_nonce + i;
		uint32_t nonce = start_nonce + i;
		for ( int i = 0; i < sizeof (uint32_t ); ++i )
			( ( (char *) inbuf ) + 39 )[i] = ( (char*) ( &nonce ) )[i];
		cryptonight_ctx cpu_ctx;

		char output[32];
		
		memset(cpu_ctx.hash_state, 0x00, 50 * sizeof(int32_t));
	
			
		/*keccak((const uint8_t *)inbuf, keccak_input, (uint8_t*) cpu_ctx.hash_state, 200);
		keccakf((uint64_t*)cpu_ctx.hash_state, 24);

		int algo = cpu_ctx.hash_state[0] & 3;
	        extra_hashes[algo](cpu_ctx.hash_state, 200, (char*)output);*/

		//cryptonight_hash_ctx(inbuf, keccak_input, &output, &cpu_ctx);
		cryptonight_hash<xmrstak_algo::cryptonight_monero, true, false>(inbuf, keccak_input, &output, &cpu_ctx);

		int same = memcmp(cpu_ctx.hash_state, state + 25 * i, 200);
		if (same == 0) {
			printf("MATCH for thread %d\n", i);
		} else {
			printf("BUG for thread %d\n", i);
		}
	
		write("gpu.bin." + std::to_string(i), (char*) (state + 25 * i), 200);
		write("cpu.bin." + std::to_string(i), (char*) cpu_ctx.hash_state, 200);
	}
	return 0;
}

int test_final_algos() {
	uint32_t state[50 * threads];
	memset(state, 0x00, 50 * threads * sizeof(int32_t));

	uint8_t inbuf[88];
	memset(inbuf, 0x05, 88);

	nvid_ctx ctx;
	ctx.device_id = 0;
	ctx.device_blocks = 16;
	ctx.device_threads = 64;
	ctx.device_bfactor = 0;
	ctx.device_bsleep = 0;

	uint32_t rescount = 0;
	uint32_t resnonce[10];


	cryptonight_extra_cpu_init(&ctx);
	cryptonight_extra_cpu_set_data(&ctx, inbuf, keccak_input);
	cryptonight_extra_cpu_prepare(&ctx, start_nonce);
	cryptonight_core_cpu_hash(&ctx, start_nonce);
	cryptonight_extra_cpu_final(&ctx, start_nonce, 0xFFFFFFFl, &rescount, resnonce);

	hipMemcpy( state, ctx.d_ctx_state, 200 * threads, hipMemcpyDeviceToHost );

	for (int i = 0; i < threads; i++) {
		*(uint32_t*)(inbuf + 39) = start_nonce + i;

		char output[32];
		uint32_t state2[50];
		memset(state2, 0x00, 50 * sizeof(int32_t));
	
		keccak((const uint8_t *)inbuf, keccak_input, (uint8_t*) state2, 200);
		keccakf((uint64_t*)state2, 24);

		int algo = state2[0] & 3;
	        extra_hashes[algo](state2, 200, (char*)output);

		memcpy(state2, output, 32);

		int same = memcmp(state2, state + 50 * i, 200);
		if (same == 0) {
			printf("MATCH algo %d for thread %d\n", algo, i);
		} else {
			printf("BUG algo %d for thread %d\n", algo, i);
		}
	
		write("gpu.bin." + std::to_string(i), (char*) (state + 50 * i), 200);
		write("cpu.bin." + std::to_string(i), (char*) state2, 200);
	}
	return 0;
}


int main() {
	test_full_run();
}
