## This repo contains

Proof of concept for a ~3% optimization in the core CryptoNight algorithm. This optimization could theoretically be implemented on every GPU compute platform/framework.

This miner, however, currently works only on Linux. It can run both Nvidia cards and AMD Vega cards. NV cards older than GeForce 10 (Pascal) might not experience a hashrate improvement. AMD cards older than Vega currently have a bug somewhere. **Speedup on Vega cards is only relative to current hash rates on Linux. It is still much less than ANY miner running on Windows.**

The code is based on xmr-stak-nvidia, i.e. the original CUDA part of xmr-stak. I have ported it to [HIP](https://github.com/ROCm-Developer-Tools/HIP), which is a framework developed by AMD that allows writing GPU compute code that can be built for both Nvidia GPUs (where it will be cross-compiled via CUDA and hence run with barely any performance impact) and AMD GPUs (where it uses the new "ROCm" driver stack).

If you aren't technically literate/interested, I would not recommend trying to use this miner. It is relatively complicated to set up, and for little gain. If the optimization turns out to be useful and stable, it can be incorporated into the normal xmr-stak anyway.

## Build environment

- You need [ROCm](https://github.com/RadeonOpenCompute/ROCm/#installing-from-amd-rocm-repositories) even if you only want to build for Nvidia cards. It is strongly suggested you run Ubuntu, as setting up ROCm anywhere else is nigh impossible for now.

- You additionally need [CUDA](https://developer.nvidia.com/cuda-downloads) if you want to build for Nvidia cards.
  
- GCC 5 is proven to work for building the non-GPU part of the project; when targeting AMD, you can probably also use clang.

## Building

Make sure you export the `HIP_PLATFORM` variable correctly in each case. I think if you paste all lines into the terminal at once, it might not work properly on some shells.

To build for target AMD:
```sh
mkdir build && cd build
export HIP_PLATFORM=hcc
cmake .. -DCUDA_COMPILER=/opt/rocm/bin/hipcc -DHIP_PLATFORM=hcc -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DHIP_ROOT_DIR=/opt/rocm/hip -DMICROHTTPD_REQUIRED=OFF`
make
```

When you build for AMD, an executable named `xmr-stak-test` will also be created in the binary output folder, allowing quick validation of the miner.

To build for target CUDA:
```sh
mkdir cuda_build && cd cuda_build
export HIP_PLATFORM=nvcc
cmake .. -DCUDA_COMPILER=/opt/rocm/bin/hipcc -DHIP_PLATFORM=nvcc -DCUDA_ARCH=30 -DCUDA_PATH=/usr -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DHIP_ROOT_DIR=/opt/rocm/hip -DMICROHTTPD_REQUIRED=OFF
make
```

## Running / configuring

Configuration is done CUDA-style with threads/blocks for both platforms.

I.e. `threads * blocks * 2` must always be less than available GPU memory in MB, and `blocks` should be a multiple of CU count for optimal performance.

As for threads:

- On AMD (Vega) cards, set threads to 8 for maximal performance. **E.g. use threads=8 blocks=448 for maximal performance on Vega 56.** 16 threads (with half the blocks) can be worth a try as well.

- **On Nvidia cards, please do not use your usual settings**, but rather set `threads` to at least 32 (possibly the sweet spot for all GeForce cards) and blocks accordingly to a lower value. E.g. best performance on a 1050 Ti is reached by `threads=32, blocks=48`.


## Explanation of the core-algo optimization

In the core loop of the CryptoNight algorithm, execution speed is basically directly dependant on the speed of scratchpad loads.
   
On AMD GCN ISA, memory operations are actually split in two: One operation to request the store/load, another operation – `s_wait vmcnt(n)` – to wait on outstanding memory operations.

We can use this to build "async" loads: First, schedule the load. Then do something else while the load runs. Finally, s_wait for the load to finish if it hasn’t yet.

Asynchronous loads introduce a new angle of optimization: Scheduling a load operation as early as possible maximizes the amount of work we can do _while the load is executing_, therefor reducing wasted time.

Until here, this could just be the compiler’s job, and I guess that AMD’s GPU compilers have some logic for this.

But the compiler cannot know we’re running an algorithm that is so extremely ill-fitted to GPU computing; it cannot know how important loads _really_ are to us.

It turns out that loads are that narrow bottlenecks, it pays off to schedule them ahead of the normally preceding scratchpad-store operations.

This means we have to check whether we actually loaded stale data (small chance, but would corrupt our whole result) that was supposed to have been overwritten by the preceding store, and if so, use the value we just stored instead.

Doing this check introduces plenty of additional operations, but since they can mostly be executed while the load is still running, it still pays off.

On CUDA, memory operations cannot be run asynchronous like that. Instead we use `prefetch` to achieve a similar effect. Theoretically we wouldn’t have to execute stores and loads in the wrong order, with this method, but somehow the proper order was still slower, in testing.

## Regarding donations

Please note that I haven’t changed the donation address after forking from xmr-stak-nvidia, hence you’ll still be donating to the original devs.

If you enjoyed this work and want to drop me some crypto coins, please use:

XMR: `45FbpewbfJf6wp7gkwAqtwNc7wqnpEeJdUH2QRgeLPhZ1Chhi2qs4sNQKJX4Ek2jm946zmyBYnH6SFVCdL5aMjqRHodYYsF`
BTC: `181TVrHPjeVZuKdqEsz8n9maqFLJAzTLc`

## Further performance and efficiency tuning

Generally, the miner has pretty stable performance on Vega cards. It does not fluctuate like on OpenCL. You therefor get a pretty direct feedback whenever you change something.

### Using more than one process on AMD cards

As you may know from the OpenCL miner, scheduling two workloads instead of a single one can bring extra performance. Unfortunately, the HIP/hcc backend seems to become confused when you simply configure the miner with e.g. two 8x224 lines. However, it mostly works when you start two seperate processes. E.g. you schedule a single 8x224 workload in the config, than simply start the miner twice, using this same config both times. It can take a few tries and some time to stabilize, but yields some extra percent, apparently.

### Kernel driver tweak

ROCm comes with a special kernel driver that is built via dkms. I’ve discovered a small performance tweak here. In the file `/usr/src/rock-1.7.148-ubuntu/amd/amdgpu/amdgpu_amdkfd_gpuvm.c` change the following code:


```c
if (coherent)
	mapping_flags |= AMDGPU_VM_MTYPE_UC;
else
	mapping_flags |= AMDGPU_VM_MTYPE_NC;
```

To this:

```c
if (coherent)
	mapping_flags |= AMDGPU_VM_MTYPE_UC;
else if (size > 1073741824)
	mapping_flags |= AMDGPU_VM_MTYPE_CC;
else
	mapping_flags |= AMDGPU_VM_MTYPE_DEFAULT;
```

Then apply the changes:

```sh
dkms uninstall "rock/1.7.148-ubuntu"
rm -rf /var/lib/dkms/rock/*
dkms install "rock/1.7.148-ubuntu" --force 
```

It will cause a different type of memory to be allocated for large chunks (like the scratchpad space we use for CryptoNight). Can increase hashrate by ~0.5%.
