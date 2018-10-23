# Development continues elsewhere

https://github.com/949f45ac/xmrig-HIP

## This repo contains

Fully functional miner that runs with a proof of concept for a [~3% optimization](#explanation-of-the-core-algo-optimization) in the core CryptoNight algorithm. This optimization could theoretically be implemented on every GPU compute platform/framework.

This miner, however, currently works only on Linux. It can run:

- AMD Vega cards – **Still slower than Windows**, but faster and more stable than the OpenCL kernels on Linux.

- AMD RX 400/500 series cards. In some cases faster and more efficient, in others slower.

- Nvidia GeForce 10 series cards. Older series may work, but not experience any speedup.

The code is based on xmr-stak-nvidia, i.e. the original CUDA part of xmr-stak. I have ported it to [HIP](https://github.com/ROCm-Developer-Tools/HIP), which is a framework developed by AMD that allows writing GPU compute code that can be built for both Nvidia GPUs (where it will be cross-compiled via CUDA and hence run with barely any performance impact) and AMD GPUs (where it uses the new "ROCm" driver stack).

If you aren't technically literate/interested, I would not recommend trying to use this miner. It is relatively complicated to set up, and for little gain. If the optimization turns out to be useful and stable, it can be incorporated into the [normal xmr-stak](https://github.com/fireice-uk/xmr-stak) anyway.
(If you run only Nvidia cards, you may check out the pre-built binary I’ve put up in the "releases" section. The only GPU lib it requires to run is libcudart, so it’s pretty portable across Linux systems.)

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
cmake .. -DCUDA_COMPILER=/opt/rocm/bin/hipcc -DHIP_PLATFORM=hcc -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DHIP_ROOT_DIR=/opt/rocm/hip -DMICROHTTPD_REQUIRED=OFF
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
You may have to export `CUDA_PATH` into your shell environment so HIP can find it.

## Running / configuring

Configuration is done CUDA-style with threads/blocks for both platforms.

I.e. `threads * blocks * 2` must always be at least 200 less than available GPU memory in MB, and `blocks` should be a multiple of CU count for optimal performance.

As for threads:

- On Polaris cards, 8 threads usually perform best. E.g. on **RX 470/570 (32 CU), try 8x228** if you have 4GB RAM.

- On AMD (Vega) cards, you should probably use 16 threads. 8 threads can be a bit faster, but tend to cause more issues. So that’s either **16x224 or 8x448 on an 8GB Vega**.

- **On Nvidia cards, please do not use your usual settings**, but rather set `threads` to at least 32 (possibly the sweet spot for all GeForce cards) and blocks accordingly to a lower value. E.g. best performance on a 1050 Ti is reached by `threads=32, blocks=48`.


## Core-algo optimization brief summary

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

If you gave a star to this repository, I’d also be glad already. :)
