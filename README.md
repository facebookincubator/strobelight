# Strobelight

Strobelight is a fleetwide profiler framework developed by Meta, designed to provide comprehensive profiling capabilities across large-scale infrastructure. This framework helps in identifying performance bottlenecks and optimizing resource utilization across a fleet of machines.

Strobelight is composed of a number of profilers, each profiler collects a certain type of profile. This can include CPU, GPU, Memory, or other types of profiles.

## gpuevent profiler
The `gpuevent` profiler attaches to `cudaLaunchKernel` events and collects information about kernels being launched, including demangled name, arguments, stacks, dimensions, etc.

## Getting Started

### Prerequisites

- A Linux-based system.
- Gpu host with [NVIDIA CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
- Cuda binary for testing
- cmake

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/facebookincubator/strobelight.git
   ```

2. Navigate to the project directory and follow the build instructions:
   ```bash
    cd strobelight
    ./scripts/setup.sh -u
   ```

### Usage

Once build is done, you can run the generated binary on any cuda pid
   ```bash
   $ strobelight/src/_build/profilers/gpuevent_snoop --help
Usage: gpuevent_snoop [OPTION...]
GpuEventSnoop.

Traces GPU kernel function execution and its input parameters

USAGE: ./gpuevent_snoop -p PID [-v] [-d duration_sec]

  -a, --args                   Collect Kernel Launch Arguments
  -d, --duration=SEC           Trace for given number of seconds
  -p, --pid=PID                Trace process with given PID
  -r, --rb-count=CNT           RingBuf max entries
  -s, --stacks                 Collect Kernel Launch Stacks
  -v, --verbose                Verbose debug output
  -?, --help                   Give this help list
   --usage                      Give a short usage message
   ```

 ```bash
./gpuevent_snoop  -p <pid> -a -s
Found Symbol cudaLaunchKernel at /data/users/rihams/fbsource/buck-out/v2/gen/fbcode/183d20b7e60f209b/strobelight/oss/src/cuda_example/__cuda_kernel_example__/cuda_kernel_example Offset: 0xca480
Started profiling at Thu Apr  4 13:20:28 2024
cuda_kernel_exa [4024506] KERNEL [0x269710] STREAM 0x0                GRID (1,1,1) BLOCK (256,1,1) add_vectors(double*, double*, do...
Args: add_vectors arg0=0x7f2096800000
double arg1=0x7f2096800400
double arg2=0x7f2096800800
double arg3=0x100000064
int arg4=0x7ffc2a866690
Stack:
00000000002cb480: cudaLaunchKernel @ 0x2cb480+0x0
000000000026a050: main @ 0x26a050+0x912
000000000002c5f0: libc_start_call_main @ 0x2c5f0+0x67
000000000002c690: libc_start_main_alias_2 @ 0x2c690+0x88
0000000000269330: _start @ 0x269330+0x21

...

 ```

## Contributing

We welcome contributions from the community. To contribute:

1. Fork the repository and create a new branch from `main`.
2. Make your changes, ensuring they adhere to the project's coding standards.
3. Submit a pull request, including a detailed description of your changes.

For more information, please refer to the [Contributing Guide](https://github.com/facebookincubator/strobelight/blob/main/CONTRIBUTING.md).

## License

Strobelight is licensed under the Apache License, Version 2.0. See the [LICENSE](https://github.com/facebookincubator/strobelight/blob/main/LICENSE) file for more details.

## Acknowledgements

This project is maintained by Meta's engineering team and is open to community contributions. We thank all contributors for their efforts in improving this project.
