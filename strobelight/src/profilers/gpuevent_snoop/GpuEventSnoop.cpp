// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <argp.h>
#include <bpf/libbpf.h>
#include <fmt/core.h>
#include <cstdio>
#include <set>
#include <vector>

#include "bpf/gpuevent_snoop.h"
#ifdef FBCODE_STROBELIGHT
#include "strobelight/src/profilers/gpuevent_snoop/gpuevent_snoop.skel.h"
#else
#include "gpuevent_snoop.skel.h"
#endif
#include "strobelight/src/utils/Guard.h"
#include "strobelight/src/utils/SymUtils.h"

#define MAX_FUNC_DISPLAY_LEN 32

static const int64_t RINGBUF_MAX_ENTRIES = 64 * 1024 * 1024;

// static const std::string kCudaLaunchSymName = "cudaLaunchKernel";

// List of CUDA symbols to attach uprobes
static const std::vector<std::string> kCudaSymbols = {
    "cudaLaunchKernel",
    "cudaLaunchCooperativeKernel",
    "cudaGraphLaunch",
    "cudaMalloc",
    "cudaFree",
    "cudaMemcpy",
    "cudaMemcpyAsync",
    "cudaStreamCreate",
    "cudaStreamDestroy",
    "cudaStreamSynchronize",
    "cudaEventRecord",
    "cudaEventSynchronize",
    "cudaEventElapsedTime",
    "cudaDeviceSynchronize"
};


using namespace facebook::strobelight::oss;

static struct env {
  bool verbose;
  pid_t pid;
  size_t rb_count;
  size_t duration_sec;
  bool args;
  bool stacks;
} env;

static const struct argp_option opts[] = {
    {"verbose", 'v', nullptr, 0, "Verbose debug output", 0},
    {"pid", 'p', "PID", 0, "Trace process with given PID", 0},
    {"rb-count", 'r', "CNT", 0, "RingBuf max entries", 0},
    {"duration", 'd', "SEC", 0, "Trace for given number of seconds", 0},
    {"args", 'a', nullptr, 0, "Collect Kernel Launch Arguments", 0},
    {"stacks", 's', nullptr, 0, "Collect Kernel Launch Stacks", 0},
    {},
};

const char argp_program_doc[] =
    "GpuEventSnoop.\n"
    "\n"
    "Traces GPU kernel function execution and its input parameters \n"
    "\n"
    "USAGE: ./gpuevent_snoop -p PID [-v] [-d duration_sec]\n";

static error_t parse_arg(int key, char* arg, struct argp_state* state) {
  switch (key) {
    case 'v':
      env.verbose = true;
      break;
    case 'a':
      env.args = true;
      break;
    case 's':
      env.stacks = true;
      break;
    case 'p':
      errno = 0;
      env.pid = strtol(arg, nullptr, 10);
      if (errno || env.pid <= 0) {
        fmt::print(stderr, "Invalid pid: {}\n", arg);
        argp_usage(state);
      }
      break;
    case 'r':
      errno = 0;
      env.rb_count = strtol(arg, nullptr, 10);
      if (errno || env.rb_count == 0) {
        fmt::print(stderr, "Invalid ringbuf size: {}\n", arg);
        argp_usage(state);
      }
      break;
    case 'd':
      errno = 0;
      env.duration_sec = strtol(arg, nullptr, 10);
      if (errno) {
        fmt::print(stderr, "Invalid duration: {}\n", arg);
        argp_usage(state);
      }
      break;
    case ARGP_KEY_ARG:
      argp_usage(state);
      break;
    default:
      return ARGP_ERR_UNKNOWN;
  }
  return 0;
}

static const struct argp argp = {
    .options = opts,
    .parser = parse_arg,
    .doc = argp_program_doc,
};

static int libbpf_print_fn(
    enum libbpf_print_level level,
    const char* format,
    va_list args) {
  if (level == LIBBPF_DEBUG && !env.verbose) {
    return 0;
  }
  return vfprintf(stderr, format, args);
}

static int handle_event(void* ctx, void* data, size_t /*data_sz*/) {
    const struct gpukern_sample* e = (struct gpukern_sample*)data;
    SymUtils* symUtils = (SymUtils*)ctx;

    // Get function symbol information
    SymbolInfo symInfo = symUtils->getSymbolByAddr(e->kern_func_off, env.args);

    // Timestamp
    auto timestamp = std::chrono::system_clock::now();
    std::time_t timestamp_t = std::chrono::system_clock::to_time_t(timestamp);

    // Process info
    fmt::print("[TIMESTAMP] {}\n", std::ctime(&timestamp_t));
    fmt::print("[PROCESS] {} [{}] CUDA API EventType {}\n", e->comm, e->pid, e->event_type);

    // Print event-specific information
    switch (e->event_type) {
        case EVENT_CUDA_LAUNCH_KERNEL:
            fmt::print("[CUDA_LAUNCH_KERNEL] Grid: ({},{},{}), Block: ({},{},{})\n",
                e->grid_x, e->grid_y, e->grid_z, e->block_x, e->block_y, e->block_z);
            break;
        case EVENT_CUDA_MALLOC:
            fmt::print("[CUDA_MALLOC] Size: {} bytes, Ptr: 0x{:x}\n", e->args[0], e->args[1]);
            break;
        case EVENT_CUDA_FREE:
            fmt::print("[CUDA_FREE] Ptr: 0x{:x}\n", e->args[0]);
            break;
        case EVENT_CUDA_MEMCPY:
            fmt::print("[CUDA_MEMCPY] Src: 0x{:x}, Dst: 0x{:x}, Size: {} bytes, Kind: {}\n",
                e->args[1], e->args[0], e->args[2], e->args[3]);
            break;
        case EVENT_CUDA_STREAM_CREATE:
            fmt::print("[CUDA_STREAM_CREATE] Stream: 0x{:x}\n", e->args[0]);
            break;
        case EVENT_CUDA_STREAM_DESTROY:
            fmt::print("[CUDA_STREAM_DESTROY] Stream: 0x{:x}\n", e->args[0]);
            break;
        case EVENT_CUDA_EVENT_RECORD:
            fmt::print("[CUDA_EVENT_RECORD] Event: 0x{:x}\n", e->args[0]);
            break;
        case EVENT_CUDA_EVENT_SYNCHRONIZE:
            fmt::print("[CUDA_EVENT_SYNCHRONIZE] Event: 0x{:x}\n", e->args[0]);
            break;
        default:
            fmt::print("[UNKNOWN_CUDA_EVENT]\n");
            break;
    }

    // Print function arguments if requested
    if (env.args) {
        fmt::print("[ARGS] ");
        for (size_t i = 0; i < symInfo.args.size() && i < MAX_GPUKERN_ARGS; i++) {
            fmt::print("{} arg{}=0x{:x} ", symInfo.args[i], i, e->args[i]);
        }
        fmt::print("\n");
    }

    // Print stack trace if requested
    if (env.stacks) {
        fmt::print("[STACK_TRACE]\n");
        auto stack = symUtils->getStackByAddrs((uint64_t*)e->ustack, e->ustack_sz);
        for (auto& frame : stack) {
            fmt::print("  {}\n", frame.name);
        }
    }

    fmt::print("{:-<80}\n", '-');  // Separator
    return 0;
}

bool hasExceededProfilingLimit(
    std::chrono::seconds duration,
    const std::chrono::steady_clock::time_point& startTime) {
  if (duration.count() == 0) { // 0 = profle forever
    return false;
  }

  if (std::chrono::steady_clock::now() - startTime >= duration) {
    fmt::print("Done Profiling: exceeded duration of {}s.\n", duration.count());
    return true;
  }
  return false;
}

int main(int argc, char* argv[]) {
  /* Parse command line arguments */
  int err = argp_parse(&argp, argc, argv, 0, nullptr, nullptr);
  if (err) {
    fmt::print(stderr, "Failed to parse arguments\n");
    return err;
  }

  if (env.pid == 0) {
    fmt::print(stderr, "please specify PID\n");
    return -1;
  }

  struct ring_buffer* ringBuffer = nullptr;
  struct gpuevent_snoop_bpf* skel;
  std::vector<bpf_link*> links;

  /* Set up libbpf errors and debug info callback */
  libbpf_set_print(libbpf_print_fn);

  /* Load and verify BPF application */
  skel = gpuevent_snoop_bpf__open();
  if (!skel) {
    fmt::print(
        stderr, "Failed to open and load BPF skeleton - err: {}\n", errno);
    return -1;
  }

  SymUtils symUtils(env.pid);

  /* Init Read only variables and maps */
  bpf_map__set_max_entries(
      skel->maps.rb, env.rb_count > 0 ? env.rb_count : RINGBUF_MAX_ENTRIES);
  skel->rodata->prog_cfg.capture_args = env.args;

  /* Load & verify BPF programs */
  err = gpuevent_snoop_bpf__load(skel);
  if (err) {
    fmt::print(stderr, "Failed to load and verify BPF skeleton\n");
    return -1;
  }

  auto guard = Guard([&] {
    gpuevent_snoop_bpf__destroy(skel);
    for (auto link : links) {
      bpf_link__destroy(link);
    }
    ring_buffer__free(ringBuffer);
  });

 /* Attach Uprobes for CUDA API tracepoints */
  for (const auto& symbol : kCudaSymbols) {
    auto offsets = symUtils.findSymbolOffsets(symbol);
    if (offsets.empty()) {
      fmt::print(stderr, "Failed to find symbol {}\n", symbol);
      continue;
    }
    for (const auto& offset : offsets) {
      auto link = bpf_program__attach_uprobe(
          skel->progs.handle_cuda_launch, false, env.pid,
          offset.first.c_str(), offset.second);
      if (link) {
        links.emplace_back(link);
      }
    }
  }

  /* Set up ring buffer polling */
  ringBuffer = ring_buffer__new(
      bpf_map__fd(skel->maps.rb), handle_event, (void*)&symUtils, nullptr);
  if (!ringBuffer) {
    fmt::print(stderr, "Failed to create ring buffer\n");
    return -1;
  }

  auto startTime = std::chrono::steady_clock::now();
  auto duration = std::chrono::seconds(env.duration_sec);

  std::time_t ttp =
      std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  fmt::print("Started profiling at {}", std::ctime(&ttp));

  while (!hasExceededProfilingLimit(duration, startTime)) {
    err = ring_buffer__poll(ringBuffer, 100 /* timeout, ms */);
    /* Ctrl-C will cause -EINTR */
    if (err == -EINTR) {
      err = 0;
      break;
    }
    if (err < 0) {
      fmt::print(stderr, "Error polling perf buffer: {}\n", err);
      break;
    }
  }
  ring_buffer__consume(ringBuffer);

  ttp = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  fmt::print("Stopped profiling at {}", std::ctime(&ttp));

  return err < 0 ? -err : 0;
}
