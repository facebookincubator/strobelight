// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once
#define TASK_COMM_LEN 16
#define MAX_GPUKERN_ARGS 16

#ifndef MAX_STACK_DEPTH
#define MAX_STACK_DEPTH 128
#endif

enum cuda_event_type {
    EVENT_CUDA_LAUNCH_KERNEL,
    EVENT_CUDA_MALLOC,
    EVENT_CUDA_FREE,
    EVENT_CUDA_MEMCPY,
    EVENT_CUDA_MEMCPY_ASYNC,
    EVENT_CUDA_STREAM_CREATE,
    EVENT_CUDA_STREAM_DESTROY,
    EVENT_CUDA_STREAM_SYNCHRONIZE,
    EVENT_CUDA_EVENT_RECORD,
    EVENT_CUDA_EVENT_SYNCHRONIZE,
    EVENT_CUDA_EVENT_ELAPSED_TIME,
    EVENT_CUDA_DEVICE_SYNCHRONIZE
};

typedef uint64_t stack_trace_t[MAX_STACK_DEPTH];

struct gpukern_sample {
    int pid, ppid;
    char comm[TASK_COMM_LEN];
    uint64_t kern_func_off; // Address of the CUDA API call

    // Fields specific to kernel execution tracking
    int grid_x, grid_y, grid_z;
    int block_x, block_y, block_z;
    uint64_t stream;

    // Generalized args to support multiple CUDA APIs
    uint64_t args[MAX_GPUKERN_ARGS];

    // Stack trace information
    size_t ustack_sz;
    stack_trace_t ustack;

    // New: Event type to distinguish between different CUDA API calls
    uint32_t event_type;
};

/*** original***
struct gpukern_sample {
  int pid, ppid;
  char comm[TASK_COMM_LEN];
  uint64_t kern_func_off;
  int grid_x, grid_y, grid_z;
  int block_x, block_y, block_z;
  uint64_t stream;
  uint64_t args[MAX_GPUKERN_ARGS];
  size_t ustack_sz;
  stack_trace_t ustack;
};
*/
