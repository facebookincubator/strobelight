// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once
#define TASK_COMM_LEN 16
#define MAX_GPUKERN_ARGS 16

struct gpukern_sample {
  int pid, ppid;
  char comm[TASK_COMM_LEN];
  uint64_t kern_func_off;
  int grid_x, grid_y, grid_z;
  int block_x, block_y, block_z;
  uint64_t stream;
  uint64_t args[MAX_GPUKERN_ARGS];
};
