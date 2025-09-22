// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#ifdef FBCODE_STROBELIGHT
#include <bpf/vmlinux/vmlinux.h>
#else
#include "vmlinux.h"
#endif

#include <bpf/bpf_core_read.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

#include "gpuevent_snoop.h"

struct {
  __uint(type, BPF_MAP_TYPE_RINGBUF);
} rb SEC(".maps");

const volatile struct {
  bool debug;
  bool capture_args;
  bool capture_stack;
} prog_cfg = {
    // These defaults will be overridden from user space
    .debug = true,
    .capture_args = true,
    .capture_stack = true,
};

#define bpf_printk_debug(fmt, ...)    \
  ({                                  \
    if (prog_cfg.debug)               \
      bpf_printk(fmt, ##__VA_ARGS__); \
  })

// The caller uses registers to pass the first 6 arguments to the callee.  Given
// the arguments in left-to-right order, the order of registers used is: %rdi,
// %rsi, %rdx, %rcx, %r8, and %r9. Any remaining arguments are passed on the
// stack in reverse order so that they can be popped off the stack in order.
#define SP_OFFSET(offset) (void*)PT_REGS_SP(ctx) + offset * 8

SEC("uprobe")
int BPF_KPROBE(
    handle_cuda_launch,
    u64 func_off,
    u64 grid_xy,
    u64 grid_z,
    u64 block_xy,
    u64 block_z,
    uintptr_t argv) {
  struct gpukern_sample* e = bpf_ringbuf_reserve(&rb, sizeof(*e), 0);
  if (!e) {
    bpf_printk_debug("Failed to allocate ringbuf entry");
    return 0;
  }

  struct task_struct* task = bpf_get_current_task_btf();

  e->pid = bpf_get_current_pid_tgid() >> 32;
  e->ppid = task->real_parent->tgid;
  bpf_get_current_comm(&e->comm, sizeof(e->comm));

  e->kern_func_off = func_off;
  e->grid_x = (u32)grid_xy;
  e->grid_y = (u32)(grid_xy >> 32);
  e->grid_z = (u32)grid_z;
  e->block_x = (u32)block_xy;
  e->block_y = (u32)(block_xy >> 32);
  e->block_z = (u32)block_z;

  bpf_probe_read_user(&e->stream, sizeof(uintptr_t), SP_OFFSET(2));

  if (prog_cfg.capture_args) {
    // Read the Cuda Kernel Launch Arguments
    for (int i = 0; i < MAX_GPUKERN_ARGS; i++) {
      const void* arg_addr;
      // We don't know how many argument this kernel has until we parse the
      // signature, so we always attemps to read the maximum number of args,
      // even if some of these arg values are not valid.
      bpf_probe_read_user(
          &arg_addr, sizeof(u64), (const void*)(argv + i * sizeof(u64)));

      bpf_probe_read_user(&e->args[i], sizeof(arg_addr), arg_addr);
    }
  }

  if (prog_cfg.capture_stack) {
    // Read the Cuda Kernel Launch Stack
    e->ustack_sz =
        bpf_get_stack(ctx, e->ustack, sizeof(e->ustack), BPF_F_USER_STACK) /
        sizeof(uint64_t);
  }

  bpf_ringbuf_submit(e, 0);
  return 0;
}

char LICENSE[] SEC("license") = "Dual MIT/GPL";
