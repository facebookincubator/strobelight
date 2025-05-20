// Copyright (c) Meta Platforms, Inc.
// Licensed under the MIT License.

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
    .debug = true,
    .capture_args = true,
    .capture_stack = true,
};

#define bpf_printk_debug(fmt, ...)    \
  ({                                  \
    if (prog_cfg.debug)               \
      bpf_printk(fmt, ##__VA_ARGS__); \
  })

#define SP_OFFSET(offset) (void*)PT_REGS_SP(ctx) + offset * 8

// CUDA Kernel Launch Tracepoint
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
    if (!e) return 0;

    struct task_struct* task = (struct task_struct*)bpf_get_current_task();
    e->pid = bpf_get_current_pid_tgid() >> 32;
    e->ppid = BPF_CORE_READ(task, real_parent, tgid);
    bpf_get_current_comm(&e->comm, sizeof(e->comm));

    e->event_type = EVENT_CUDA_LAUNCH_KERNEL;
    e->kern_func_off = func_off;
    e->grid_x = (u32)grid_xy;
    e->grid_y = (u32)(grid_xy >> 32);
    e->grid_z = (u32)grid_z;
    e->block_x = (u32)block_xy;
    e->block_y = (u32)(block_xy >> 32);
    e->block_z = (u32)block_z;

    bpf_probe_read_user(&e->stream, sizeof(uintptr_t), SP_OFFSET(2));

    if (prog_cfg.capture_args) {
        for (int i = 0; i < MAX_GPUKERN_ARGS; i++) {
            const void* arg_addr;
            bpf_probe_read_user(&arg_addr, sizeof(u64), (const void*)(argv + i * sizeof(u64)));
            bpf_probe_read_user(&e->args[i], sizeof(arg_addr), arg_addr);
        }
    }

    if (prog_cfg.capture_stack) {
        e->ustack_sz = bpf_get_stack(ctx, e->ustack, sizeof(e->ustack), BPF_F_USER_STACK) /
                       sizeof(uint64_t);
    }

    bpf_ringbuf_submit(e, 0);
    return 0;
}

// CUDA Memory Management Tracepoints
SEC("uprobe")
int BPF_KPROBE(handle_cuda_malloc, size_t size, void **devPtr) {
    struct gpukern_sample* e = bpf_ringbuf_reserve(&rb, sizeof(*e), 0);
    if (!e) return 0;

    e->pid = bpf_get_current_pid_tgid() >> 32;
    bpf_get_current_comm(&e->comm, sizeof(e->comm));

    e->event_type = EVENT_CUDA_MALLOC;
    e->kern_func_off = (uint64_t)PT_REGS_IP(ctx);
    e->args[0] = size;
    bpf_probe_read_user(&e->args[1], sizeof(void *), devPtr);
    
    bpf_ringbuf_submit(e, 0);
    return 0;
}

SEC("uprobe")
int BPF_KPROBE(handle_cuda_free, void *devPtr) {
    struct gpukern_sample* e = bpf_ringbuf_reserve(&rb, sizeof(*e), 0);
    if (!e) return 0;

    e->pid = bpf_get_current_pid_tgid() >> 32;
    bpf_get_current_comm(&e->comm, sizeof(e->comm));

    e->event_type = EVENT_CUDA_FREE;
    e->kern_func_off = (uint64_t)PT_REGS_IP(ctx);
    e->args[0] = (uint64_t)devPtr;

    bpf_ringbuf_submit(e, 0);
    return 0;
}

// CUDA Memory Copy Tracepoints
SEC("uprobe")
int BPF_KPROBE(handle_cuda_memcpy, void *dst, const void *src, size_t count, int kind) {
    struct gpukern_sample* e = bpf_ringbuf_reserve(&rb, sizeof(*e), 0);
    if (!e) return 0;

    e->pid = bpf_get_current_pid_tgid() >> 32;
    bpf_get_current_comm(&e->comm, sizeof(e->comm));

    e->event_type = EVENT_CUDA_MEMCPY;
    e->kern_func_off = (uint64_t)PT_REGS_IP(ctx);
    e->args[0] = (uint64_t)dst;
    e->args[1] = (uint64_t)src;
    e->args[2] = count;
    e->args[3] = kind;

    bpf_ringbuf_submit(e, 0);
    return 0;
}

// CUDA Stream and Event Management Tracepoints
SEC("uprobe")
int BPF_KPROBE(handle_cuda_stream_create, void **stream) {
    struct gpukern_sample* e = bpf_ringbuf_reserve(&rb, sizeof(*e), 0);
    if (!e) return 0;

    e->pid = bpf_get_current_pid_tgid() >> 32;
    bpf_get_current_comm(&e->comm, sizeof(e->comm));

    e->event_type = EVENT_CUDA_STREAM_CREATE;
    bpf_probe_read_user(&e->args[0], sizeof(void *), stream);

    bpf_ringbuf_submit(e, 0);
    return 0;
}

SEC("uprobe")
int BPF_KPROBE(handle_cuda_stream_destroy, void *stream) {
    struct gpukern_sample* e = bpf_ringbuf_reserve(&rb, sizeof(*e), 0);
    if (!e) return 0;

    e->pid = bpf_get_current_pid_tgid() >> 32;
    bpf_get_current_comm(&e->comm, sizeof(e->comm));

    e->event_type = EVENT_CUDA_STREAM_DESTROY;
    e->args[0] = (uint64_t)stream;

    bpf_ringbuf_submit(e, 0);
    return 0;
}

// CUDA Synchronization Tracepoints
SEC("uprobe")
int BPF_KPROBE(handle_cuda_device_synchronize) {
    struct gpukern_sample* e = bpf_ringbuf_reserve(&rb, sizeof(*e), 0);
    if (!e) return 0;

    e->pid = bpf_get_current_pid_tgid() >> 32;
    bpf_get_current_comm(&e->comm, sizeof(e->comm));

    e->event_type = EVENT_CUDA_DEVICE_SYNCHRONIZE;
    e->kern_func_off = (uint64_t)PT_REGS_IP(ctx);

    bpf_ringbuf_submit(e, 0);
    return 0;
}

char LICENSE[] SEC("license") = "Dual MIT/GPL";

