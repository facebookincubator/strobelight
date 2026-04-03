#!/bin/sh
# Copyright (c) Meta Platforms, Inc. and affiliates.

$(dirname "$0")/bpftool btf dump file ${1:-/sys/kernel/btf/vmlinux} format c
