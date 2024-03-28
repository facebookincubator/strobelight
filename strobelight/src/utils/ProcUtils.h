// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <fmt/core.h>
#include <stdio.h>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace facebook::strobelight::oss {

#define PATH_MAX 4096

struct MemoryMapping {
  uintptr_t startAddr;
  uintptr_t endAddr;
  unsigned long long fileOffset;
  bool readable;
  bool writable;
  bool executable;
  bool shared;
  dev_t devMajor;
  dev_t devMinor;
  ino_t inode;
  std::string name;
};

class ProcUtils {
 public:
  static std::vector<MemoryMapping> getAllMemoryMappings(pid_t pid);
};

} // namespace facebook::strobelight::oss
