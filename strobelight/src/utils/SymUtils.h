// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstdio>
#include <filesystem>
#include <map>
#include <set>
#include <string>
#include <vector>
#include "blazesym.h" // @manual=fbsource//third-party/rust:blazesym-c-cxx

namespace facebook::strobelight::oss {
struct StackFrame {
  std::string name;
  size_t address;
  std::string module;
  std::string file;
  size_t line;
  size_t offset;
  bool inlines;
  void print() {
    printf(
        "%016lx: %s @ 0x%lx+0x%lx\n", address, name.c_str(), address, offset);
  }
};

struct SymbolInfo {
  std::string name;
  std::vector<std::string> args;
};

class SymUtils {
 public:
  explicit SymUtils(pid_t pid) : pid_(pid) {
    symbolizer_ = blaze_symbolizer_new();
  }

  std::vector<std::pair<std::string, size_t>> findSymbolOffsets(
      const std::string& symName,
      bool searchAllMappings = true,
      bool exitOnFirstMatch = false);

  std::vector<StackFrame> getStackByAddrs(uint64_t* stack, size_t stack_sz);

  SymbolInfo getSymbolByAddr(size_t addr, bool parseArgs = false);

  ~SymUtils() {
    if (symbolizer_) {
      blaze_symbolizer_free(symbolizer_);
    }
  }

 private:
  pid_t pid_;
  struct blaze_symbolizer* symbolizer_;
  std::map<size_t, SymbolInfo> cachedSyms_;
};
} // namespace facebook::strobelight::oss
