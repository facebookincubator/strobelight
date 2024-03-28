// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <stdio.h>
#include <filesystem>
#include <map>
#include <set>
#include <string>
#include <vector>
#include "blazesym/blazesym.h" // @manual=fbsource//third-party/rust:blazesym-c-cxx

namespace facebook::strobelight::oss {

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
