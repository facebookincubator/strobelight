// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "ProcUtils.h"
namespace facebook::strobelight::oss {

// Stringify helpers so the %[^\n] conversion below can carry a field width of
// PATH_MAX, bounding how many path characters sscanf writes into buf.
#define STR(x) #x
#define XSTR(x) STR(x)

bool parseMemoryMapLine(const std::string& line, MemoryMapping& module) {
  char read;
  char write;
  char exec;
  char share;
  char buf[PATH_MAX + 1];
  buf[0] = '\0';
  auto res = std::sscanf(
      line.c_str(),
      // From Kernel source fs/proc/task_mmu.c. The path field carries a
      // PATH_MAX width specifier: the kernel octal-escapes chars <= 0x20 in
      // /proc/<pid>/maps (e.g. space -> \040), so a near-PATH_MAX raw path can
      // render up to ~4x longer. Without the bound, sscanf would overflow buf
      // (a PATH_MAX + 1 stack buffer). See T267287915.
      "%lx-%lx %c%c%c%c %llx %lx:%lx %lu %" XSTR(PATH_MAX) "[^\n]",
      &module.startAddr,
      &module.endAddr,
      &read,
      &write,
      &exec,
      &share,
      &module.fileOffset,
      &module.devMajor,
      &module.devMinor,
      &module.inode,
      buf);
  // The module name might be empty, where res would be 10 and buf untouched
  if (res < 10) {
    return false;
  }

  module.name = buf;
  module.readable = (read == 'r');
  module.writable = (write == 'w');
  module.executable = (exec == 'x');
  module.shared = (share == 's');

  return true;
}

#undef XSTR
#undef STR

std::string getProcFolderPath(pid_t pid, const char* path) {
  return fmt::format("/proc/{}/{}", pid, path);
}

std::vector<MemoryMapping> ProcUtils::getAllMemoryMappings(pid_t pid) {
  std::vector<MemoryMapping> mappings;

  std::string filename = getProcFolderPath(pid, "maps");

  std::ifstream fs(filename.c_str());
  if (!fs.is_open()) {
    fmt::print(
        stderr,
        "[{}] Unable to open procfs mapfile: '{}'\n",
        pid,
        filename.c_str());
    return mappings;
  }

  MemoryMapping module;
  std::string line;
  while (std::getline(fs, line)) {
    if (!parseMemoryMapLine(line, module)) {
      fmt::print(
          "[pid: {}] Error reading from procfs mapfile: '{}'",
          pid,
          filename.c_str());
      return mappings;
    }

    mappings.push_back(module);
  }
  fs.close();
  return mappings;
}
} // namespace facebook::strobelight::oss
