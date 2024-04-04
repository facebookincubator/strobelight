// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "SymUtils.h"
#include <folly/ScopeGuard.h>
#include <folly/experimental/symbolizer/Elf.h>
#include <re2/re2.h>
#include "ProcUtils.h"

namespace facebook::strobelight::oss {

const std::string kUnknownSymbol = "[Unknown]";

bool findSymbolOffsetInFile(
    const folly::symbolizer::ElfFile& elf,
    const std::string& symName,
    size_t& addr) {
  auto sym = elf.getSymbolByName(symName.c_str());
  if (sym.first == nullptr || sym.second == nullptr) {
    return false;
  }
  auto sec = elf.getSectionByIndex(sym.second->st_shndx);
  // Get the offset of the symbol in the section
  addr = sym.second->st_value - (sec->sh_addr - sec->sh_offset);
  return true;
}

bool findSymbolOffsetInMMap(
    const pid_t pid,
    const MemoryMapping& mm,
    const std::string& symName,
    size_t& addr) {
  folly::symbolizer::ElfFile elf;
  const auto& libExePath = fmt::format(
      "/proc/{}/map_files/{:x}-{:x}", pid, mm.startAddr, mm.endAddr);

  if (elf.openNoThrow(libExePath.c_str()) != 0) {
    return false;
  }
  return findSymbolOffsetInFile(elf, symName, addr);
}

std::vector<std::pair<std::string, size_t>> SymUtils::findSymbolOffsets(
    const std::string& symbolName,
    bool searchAllMappings,
    bool exitOnFirstMatch) {
  std::vector<std::pair<std::string, size_t>> uprobesToAttach;
  // This is a shortcut for the case where we only want to search the main
  // binary, we can skip the rest of the mappings in this case
  if (!searchAllMappings) {
    std::string path = fmt::format("/proc/{}/exe", pid_);
    folly::symbolizer::ElfFile elf;
    size_t offset;
    if (elf.openNoThrow(path.c_str()) != 0) {
      fmt::print(stderr, "Failed to open {} to read ELF data\n", path.c_str());
      return uprobesToAttach;
    }
    bool symbolFound = findSymbolOffsetInFile(elf, symbolName, offset);
    if (symbolFound) {
      uprobesToAttach.emplace_back(path, offset);
      fmt::print(
          "Found CUDA kernel launch symbol at offset [0x{:x}] in {}\n",
          offset,
          path.c_str());
    } else {
      fmt::print("Failed to find {} symbol in {}\n", symbolName, path.c_str());
    }
    return uprobesToAttach;
  }

  // Otherwise, we need to search all mappings for the symbol in case it is in a
  // shared library
  std::set<std::string> searchedMappings;
  for (auto& mapping : ProcUtils::getAllMemoryMappings(pid_)) {
    if (mapping.name.empty() || searchedMappings.contains(mapping.name)) {
      continue;
    }
    searchedMappings.emplace(mapping.name);

    size_t offset;
    bool symbolFound =
        findSymbolOffsetInMMap(pid_, mapping, symbolName, offset);
    if (!symbolFound) {
      continue;
    }

    fmt::print(
        "Found Symbol {} at {} Offset: 0x{:x}\n",
        symbolName,
        mapping.name,
        offset);

    uprobesToAttach.emplace_back(mapping.name, offset);
    if (exitOnFirstMatch) {
      break;
    }
  }
  return uprobesToAttach;
}

std::vector<std::string> parseFunctionArgs(const std::string& signature) {
  std::vector<std::string> args;
  // Define the regular expression pattern to match function arguments
  re2::RE2 pattern("\\b\\w+<([^<>]|<([^<>]|<[^<>]*>)*>)*>|\\b\\w+");
  // Create a RE2 object to search for matches in the signature
  re2::StringPiece input(signature);
  re2::StringPiece match;
  // Iterate over all matches and push them into the args vector
  while (RE2::FindAndConsume(&input, pattern, &match)) {
    args.push_back(match.as_string());
  }
  return args;
}

SymbolInfo SymUtils::getSymbolByAddr(size_t addr, bool parseArgs) {
  if (cachedSyms_.find(addr) != cachedSyms_.end()) {
    return cachedSyms_[addr];
  }
  const struct blaze_result* result;
  const struct blaze_sym* sym;

  struct blaze_symbolize_src_process src = {
      .type_size = sizeof(src),
      .pid = (uint32_t)pid_,
  };

  uintptr_t stack[1] = {addr};
  result = blaze_symbolize_process_abs_addrs(
      symbolizer_, &src, (const uintptr_t*)stack, 1);

  if (!result || result->cnt == 0 || !result->syms[0].name) {
    return {kUnknownSymbol, {}};
  }

  sym = &result->syms[0];
  std::string symName = sym->name;
  blaze_result_free(result);
  if (!parseArgs) {
    return {symName, {}};
  }

  return {symName, parseFunctionArgs(symName)};
}

std::vector<StackFrame> SymUtils::getStackByAddrs(
    uint64_t* stack,
    size_t stack_sz) {
  std::vector<StackFrame> frames;

  const struct blaze_result* result;
  const struct blaze_sym* sym;
  const struct blaze_symbolize_inlined_fn* inlined;

  struct blaze_symbolize_src_process src = {
      .type_size = sizeof(src),
      .pid = (uint32_t)pid_,
  };

  result = blaze_symbolize_process_abs_addrs(
      symbolizer_, &src, (const uintptr_t*)stack, stack_sz);

  if (!result) {
    fmt::print(stderr, "Failed to symbolize stack\n");
    return frames;
  }

  auto guard = folly::makeGuard([&] { blaze_result_free(result); });

  frames.reserve(result->cnt * 2); // Accounting for potential inlined symbols.

  for (size_t i = 0; i < result->cnt; i++) {
    if (result->syms[i].name == NULL) {
      continue;
    }

    sym = &result->syms[i];

    StackFrame frame = {
        .name = sym->name,
        .address = sym->addr,
        .offset = sym->offset,
    };

    if (sym->code_info.file) {
      frame.file = sym->code_info.file;
      frame.line = sym->code_info.line;
    }

    frames.emplace_back(frame);

    for (size_t j = 0; j < sym->inlined_cnt; j++) {
      inlined = &sym->inlined[j];
      StackFrame inlined_frame = {
          .name = sym->name,
          .address = 0,
          .offset = 0,
      };

      if (sym->code_info.file) {
        inlined_frame.file = inlined->code_info.file;
        inlined_frame.line = inlined->code_info.line;
      }
      frames.emplace_back(inlined_frame);
    }
  }
  return frames;
}

} // namespace facebook::strobelight::oss
