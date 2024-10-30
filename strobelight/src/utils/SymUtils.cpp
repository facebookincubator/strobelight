// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "SymUtils.h"
#include <elf.h>
#include <fcntl.h>
#include <gelf.h>
#include <libelf.h>
#include <stdio.h>
#include <unistd.h>
#include <iostream>
#include <regex>
#include "Guard.h"
#include "ProcUtils.h"

namespace facebook::strobelight::oss {

const std::string kUnknownSymbol = "[Unknown]";

bool findSymbolOffsetInFile(
    const std::string& elfPath,
    const std::string& symbolName,
    size_t& symAddr) {
  const char* path = elfPath.c_str();
  Elf* elf;
  int fd;

  if (elf_version(EV_CURRENT) == EV_NONE) {
    fmt::print(stderr, "libelf initialization failed: {}\n", elf_errmsg(-1));
    return false;
  }

  fd = open(path, O_RDONLY);
  if (fd < 0) {
    return false;
  }
  auto guard = Guard([&] {
    if (elf) {
      elf_end(elf);
    }
    close(fd);
  });

  elf = elf_begin(fd, ELF_C_READ, nullptr);
  if (!elf) {
    return false;
  }

  Elf_Kind ekind = elf_kind(elf);
  if (ekind != ELF_K_ELF) {
    return false;
  }
  int eclass = gelf_getclass(elf);
  if (eclass != ELFCLASS64) {
    return false;
  }
  size_t shdr_stridx;
  if (elf_getshdrstrndx(elf, &shdr_stridx)) {
    return false;
  }

  // Elf is corrupted/truncated, avoid calling elf_strptr.
  if (!elf_rawdata(elf_getscn(elf, shdr_stridx), nullptr)) {
    return false;
  }

  Elf64_Ehdr* ehdr = elf64_getehdr(elf);
  Elf_Scn* scn = nullptr;
  while ((scn = elf_nextscn(elf, scn)) != nullptr) {
    GElf_Shdr sh;
    Elf_Data* data;
    int idx, symstrs_idx;
    Elf64_Sym* sym;

    if (gelf_getshdr(scn, &sh) != &sh) {
      fmt::print(
          stderr,
          "failed to get section(%ld) header from %s\n",
          elf_ndxscn(scn),
          path);
      return false;
    }

    // we only care about symbols table
    if (sh.sh_type != SHT_SYMTAB && sh.sh_type != SHT_DYNSYM) {
      continue;
    }

    idx = elf_ndxscn(scn);
    symstrs_idx = sh.sh_link;
    const char* name = elf_strptr(elf, shdr_stridx, sh.sh_name);
    if (!name) {
      fmt::print(stderr, "failed to get section(%d) name from %s\n", idx, path);
      return false;
    }

    data = elf_getdata(scn, 0);
    if (!data) {
      fmt::print(
          stderr,
          "failed to get section(%d) data from %s(%s)\n",
          idx,
          name,
          path);
      return false;
    }

    sym = (Elf64_Sym*)data->d_buf;
    size_t n = sh.sh_size / sh.sh_entsize;
    for (size_t i = 0; i < n; i++, sym++) {
      if (ELF64_ST_TYPE(sym->st_info) != STT_FUNC) {
        continue;
      }

      name = elf_strptr(elf, symstrs_idx, sym->st_name);
      if (!name) {
        continue;
      }
      if (symbolName == name) {
        // Get the section by index
        if (sym->st_shndx > ehdr->e_shnum) {
          fmt::print(
              "ERROR: st_shndx = {} > e_shnum = {}\n",
              sym->st_shndx,
              ehdr->e_shnum);
          return false;
        }
        auto sym_scn = elf_getscn(elf, sym->st_shndx);
        GElf_Shdr symbolShdr;
        if (gelf_getshdr(sym_scn, &symbolShdr) == nullptr) {
          fmt::print(
              stderr, "Failed to get section header for section {} \n", i);

          return false;
        }
        symAddr = sym->st_value - (symbolShdr.sh_addr - symbolShdr.sh_offset);
        return true;
      }
    }
  }
  return false;
}

bool findSymbolOffsetInMMap(
    const pid_t pid,
    const MemoryMapping& mm,
    const std::string& symName,
    size_t& addr) {
  const auto& libExePath = fmt::format(
      "/proc/{}/map_files/{:x}-{:x}", pid, mm.startAddr, mm.endAddr);
  return findSymbolOffsetInFile(libExePath.c_str(), symName, addr);
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
    size_t offset;
    bool symbolFound = findSymbolOffsetInFile(path.c_str(), symbolName, offset);
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
    if (mapping.name.empty() ||
        searchedMappings.find(mapping.name) != searchedMappings.end()) {
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
  // @lint-ignore-every CLANGTIDY facebook-hte-StdRegexIsAwful
  std::regex pattern("\\b\\w+<([^<>]|<([^<>]|<[^<>]*>)*>)*>|\\b\\w+");
  // Create an iterator to iterate over all matches in the signature
  // @lint-ignore-every CLANGTIDY facebook-hte-StdRegexIsAwful
  std::sregex_iterator it(signature.begin(), signature.end(), pattern);
  std::sregex_iterator end;
  // Iterate over all matches and push them into the args vector
  while (it != end) {
    args.push_back(it->str());
    ++it;
  }
  return args;
}

SymbolInfo SymUtils::getSymbolByAddr(size_t addr, bool parseArgs) {
  if (cachedSyms_.find(addr) != cachedSyms_.end()) {
    return cachedSyms_[addr];
  }
  const struct blaze_syms* syms;
  const struct blaze_sym* sym;

  struct blaze_symbolize_src_process src = {
      .type_size = sizeof(src),
      .pid = (uint32_t)pid_,
  };

  uint64_t stack[1] = {addr};
  syms = blaze_symbolize_process_abs_addrs(symbolizer_, &src, stack, 1);

  if (!syms || syms->cnt == 0 || !syms->syms[0].name) {
    return {kUnknownSymbol, {}};
  }

  sym = &syms->syms[0];
  std::string symName = sym->name;
  blaze_syms_free(syms);
  if (!parseArgs) {
    return {symName, {}};
  }

  return {symName, parseFunctionArgs(symName)};
}

std::vector<StackFrame> SymUtils::getStackByAddrs(
    uint64_t* stack,
    size_t stack_sz) {
  std::vector<StackFrame> frames;

  const struct blaze_syms* syms;
  const struct blaze_sym* sym;
  const struct blaze_symbolize_inlined_fn* inlined;

  struct blaze_symbolize_src_process src = {
      .type_size = sizeof(src),
      .pid = (uint32_t)pid_,
  };

  syms = blaze_symbolize_process_abs_addrs(symbolizer_, &src, stack, stack_sz);

  if (!syms) {
    fmt::print(stderr, "Failed to symbolize stack\n");
    return frames;
  }

  auto guard = Guard([&] { blaze_syms_free(syms); });

  frames.reserve(syms->cnt * 2); // Accounting for potential inlined symbols.

  for (size_t i = 0; i < syms->cnt; i++) {
    if (syms->syms[i].name == NULL) {
      continue;
    }

    sym = &syms->syms[i];

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
