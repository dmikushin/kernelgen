//===- Elf.h - Old ELF manipulation functions (deprecated) ----------------===//
//
//     KernelGen -- A prototype of LLVM-based auto-parallelizing Fortran/C
//        compiler for NVIDIA GPUs, targeting numerical modeling code.
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef KERNELGEN_ELF_H
#define KERNELGEN_ELF_H

#include "IO.h"
#include <gelf.h>
#include <libelf.h>
#include <regex.h>
#include <string>
#include <vector>

namespace util {
namespace elf {

// Defines regular expression processor.
class cregex {
  regex_t regex;

public:
  bool matches(std::string value);

  cregex(std::string pattern, int flags);
  ~cregex();
};

class celf;

// Defines ELF section.
class csection {
protected:
  celf *e;
  Elf_Scn *scn;
  std::string name;

  csection();
  csection(celf *e, Elf_Scn *scn, std::string name);

public:
  void addSymbol(std::string name, const char *data, size_t length);

  friend class celf;
};

class csymtab;

// Defines ELF symbol.
class csymbol {
  const celf *e;
  std::string name;
  char *data;
  size_t size;
  int shndx;
  bool data_loaded, data_allocated;

  csymbol();

public:
  const std::string &getName() const;
  const char *getData();
  size_t getSize() const;

  csymbol(const celf *e, std::string name, char *data, size_t size, int shndx);

  ~csymbol();

  friend class csymtab;
};

// Defines ELF symbol table section.
class csymtab : public csection {
  int nsymbols;
  csymbol *symbols;

public:

  // Find symbols names by the specified pattern.
  std::vector<csymbol *> find(cregex &regex) const;

  csymtab(const csection *section);
  ~csymtab();
};

// Defines ELF image section.
class celf {
  // Fields for the underlying input and output
  // file descriptors.
  util::io::cfiledesc *ifd;
  bool managed_fd;
  std::string ofilename;

  Elf *e;
  csection *sections_array;
  std::vector<csection *> sections;
  csymtab *symtab;
  GElf_Ehdr header;
  bool opened;

  void open();

public:
  const csymtab *getSymtab();
  const GElf_Ehdr *getHeader();
  csection *getSection(std::string name);

  void setSymtab32(Elf32_Sym *sym, int count);
  void setSymtab64(Elf64_Sym *sym, int count);

  void setStrtab(GElf_Ehdr *ehdr, const char *content, size_t length);
  void setData(const char *content, size_t length);

  celf(std::string ifilename, std::string ofilename);
  celf(util::io::cfiledesc *ifd, std::string ofilename);
  ~celf();

  friend class csection;
  friend class csymtab;
  friend class csymbol;
};

}
} // namespace

#endif // KERNELGEN_ELF_H
