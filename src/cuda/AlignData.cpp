/*
 * KernelGen - the LLVM-based compiler with GPU kernels generation over C
 * backend.
 *
 * Copyright (c) 2011 Dmitry Mikushin
 *
 * This software is provided 'as-is', without any express or implied warranty.
 * In no event will the authors be held liable for any damages arising
 * from the use of this software.
 * Permission is granted to anyone to use this software for any purpose,
 * including commercial applications, and to alter it and redistribute it
 * freely,
 * subject to the following restrictions:
 *
 * 1. The origin of this software must not be misrepresented;
 * you must not claim that you wrote the original software.
 * If you use this software in a product, an acknowledgment
 * in the product documentation would be appreciated but is not required.
 * 2. Altered source versions must be plainly marked as such,
 * and must not be misrepresented as being the original software.
 * 3. This notice may not be removed or altered from any source distribution.
 */

#include "Cuda.h"

#include <cstdio>
#include <elf.h>
#include <fcntl.h>
#include <gelf.h>
#include <string.h>
#include <unistd.h>
#include <vector>

using namespace std;

// Align cubin global data to the specified boundary.
void kernelgen::bind::cuda::CUBIN::AlignData(const char *cubin, size_t align) {
  int fd = -1;
  Elf *e = NULL;
  try {
    //
    // 1) First, load the ELF file.
    //
    if ((fd = open(cubin, O_RDWR)) < 0) {
      fprintf(stderr, "Cannot open file %s\n", cubin);
      throw;
    }

    if ((e = elf_begin(fd, ELF_C_RDWR, e)) == 0) {
      fprintf(stderr, "Cannot read ELF image from %s\n", cubin);
      throw;
    }

    //
    // 2) Find ELF section containing the symbol table and
    // load its data. Meanwhile, search for the constant, global and
    // initialized global sections.
    //
    size_t shstrndx;
    if (elf_getshdrstrndx(e, &shstrndx)) {
      fprintf(stderr, "elf_getshdrstrndx() failed for %s: %s\n", cubin,
              elf_errmsg(-1));
      throw;
    }
    Elf_Data *symbols = NULL;
    int nsymbols = 0, nsections = 0, link = 0;
    Elf_Scn *scn = elf_nextscn(e, NULL);
    Elf_Scn *sconst, *sglobal, *sglobal_init = NULL;
    GElf_Shdr shconst, shglobal, shglobal_init;
    int iconst = -1, iglobal = -1, iglobal_init = -1;
    for (int i = 1; scn != NULL; scn = elf_nextscn(e, scn), i++, nsections++) {
      GElf_Shdr shdr;

      if (!gelf_getshdr(scn, &shdr)) {
        fprintf(stderr, "gelf_getshdr() failed for %s: %s\n", cubin,
                elf_errmsg(-1));
        throw;
      }

      if (shdr.sh_type == SHT_SYMTAB) {
        symbols = elf_getdata(scn, NULL);
        if (!symbols) {
          fprintf(stderr, "elf_getdata() failed for %s: %s\n", cubin,
                  elf_errmsg(-1));
          throw;
        }
        if (shdr.sh_entsize)
          nsymbols = shdr.sh_size / shdr.sh_entsize;
        link = shdr.sh_link;
      }

      char *name = NULL;
      if ((name = elf_strptr(e, shstrndx, shdr.sh_name)) == NULL)
        throw;

      // Two variants for different CC: sm_2x or sm_3x.
      if (!strcmp(name, ".nv.constant2") || !strcmp(name, ".nv.constant3")) {
        iconst = i;
        sconst = scn;
        shconst = shdr;
      } else if (!strcmp(name, ".nv.global")) {
        iglobal = i;
        sglobal = scn;
        shglobal = shdr;
      } else if (!strcmp(name, ".nv.global.init")) {
        iglobal_init = i;
        sglobal_init = scn;
        shglobal_init = shdr;
      }
    }

    if (!symbols)
      return;

    //
    // 3) Count what size is needed to keep constant, global,
    // and initialized global symbols, if they were aligned.
    // Also update the sizes and offsets for individual initialized symbols.
    //
    vector<char>::size_type szconst_new = 0;
    vector<char>::size_type szglobal_new = 0, szglobal_init_new = 0;
    for (int isymbol = 0; isymbol < nsymbols; isymbol++) {
      GElf_Sym symbol;
      gelf_getsym(symbols, isymbol, &symbol);

      if (symbol.st_shndx == iconst) {
        size_t szaligned = symbol.st_size;
        if (szaligned % align)
          szaligned += align - szaligned % align;
        szconst_new += szaligned;
      } else if (symbol.st_shndx == iglobal) {
        symbol.st_value = szglobal_new;
        if (symbol.st_size % align)
          symbol.st_size += align - symbol.st_size % align;
        szglobal_new += symbol.st_size;

        if (!gelf_update_sym(symbols, isymbol, &symbol)) {
          fprintf(stderr, "gelf_update_sym() failed for %s: %s\n", cubin,
                  elf_errmsg(-1));
          throw;
        }
      } else if (symbol.st_shndx == iglobal_init) {
        size_t szaligned = symbol.st_size;
        if (szaligned % align)
          szaligned += align - szaligned % align;
        szglobal_init_new += szaligned;
      }
    }

    //
    // 4) Add new data for aligned constants section.
    //
    vector<char> vconst_new;
    if (iconst != -1) {
      vconst_new.resize(szconst_new);

      Elf_Data *data = elf_getdata(sconst, NULL);
      if (!data) {
        fprintf(stderr, "elf_newdata() failed: %s\n", elf_errmsg(-1));
        throw;
      }
      char *dconst = (char *)data->d_buf;

      char *pconst = (char *)dconst;
      char *pconst_new = (char *)&vconst_new[0];
      memset(pconst_new, 0, szconst_new);
      szconst_new = 0;
      for (int isymbol = 0; isymbol < nsymbols; isymbol++) {
        GElf_Sym symbol;
        gelf_getsym(symbols, isymbol, &symbol);

        if (symbol.st_shndx == iconst) {
          memcpy(pconst_new, pconst + symbol.st_value,
                 symbol.st_size);

          symbol.st_value = szconst_new;
          if (symbol.st_size % align)
            symbol.st_size += align - symbol.st_size % align;
          szconst_new += symbol.st_size;
          pconst_new += symbol.st_size;

          if (!gelf_update_sym(symbols, isymbol, &symbol)) {
            fprintf(stderr, "gelf_update_sym() failed for %s: %s\n", cubin,
                    elf_errmsg(-1));
            throw;
          }
        }
      }

      data->d_buf = (char *)&vconst_new[0];
      data->d_size = szconst_new;

      if (!gelf_update_shdr(sconst, &shconst)) {
        fprintf(stderr, "gelf_update_shdr() failed: %s\n", elf_errmsg(-1));
        throw;
      }
    }

    //
    // 5) Add new data for aligned globals section.
    //
    vector<char> vglobal_new;
    if (iglobal != -1) {
      vglobal_new.resize(szglobal_new);

      Elf_Data *data = elf_getdata(sglobal, NULL);
      if (!data) {
        fprintf(stderr, "elf_newdata() failed: %s\n", elf_errmsg(-1));
        throw;
      }

      data->d_buf = (char *)&vglobal_new[0];
      data->d_size = szglobal_new;

      if (!gelf_update_shdr(sglobal, &shglobal)) {
        fprintf(stderr, "gelf_update_shdr() failed: %s\n", elf_errmsg(-1));
        throw;
      }
    }

    //
    // 6) Add new data for aligned initialized globals section.
    //
    vector<char> vglobal_init_new;
    if (iglobal_init != -1) {
      vglobal_init_new.resize(szglobal_init_new);

      Elf_Data *data = elf_getdata(sglobal_init, NULL);
      if (!data) {
        fprintf(stderr, "elf_newdata() failed: %s\n", elf_errmsg(-1));
        throw;
      }
      char *dglobal_init = (char *)data->d_buf;

      char *pglobal_init = (char *)dglobal_init;
      char *pglobal_init_new = (char *)&vglobal_init_new[0];
      memset(pglobal_init_new, 0, szglobal_init_new);
      szglobal_init_new = 0;
      for (int isymbol = 0; isymbol < nsymbols; isymbol++) {
        GElf_Sym symbol;
        gelf_getsym(symbols, isymbol, &symbol);

        if (symbol.st_shndx == iglobal_init) {
          memcpy(pglobal_init_new, pglobal_init + symbol.st_value,
                 symbol.st_size);

          symbol.st_value = szglobal_init_new;
          if (symbol.st_size % align)
            symbol.st_size += align - symbol.st_size % align;
          szglobal_init_new += symbol.st_size;
          pglobal_init_new += symbol.st_size;

          if (!gelf_update_sym(symbols, isymbol, &symbol)) {
            fprintf(stderr, "gelf_update_sym() failed for %s: %s\n", cubin,
                    elf_errmsg(-1));
            throw;
          }
        }
      }

      data->d_buf = (char *)&vglobal_init_new[0];
      data->d_size = szglobal_init_new;

      if (!gelf_update_shdr(sglobal_init, &shglobal_init)) {
        fprintf(stderr, "gelf_update_shdr() failed: %s\n", elf_errmsg(-1));
        throw;
      }
    }

    //
    // 7) Commit changes into the underlying ELF binary.
    //
    if (elf_update(e, ELF_C_WRITE) == -1) {
      fprintf(stderr, "Cannot update the ELF image %s\n", cubin);
      throw;
    }

    elf_end(e);
    close(fd);
    e = NULL;
  }
  catch (...) {
    if (e)
      elf_end(e);
    if (fd >= 0)
      close(fd);
    throw;
  }
}
