# A prototype of LLVM-based auto-parallelizing Fortran/C compiler for NVIDIA GPUs, targeting numerical modeling code.

This is a legacy repository imported from HPCForge.org upon its shut down. It contains the code developed for the KernelGen project:

```bibtex
@inproceedings{Mikushin:2014:KDI:2672598.2672916,
 author = {Mikushin, Dmitry and Likhogrud, Nikolay and Zhang, Eddy Z. and Bergstr\"{o}m, Christopher},
 title = {KernelGen  --  The Design and Implementation of a Next Generation Compiler Platform for Accelerating Numerical Models on GPUs},
 booktitle = {Proceedings of the 2014 IEEE International Parallel \& Distributed Processing Symposium Workshops},
 series = {IPDPSW '14},
 year = {2014},
 isbn = {978-1-4799-4116-2},
 pages = {1011--1020},
 numpages = {10},
 url = {http://dx.doi.org/10.1109/IPDPSW.2014.115},
 doi = {10.1109/IPDPSW.2014.115},
 acmid = {2672916},
 publisher = {IEEE Computer Society},
 address = {Washington, DC, USA},
 keywords = {GPU, LLVM, OpenACC, JIT-compilation, stencils},
}
```

KernelGen has developed a nice [performance test suite](https://github.com/dmikushin/kernelgen-perf-tests) for evaluating CPU/GPU compilers by example of so called stencils. Stencil codes for PDEs often are a function core in numerical simulation.
