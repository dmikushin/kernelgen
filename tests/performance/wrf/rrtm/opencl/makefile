all: module_ra_rrtm.o module_ra_rrtmg_lw.o module_ra_rrtmg_sw.o

WRFFLAGS = -m32

COMP = kgen-gfortran -Wk,--keep -Wk,--opencl-compiler=kgen-opencl-embed -Wk,--kernel-target=opencl $(WRFFLAGS)

module_model_constants.o: module_model_constants.f90
	$(COMP) -c $< -o $@

module_wrf_error.o: module_wrf_error.f90
	$(COMP) -c $< -o $@

module_ra_rrtm.o: module_ra_rrtm.f90 module_model_constants.o module_wrf_error.o
	$(COMP) -c $< -o $@

module_ra_rrtmg_lw.o: module_ra_rrtmg_lw.f90 module_model_constants.o module_wrf_error.o
	$(COMP) -c $< -o $@

module_ra_rrtmg_sw.o: module_ra_rrtmg_sw.f90 module_model_constants.o module_wrf_error.o
	$(COMP) -c $< -o $@

clean:
	rm -rf *.o *.mod *.device.* *.host.* *.ir.* *.ir

