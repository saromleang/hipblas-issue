SHELL:=/bin/bash

.PHONY : clean

binaries=hip_dgemm.cpp.x hip_dgemm.f03.x

all : clean $(binaries)

hip_dgemm.cpp.x : hip_dgemm.cpp
	module load rocm/5.4.0
	hipcc -L/opt/rocm-5.4.0/lib -lhipblas hip_dgemm.cpp -o hip_dgemm.cpp.x

hip_dgemm.f03.x : hip_dgemm.f03
	module load cpe-22.12 cce/15.0.0 craype-accel-amd-gfx90a rocm/5.4.0
	module use /lustre/orion/world-shared/chm135/hipfort/crusher/cpe-22.12/modulefiles/cce/15.0/rocm5.4
	module load hipfort
	hipfc -M878 -lhipblas hip_dgemm.f03 -o hip_dgemm.f03.x

clean : 
	rm -f $(binaries)
