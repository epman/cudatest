CUDA_HOME=/usr/local/cuda-6.0
CXX=$(CUDA_HOME)/bin/nvcc

run: cudatest
	./cudatest

cudatest: cudatest.cu
	$(CXX) -o cudatest -O0  cudatest.cu

checkcuda:
	lspci | grep -i nvidia
	cat /proc/driver/nvidia/version

clean:
	rm -f *.o
