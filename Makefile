run: cudatest
	./cudatest

cudatest: cudatest.cpp
	g++ -o cudatest -O2  cudatest.cpp

checkcuda:
	lspci | grep -i nvidia
	cat /proc/driver/nvidia/version

clean:
	rm -f *.o
