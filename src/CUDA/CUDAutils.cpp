/*
 * CUDAutils.cpp
 *
 *  Created on: 2012
 *      Author: sk
 */

#include "ShapeApp.h"

void sync() {
	cudaDeviceSynchronize();
}

void copyToDevice(void *to, void *from, size_t size, bool do_sync) {
	cutilSafeCall(cudaMemcpy(to, from, size, cudaMemcpyHostToDevice));
	if (do_sync) sync();
}
void copyFromDevice(void *to, void *from, size_t size, bool do_sync) {
	cutilSafeCall(cudaMemcpy(to, from, size, cudaMemcpyDeviceToHost));
	if (do_sync) sync();
}
