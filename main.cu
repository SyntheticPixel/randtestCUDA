#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#define GLM_FORCE_CUDA
#include <glm.hpp>

using namespace std;
using namespace glm;

/*
@brief
Wanghash for seeds
*/
__host__ __device__ unsigned int WangHash(unsigned int a) {
	a = (a ^ 61) ^ (a >> 16);
	a = a + (a << 3);
	a = a ^ (a >> 4);
	a = a * 0x27d4eb2d;
	a = a ^ (a >> 15);
	return a;
}

/*
Generate a uniform distribution within a a 1D range of [0, 1)
*/
__device__ float RandUniform1D(curandState *s){
	return 1.0f - curand_uniform(s);
}

/*
Generate a uniform distribution within a unit square
*/
__device__ glm::vec3 RandUniformSquare(curandState *s){
	glm::vec3 v;
	v.x = 1.0f - curand_uniform(s);
	v.y = 1.0f - curand_uniform(s);
	v.z = 0.0f;
	return v;
}

/*
Generate a uniform distribution on a unit disc
*/
__device__ glm::vec3 RandUniformDisc(curandState *s){
	glm::vec3 p;

	float t = curand_uniform(s) * 2 * M_PI;
	float r = 2 * curand_uniform(s);
	if(r > 2){r -= 2.0f;}

	p.x = cos(t)*r;
	p.y = sin(t)*r;
	p.z = 0.0f;

	return p;

}

/*
Given a normal, generate a cosine weighted distribution on a unit hemisphere
*/
__device__ glm::vec3 RandCosineHemisphere(curandState *s, glm::vec3 normal){
	glm::vec3 p;

	float x = curand_uniform(s) * 2 * M_PI;
	float y = curand_uniform(s);

	vec3 t = abs(normal.x) > 0.1 ? vec3(-normal.z, 0, normal.x) : vec3(0, -normal.z, normal.y);
	vec3 u = glm::normalize(t);
	vec3 v = glm::cross(normal, u);

	p = (u*cos(x) + v*sin(x))*sqrt(y) + normal*sqrt(1-y);

	return p;
}

/*
Given a radius, generate a unform distribution over a sphere
*/
__device__ glm::vec3 RandUniformSphere(curandState *s, float radius){
	glm::vec3 p;

	float u = curand_uniform(s);
	float v = curand_uniform(s);

	float theta = u * 2.0f * M_PI;
	float phi = acos(2.0f * v - 1.0f);

	p.x = radius * sin(phi) * cos(theta);
	p.y = radius * sin(phi) * sin(theta);
	p.z = radius * cos(phi);

	return p;
}

__device__ void sample(int option, glm::vec3 *s, unsigned int id, unsigned int seed){

	glm::vec3 t;

	curandState rngState;
	curand_init((unsigned long long)WangHash(seed)+id, 0, 0, &rngState);

	switch(option){
	case 0:
		t = RandUniformSquare(&rngState);
		break;
	case 1:
		t = RandUniformDisc(&rngState);
		break;
	case 2:
		// normal facing the positive z axis, for testing purposes only
		t = RandCosineHemisphere(&rngState, glm::vec3(0.0f, 0.0f, 1.0f));
		break;
	case 3:
		t = RandUniformSphere(&rngState, 1.0f);
		break;
	default:
		t = RandUniformSquare(&rngState);
		break;
	}

	// copy back
	s->x = t.x;
	s->y = t.y;
	s->z = t.z;
}

__global__ void KERNEL_SAMPLE(int option, glm::vec3 *v, int samples, unsigned int seed){

	unsigned int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	unsigned int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	if (threadId == samples) return;

	// generate sample
	sample(option, &v[threadId], threadId, seed);
}

// main
int main(int argc, const char * argv[]){
	glm::vec3	*host_vectors;
	glm::vec3	*device_vectors;
	size_t		size;
	ofstream	txtfile;

	int x_samples = 1;
	int y_samples = 1;
	int option = 0;

	// make sure we have exactly 4 arguments
	if(argc == 4){
		x_samples = atoi(argv[1]);
		y_samples = atoi(argv[2]);
		option = atoi(argv[3]);
	}else{
		cout << " ./randtestCUDA -samplesX -samplesY -option (0=square, 1=disc, 2=hemisphere, 3=sphere)" << std::endl;
		return EXIT_FAILURE;
	}

	// calculate memory size
	unsigned int total_samples = x_samples*y_samples;
	size = sizeof(glm::vec3)*total_samples;

	// assign on host
	cudaHostAlloc((void**)&host_vectors, size, cudaHostAllocWriteCombined);

	// assign on device
	cudaMalloc((void**)&device_vectors, size);

	// sample
	dim3 blockSize = dim3(16, 16, 1);	// 256 threads
	dim3 gridSize = dim3((x_samples + blockSize.x - 1) / blockSize.x, (y_samples + blockSize.y - 1) / blockSize.y, 1);
	KERNEL_SAMPLE<<<gridSize, blockSize>>>(option, device_vectors, total_samples, (unsigned int)time(NULL));
	cudaDeviceSynchronize();

	// copy from device to host
	cudaMemcpy(host_vectors, device_vectors, size, cudaMemcpyDeviceToHost);

	// open the file
	txtfile.open("out.txt");

	// write out the file
	for(int i=0; i < total_samples; i++){
		glm::vec3 v = host_vectors[i];
		// print to file
		txtfile << v.x << ";" << v.y << ";" << v.z << "\n";
	}

	// close the file
	txtfile.close();
	cout << "File generated: out.txt" << std::endl;

	// delete device memory
	if( device_vectors != NULL ){
		cudaFree(device_vectors);
	}
	// delete host memory
	if( host_vectors != NULL ){
		cudaFreeHost(host_vectors);
	}

	return 0;
}
