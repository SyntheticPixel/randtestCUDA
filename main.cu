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

	/*
	float a = curand_uniform(s);
	float b = curand_uniform(s);

	//swap if b < a
	if (b < a){
		float c;
		c = a;
		a = b;
		b = c;
	}

	// this boils down to unit radius
	p.x = a*cos(2 * M_PI * (a / b));
	p.y = b*sin(2 * M_PI * (a / b));
	p.z = 0.0f;

	return p;
	*/


	float t = curand_uniform(s) *2 * M_PI;
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

__device__ void sample(glm::vec3 *s, unsigned long id, unsigned int seed){

	glm::vec3 t;

	curandState rngState;
	curand_init(WangHash(seed)+id, 0, 0, &rngState);

	// function to test
	//t = RandUniformSquare(&rngState);
	t = RandUniformDisc(&rngState);
	//t = RandCosineHemisphere(&rngState, glm::vec3(0.0f, 1.0f, 0.0f));
	//t = RandUniformSphere(&rngState, 1.0f);

	// copy back
	s->x = t.x;
	s->y = t.y;
	s->z = t.z;
}

__global__ void KERNEL_SAMPLE(glm::vec3 *v, int samples, unsigned int seed){

	unsigned long blockId = blockIdx.x + blockIdx.y * gridDim.x;
	unsigned long threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	if (threadId >= samples) return;

	// generate sample
	sample(&v[threadId], threadId, seed);
}

// main
int main(){
	glm::vec3	*host_vectors;
	glm::vec3	*device_vectors;
	size_t		size;

	std::string input="";
	int x_samples = 1;
	int y_samples = 1;

	std::cout << "Enter the number of  X samples:" << std::endl;
	cin >> x_samples;
	if(x_samples < 1) x_samples = 1;

	std::cout << "Enter the number of  Y samples:" << std::endl;
	cin >> y_samples;
	if(y_samples < 1) y_samples = 1;

	// calculate memory size
	size = sizeof(glm::vec3)*x_samples*y_samples;

	// assign on host
	cudaHostAlloc((void**)&host_vectors, size, cudaHostAllocWriteCombined);

	// assign on device
	cudaMalloc((void**)&device_vectors, size);

	// sample
	dim3 blockSize = dim3(16, 16, 1);	// 256 threads
	dim3 gridSize = dim3((x_samples + blockSize.x - 1) / blockSize.x, (y_samples + blockSize.y - 1) / blockSize.y, 1);
	KERNEL_SAMPLE<<<gridSize, blockSize>>>(device_vectors, x_samples*y_samples, time(NULL));
	cudaDeviceSynchronize();

	// copy from device to host
	cudaMemcpy(host_vectors, device_vectors, size, cudaMemcpyDeviceToHost);

	// open the file
	ofstream ascfile;
	ascfile.open("out.txt");

	// write out the file
	for(int i=0; i < x_samples*y_samples; i++){
		glm::vec3 v = host_vectors[i];
		// print to file
		ascfile << v.x << " " << v.y << " " << v.z << "\n";

	}

	// close the file
	ascfile.close();

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
