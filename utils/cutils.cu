extern "C" {
	#include "lua.h"
	#include "lualib.h"
	#include "lauxlib.h"
}

#include "luaT.h"
#include "THC.h"

#include <stdio.h>
#include <assert.h>
#include <math_constants.h>
#include <stdint.h>
#include <unistd.h>
#include <curand.h>
#include <curand_kernel.h>
#include <limits.h>

#define TB 128

// PARAMETERS
#define FB_R_VAR 0.01

#define FB_V_INCR 1
#define FB_V_DECR 0.1

#define FB_T_INCR 0.5
#define FB_T_DECR 0.25
#define FB_T_LOWER 2
#define FB_T_UPPER 255

#define UNSTABLE_REG_RATIO_MIN 0.1
#define UNSTABLE_REG_RDIST_MIN 3.0

#define Rd_0 0.4
#define Rc_0 50

#define Rc_ofs Rc_0/5
#define Rd_ofs 0.6

#define max( a, b ) ( ((a) > (b)) ? (a) : (b) )
#define min( a, b ) ( ((a) < (b)) ? (a) : (b) )

THCState* getCutorchState(lua_State* L)
{
	lua_getglobal(L, "cutorch");
	lua_getfield(L, -1, "getState");
	lua_call(L, 0, 1);
	THCState *state = (THCState*) lua_touserdata(L, -1);
	lua_pop(L, 2);
	return state;
}

void checkCudaError(lua_State *L) {
	cudaError_t status = cudaPeekAtLastError();
	if (status != cudaSuccess) {
		luaL_error(L, cudaGetErrorString(status));
	}
}


__global__ void update_model_(float *modelD, float *modelC, float *sampleD, float *sampleC,
	 float *lR, int sizeHW, int sizeFHW, int sizeF, int sizeN, float *FG, float* r_smp, float* r_prb, int trigger){

	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if (id < sizeHW){

		int lr = (FG[id])?FB_T_LOWER:(trigger)?1:ceil(lR[id]);
		int prb = ceil(r_prb[id]);

		if ( prb % lr == 0 ){
			int smp = ceil(r_smp[id]);
			int smp_idx_d = smp * sizeFHW;
			int smp_idx_c = smp * sizeHW;

			// copy color feature
			modelC[smp_idx_c + id] = sampleC[id];

			// copy embedding
			for (int i = 0; i < sizeF; i++){
				modelD[smp_idx_d + i*sizeHW + id] = sampleD[i*sizeHW + id];
			}
		}
	}
}

int update_model(lua_State *L){
  THCState *state = getCutorchState(L);
  THCudaTensor *modelD 			 = (THCudaTensor*)luaT_checkudata(L, 1, "torch.CudaTensor");
	THCudaTensor *modelC 			 = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *sampleD 		 = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
	THCudaTensor *sampleC 		 = (THCudaTensor*)luaT_checkudata(L, 4, "torch.CudaTensor");
	THCudaTensor *learningRate = (THCudaTensor*)luaT_checkudata(L, 5, "torch.CudaTensor");
	THCudaTensor *FG 					 = (THCudaTensor*)luaT_checkudata(L, 6, "torch.CudaTensor");
	THCudaTensor *r_smp 			 = (THCudaTensor*)luaT_checkudata(L, 7, "torch.CudaTensor");
	THCudaTensor *r_prb 			 = (THCudaTensor*)luaT_checkudata(L, 8, "torch.CudaTensor");
	int trigger 							 = luaL_checkinteger(L, 9);

	int sizeHW 	= THCudaTensor_size(state, modelD, 3) * THCudaTensor_size(state, modelD, 4);
  int sizeFHW = THCudaTensor_size(state, modelD, 2) * THCudaTensor_size(state, modelD, 3) * THCudaTensor_size(state, modelD, 4);
	int sizeF 	= THCudaTensor_size(state, modelD, 2);
	int sizeN 	= THCudaTensor_size(state, modelD, 0);

	update_model_<<< (sizeHW - 1)/ TB + 1, TB >>>(
		THCudaTensor_data(state, modelD),
		THCudaTensor_data(state, modelC),
		THCudaTensor_data(state, sampleD),
		THCudaTensor_data(state, sampleC),
		THCudaTensor_data(state, learningRate),
		sizeHW,
		sizeFHW,
		sizeF,
		sizeN,
		THCudaTensor_data(state, FG),
		THCudaTensor_data(state, r_smp),
		THCudaTensor_data(state, r_prb),
		trigger
	);

	checkCudaError(L);
	return 1;
}

__device__ void sort(float *x, int n){
	for (int i = 0; i < n - 1; i++) {
		int min = i;
		for (int j = i + 1; j < n; j++) {
			if (x[j] < x[min]) {
				min = j;
			}
		}
		float tmp = x[min];
		x[min] = x[i];
		x[i] = tmp;
	}
}

__global__ void median2d(float *img, float *out, int size, int dim2, int dim3, int kernel_radius){
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < size) {
		int x = id % dim3;
		int y = id / dim3;

		float xs[11 * 11];
		int xs_size = 0;
		for (int xx = x - kernel_radius; xx <= x + kernel_radius; xx++) {
			for (int yy = y - kernel_radius; yy <= y + kernel_radius; yy++) {
				if (0 <= xx && xx < dim3 && 0 <= yy && yy < dim2) {
					xs[xs_size++] = img[yy * dim3 + xx];
				}
			}
		}
		sort(xs, xs_size);
		out[id] = xs[xs_size / 2];
	}
}

int median2d(lua_State *L) {
	THCState *state = getCutorchState(L);
	THCudaTensor *img = (THCudaTensor*)luaT_checkudata(L, 1, "torch.CudaTensor");
	THCudaTensor *out = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
	int kernel_size = luaL_checkinteger(L, 3);

	assert(kernel_size % 2 == 1);
	assert(kernel_size <= 11);
	median2d<<<(THCudaTensor_nElement(state, out) - 1) / TB + 1, TB>>>(
		THCudaTensor_data(state, img),
		THCudaTensor_data(state, out),
		THCudaTensor_nElement(state, out),
		THCudaTensor_size(state, out, 2),
		THCudaTensor_size(state, out, 3),
		kernel_size / 2);
	checkCudaError(L);
	return 0;
}

__global__ void binary_dilate(float* img, float* out, int size, int dim2, int dim3, int r){
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < size)	{
		int x = id % dim3;
		int y = id / dim3;
		bool flag = 0;

		for (int xx = x - r; xx <= x + r; xx++)	{
			for (int yy = y - r; yy<= y + r; yy++){
				if (0 <= xx && xx < dim3 && 0 <= yy && yy < dim2){
					if (img[yy*dim3 + xx] == 1){
						flag = 1;
						break;
					}
				}
			}

			if (flag) break;
		}
		out[id] = (flag)?1:0;
	}
}

int binary_dilate(lua_State *L){
	THCState *state = getCutorchState(L);

	THCudaTensor *img = (THCudaTensor*)luaT_checkudata(L, 1, "torch.CudaTensor");
	THCudaTensor *out = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
	int kernel_size = luaL_checkinteger(L, 3);
	assert(kernel_size % 2 == 1);
	binary_dilate<<<(THCudaTensor_nElement(state, out) - 1) / TB + 1, TB>>>(
		THCudaTensor_data(state, img),
		THCudaTensor_data(state, out),
		THCudaTensor_nElement(state, out),
		THCudaTensor_size(state, img, 2),
		THCudaTensor_size(state, img, 3),
		kernel_size / 2);
		checkCudaError(L);
		return 1;
}

__global__ void binary_erode_(float* img, float* out, int size, int dim2, int dim3, int r){

	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < size)	{
		int x = id % dim3;
		int y = id / dim3;
		bool flag = 0;

		for (int xx = x - r; xx <= x + r; xx++)	{
			for (int yy = y - r; yy<= y + r; yy++){
				if (0 <= xx && xx < dim3 && 0 <= yy && yy < dim2){
					if (img[yy*dim3 + xx] == 0){
						flag = 1;
						break;
					}
				}
			}

			if (flag) break;
		}
		out[id] = (flag)?0:1;
	}
}

int binary_erode(lua_State *L){
	THCState *state = getCutorchState(L);

	THCudaTensor *img = (THCudaTensor*)luaT_checkudata(L, 1, "torch.CudaTensor");
	THCudaTensor *out = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
	int kernel_size = luaL_checkinteger(L, 3);
	assert(kernel_size % 2 == 1);
	binary_erode_<<<(THCudaTensor_nElement(state, out) - 1) / TB + 1, TB>>>(
		THCudaTensor_data(state, img),
		THCudaTensor_data(state, out),
		THCudaTensor_nElement(state, out),
		THCudaTensor_size(state, img, 2),
		THCudaTensor_size(state, img, 3),
		kernel_size / 2);
		checkCudaError(L);
		return 1;
}

__global__ void update_params_(float a_lt,
												 float a_st,
												 float* R,
												 float* T,
												 float* v,
												 float* D_LT,
												 float* D_ST,
												 float* rSgm_LT,
												 float* rSgm_ST,
												 float* Sgm_LT,
												 float* Sgm_ST,
												 float* US,
												 float* curFG,
												 float* lastFG,
												 float* blink,
												 float	h,
												 float	w,
											 	 float  sizeHW,
											 	 float* d_m){
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < sizeHW)	{
		if (curFG[id])	{
			// Update D_m
			D_LT[id] = D_LT[id] * (1.0 - a_lt) + d_m[id]*a_lt;
			D_ST[id] = D_ST[id] * (1.0 - a_st) + d_m[id]*a_st;

			// Update mean raw segmentation
			rSgm_LT[id] = rSgm_LT[id] * (1.0 - a_lt) + a_lt;
			rSgm_ST[id] = rSgm_ST[id] * (1.0 - a_st) + a_st;
		}else{
			// Update D_m
			D_LT[id] = D_LT[id] * (1.0 - a_lt) + d_m[id]*a_lt;
			D_ST[id] = D_ST[id] * (1.0 - a_st) + d_m[id]*a_st;

			// Update mean raw segmentation
			rSgm_LT[id] = rSgm_LT[id] * (1.0 - a_lt);
			rSgm_ST[id] = rSgm_ST[id] * (1.0 - a_st);
		}

		// Update learning rate T
		if (lastFG[id] || (min(D_LT[id], D_ST[id]) < UNSTABLE_REG_RATIO_MIN) && curFG[id] )	{
			T[id] += FB_T_INCR/(max(D_LT[id], D_ST[id]) * v[id]);
		}else{
			T[id] -= FB_T_DECR * v[id] / (max(D_LT[id], D_ST[id]));
		}

		if (T[id] > FB_T_UPPER)
			T[id] = FB_T_UPPER;
		else if (T[id] < FB_T_LOWER)
			T[id] = FB_T_LOWER;

		// Update v
		if (max(D_LT[id], D_ST[id]) > UNSTABLE_REG_RATIO_MIN && blink[id])
			v[id] += FB_V_INCR;
		else if (v[id] > FB_V_DECR){
			v[id] -= lastFG[id]?FB_V_DECR/4:US[id]?FB_V_DECR/2:FB_V_DECR;
			if (v[id] < FB_V_DECR)
				v[id] = FB_V_DECR;
		}

		// Update R
		float min_D = 2*min(D_LT[id], D_ST[id]) + 1;
		min_D *=min_D;
		if (R[id] < min_D)
			R[id] += FB_R_VAR*(v[id] - FB_V_DECR);
		else{
			R[id] -= FB_R_VAR/(v[id]);
		}
		if (R[id] < 1.0)
			R[id] = 1.0;
	}
}

int update_params(lua_State *L){
	THCState *state = getCutorchState(L);

	float a_lt = luaL_checknumber(L, 1);
	float a_st = luaL_checknumber(L, 2);
	THCudaTensor *R 			= (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
	THCudaTensor *T 			= (THCudaTensor*)luaT_checkudata(L, 4, "torch.CudaTensor");
	THCudaTensor *v 			= (THCudaTensor*)luaT_checkudata(L, 5, "torch.CudaTensor");
	THCudaTensor *D_LT 		= (THCudaTensor*)luaT_checkudata(L, 6, "torch.CudaTensor");
	THCudaTensor *D_ST 		= (THCudaTensor*)luaT_checkudata(L, 7, "torch.CudaTensor");
	THCudaTensor *rSgm_LT = (THCudaTensor*)luaT_checkudata(L, 8, "torch.CudaTensor");
	THCudaTensor *rSgm_ST = (THCudaTensor*)luaT_checkudata(L, 9, "torch.CudaTensor");
	THCudaTensor *Sgm_LT 	= (THCudaTensor*)luaT_checkudata(L, 10, "torch.CudaTensor");
	THCudaTensor *Sgm_ST 	= (THCudaTensor*)luaT_checkudata(L, 11, "torch.CudaTensor");
	THCudaTensor *US		  = (THCudaTensor*)luaT_checkudata(L, 12, "torch.CudaTensor");
	THCudaTensor *curFG	  = (THCudaTensor*)luaT_checkudata(L, 13, "torch.CudaTensor");
	THCudaTensor *lastFG	= (THCudaTensor*)luaT_checkudata(L, 14, "torch.CudaTensor");
	THCudaTensor *blink		= (THCudaTensor*)luaT_checkudata(L, 15, "torch.CudaTensor");
	THCudaTensor *d_m			= (THCudaTensor*)luaT_checkudata(L, 16, "torch.CudaTensor");

	update_params_<<<(THCudaTensor_nElement(state, R) - 1) / TB + 1, TB>>>(
		a_lt,
		a_st,
		THCudaTensor_data(state, R),
		THCudaTensor_data(state, T),
		THCudaTensor_data(state, v),
		THCudaTensor_data(state, D_LT),
		THCudaTensor_data(state, D_ST),
		THCudaTensor_data(state, rSgm_LT),
		THCudaTensor_data(state, rSgm_ST),
		THCudaTensor_data(state, Sgm_LT),
		THCudaTensor_data(state, Sgm_ST),
		THCudaTensor_data(state, US),
		THCudaTensor_data(state, curFG),
		THCudaTensor_data(state, lastFG),
		THCudaTensor_data(state, blink),
		THCudaTensor_size(state, R, 0),			//height
		THCudaTensor_size(state, R, 1),			//weight
		THCudaTensor_nElement(state, R),		//h*w
		THCudaTensor_data(state, d_m)
	);
	checkCudaError(L);
	return 1;
}

__global__ void update_threshold_(float* R, float* R_c, float* R_d, float* US, int size01){
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < size01){
		R_c[id] = ((R[id] * Rc_0) - (!US[id] * Rc_ofs))/2;

		R_d[id] = (R[id] * Rd_0) + (US[id] * Rd_ofs);
	}
}

int update_threshold(lua_State *L){
	THCState *state = getCutorchState(L);

	THCudaTensor *R		 = (THCudaTensor*)luaT_checkudata(L, 1, "torch.CudaTensor");
	THCudaTensor *R_c  = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
	THCudaTensor *R_d  = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
	THCudaTensor *US 	 = (THCudaTensor*)luaT_checkudata(L, 4, "torch.CudaTensor");

	update_threshold_<<<(THCudaTensor_nElement(state, R_c) - 1) / TB + 1, TB>>>(
		THCudaTensor_data(state, R),
		THCudaTensor_data(state, R_c),
		THCudaTensor_data(state, R_d),
		THCudaTensor_data(state, US),
		THCudaTensor_size(state, R_c, 0) * THCudaTensor_size(state, R_c, 1)
	);
	checkCudaError(L);
	return 1;
}

__global__ void check_unstable_(float *US, float *R, float* rSgm_LT, float *rSgm_ST,
														float *Sgm_LT, float *Sgm_ST, int sizeHW){

	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < sizeHW){

		US[id] = ( (R[id] > UNSTABLE_REG_RDIST_MIN) ||
						   ((rSgm_LT[id] - Sgm_LT[id]) > UNSTABLE_REG_RATIO_MIN) ||
						   ((rSgm_ST[id] - Sgm_ST[id]) > UNSTABLE_REG_RATIO_MIN) ) ? 1 : 0;
	}
}

int check_unstable(lua_State *L){
	THCState *state = getCutorchState(L);

	THCudaTensor *US		 	= (THCudaTensor*)luaT_checkudata(L, 1, "torch.CudaTensor");
	THCudaTensor *R  			= (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
	THCudaTensor *rSgm_LT = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
	THCudaTensor *rSgm_ST = (THCudaTensor*)luaT_checkudata(L, 4, "torch.CudaTensor");
	THCudaTensor *Sgm_LT  = (THCudaTensor*)luaT_checkudata(L, 5, "torch.CudaTensor");
	THCudaTensor *Sgm_ST  = (THCudaTensor*)luaT_checkudata(L, 6, "torch.CudaTensor");

	check_unstable_<<<(THCudaTensor_nElement(state, R) - 1) / TB + 1, TB>>>(
		THCudaTensor_data(state, US),
		THCudaTensor_data(state, R),
		THCudaTensor_data(state, rSgm_LT),
		THCudaTensor_data(state, rSgm_ST),
		THCudaTensor_data(state, Sgm_LT),
		THCudaTensor_data(state, Sgm_ST),
		THCudaTensor_size(state, R, 0) * THCudaTensor_size(state, R, 1)
	);
	checkCudaError(L);
	return 1;
}

__global__ void Normalize_get_norm_(float *input, float *norm, int size1, int size23, int size023){
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < size023) {
		int dim23 = id % size23;
		int dim0 = id / size23;

		float sum = 0.0;
		for (int dim1 = 0; dim1 < size1; dim1++) {
			float x = input[(dim0 * size1 + dim1) * size23 + dim23];
			sum += x * x;
		}
		norm[dim0 * size23 + dim23] = sum + 1e-5;
	}
}

__global__ void Normalize_forward_(float *input, float *norm, float *output, int size23, int size123, int size0123){
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < size0123) {
		int dim23 = id % size23;
		int dim0 = (id / size123);
		output[id] = input[id] / sqrtf(norm[dim0 * size23 + dim23]);
	}
}

int Normalize_forward(lua_State *L){
	THCState *state = getCutorchState(L);
	THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 1, "torch.CudaTensor");
	THCudaTensor *norm = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
	THCudaTensor *output = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");

	Normalize_get_norm_<<<(THCudaTensor_nElement(state, norm) - 1) / TB + 1, TB>>>(
		THCudaTensor_data(state, input),
		THCudaTensor_data(state, norm),
		THCudaTensor_size(state, input, 1),
		THCudaTensor_size(state, input, 2) * THCudaTensor_size(state, input, 3),
		THCudaTensor_nElement(state, norm));

	Normalize_forward_<<<(THCudaTensor_nElement(state, output) - 1) / TB + 1, TB>>>(
		THCudaTensor_data(state, input),
		THCudaTensor_data(state, norm),
		THCudaTensor_data(state, output),
		THCudaTensor_size(state, input, 2) * THCudaTensor_size(state, input, 3),
		THCudaTensor_size(state, input, 1) * THCudaTensor_size(state, input, 2) * THCudaTensor_size(state, input, 3),
		THCudaTensor_nElement(state, output));
	checkCudaError(L);
	return 0;
}

__global__ void computeDescDist_(float *input_L, float *input_R, float *output_L,
															int size1_input, int size1, int size3, int size23){

	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < size23) {
		int dim3 = id % size3;
		assert(size1_input <= 128);
		float L_cache[128];
		for (int i = 0; i < size1_input; i++) {
			L_cache[i] = input_L[i * size23 + id];
		}

		if (dim3 >= 0) {
			float sum = 0;
			for (int i = 0; i < size1_input; i++) {
				sum += L_cache[i] * input_R[i * size23 + id];
			}
			output_L[id] = 1 - sum;
		}

	}
}

int computeDescDist(lua_State *L){
	THCState *state = getCutorchState(L);
	THCudaTensor *input_L = (THCudaTensor*)luaT_checkudata(L, 1, "torch.CudaTensor");
	THCudaTensor *input_R = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
	THCudaTensor *output_L = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");

	int size23 = THCudaTensor_size(state, output_L, 2) * THCudaTensor_size(state, output_L, 3);
	computeDescDist_<<<(size23 - 1) / TB + 1, TB>>>(
		THCudaTensor_data(state, input_L),
		THCudaTensor_data(state, input_R),
		THCudaTensor_data(state, output_L),

		THCudaTensor_size(state, input_L, 1),
		THCudaTensor_size(state, output_L, 1),
		THCudaTensor_size(state, output_L, 3),
		size23);
	checkCudaError(L);
	return 0;
}

__global__ void update_seg_(float *Sgm_LT, float *Sgm_ST, float *FG, float a_LT, float a_ST, int sizeHW){

	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < sizeHW) {

		Sgm_LT[id] = Sgm_LT[id] * (1.0 - a_LT) + FG[id] * a_LT;
		Sgm_ST[id] = Sgm_ST[id] * (1.0 - a_ST) + FG[id] * a_ST;
	}
}

int update_seg(lua_State *L){
	THCState *state = getCutorchState(L);
	THCudaTensor *Sgm_LT = (THCudaTensor*)luaT_checkudata(L, 1, "torch.CudaTensor");
	THCudaTensor *Sgm_ST = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
	THCudaTensor *FG = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
	float a_LT = luaL_checknumber(L, 4);
	float a_ST = luaL_checknumber(L, 5);

	int sizeHW = THCudaTensor_size(state, FG, 2) * THCudaTensor_size(state, FG, 3);
	update_seg_<<<(sizeHW - 1) / TB + 1, TB>>>(
		THCudaTensor_data(state, Sgm_LT),
		THCudaTensor_data(state, Sgm_ST),
		THCudaTensor_data(state, FG),
		a_LT,
		a_ST,
		sizeHW);
	checkCudaError(L);
	return 0;
}

static const struct luaL_Reg funcs[] = {
	{"update_model", update_model},
	{"median2d", median2d},
	{"binary_dilate", binary_dilate},
	{"update_params", update_params},
	{"Normalize_forward", Normalize_forward},
	{"computeDescDist", computeDescDist},
	{"update_threshold", update_threshold},
	{"check_unstable", check_unstable},
	{"update_seg", update_seg},
	{"binary_erode", binary_erode},
  {NULL, NULL}
};

extern "C" int luaopen_libcutils(lua_State *L) {
	luaL_openlib(L, "cutils", funcs, 0);
	return 1;
}
