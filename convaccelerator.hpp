#ifndef _CONVACCELERATOR_H_
#define _CONVACCELERATOR_H_

#include <hls_stream.h>
#include <ap_int.h>
#include <hls_vector.h>
#include <stdint.h>
#include <hls_vector.h>


#define INPUT_SIZE      1300U
#define KERNEL_SIZE     3U
#define OUT_SIZE   		(INPUT_SIZE - KERNEL_SIZE + 1)
#define TILE_SIZE		120
#define VECTOR_SIZE		4
#define TILE_H			120
#define TILE_W			120

// Define a custom data width (e.g., 128 bits)
#define CUSTOM_DATAWIDTH 128
typedef ap_uint<CUSTOM_DATAWIDTH> uintX_t;

void CPUConvolution(int *image, int *kernel, int *output);
void MatrixGen(int* image, uint32_t rows, uint32_t cols);
void PrintMatrix(int* image, uint32_t rows, uint32_t cols);
void Compare(int* ACC_OUT, int* CPU_OUT, uint32_t rows, uint32_t cols, int* errCnt);

extern"C"{
void convolve_all(
    int *subimage0, int *subimage1, int *subimage2,
    int *subimage3, int *subimage4	, int *subimage5
    ,int *output0, int *output1, int *output2, int *output3,
    int *output4, int *output5);


void FixedConvolution(
        int *image,
        int *output);}

#endif
