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

void CPUConvolution(int *image, int *kernel, int *output);
void MatrixGen(int* image, uint32_t rows, uint32_t cols);
void PrintMatrix(int* image, uint32_t rows, uint32_t cols);
void Compare(int* ACC_OUT, int* CPU_OUT, uint32_t rows, uint32_t cols, int* errCnt);

extern"C"{
void convolve_all(
    hls::stream<int> &input0_stream,
    hls::stream<int> &input1_stream,
    hls::stream<int> &input2_stream,
    hls::stream<int> &input3_stream,
    hls::stream<int> &input4_stream,
    hls::stream<int> &input5_stream,
    hls::stream<int> &input6_stream,
    hls::stream<int> &input7_stream,
    hls::stream<int> &input8_stream,
    hls::stream<int> &input9_stream,
    hls::stream<int> &input10_stream,
    hls::stream<int> &output0_stream,
    hls::stream<int> &output1_stream,
    hls::stream<int> &output2_stream,
    hls::stream<int> &output3_stream,
    hls::stream<int> &output4_stream,
    hls::stream<int> &output5_stream,
    hls::stream<int> &output6_stream,
    hls::stream<int> &output7_stream,
    hls::stream<int> &output8_stream,
    hls::stream<int> &output9_stream,
    hls::stream<int> &output10_stream);}

#endif
