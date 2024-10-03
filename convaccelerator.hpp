#ifndef CONVACCELERATOR_HPP
#define CONVACCELERATOR_HPP

// #include <hls_stream.h>
#include <stdint.h>
#include <stdio.h>
// #include <ap_int.h>
// #include <ap_axi_sdata.h>

#define INPUT_SIZE      1300U
#define KERNEL_SIZE     3U
#define OUT_SIZE   (INPUT_SIZE - KERNEL_SIZE + 1)

void CPUConvolution(int *image, int *kernel, int *output);
void MatrixGen(uint32_t* image, uint32_t rows, uint32_t cols);
void PrintMatrix(uint32_t* image, uint32_t rows, uint32_t cols);
void Compare(uint32_t* ACC_OUT, uint32_t* CPU_OUT, uint32_t rows, uint32_t cols, int* errCnt);

#endif