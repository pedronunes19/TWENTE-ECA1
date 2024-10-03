#include "convaccelerator.hpp"
//#include <hls_stream.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
//#include <ap_int.h>
//#include <ap_axi_sdata.h>
#include <stdint.h>

void CPUConvolution(int *image, int *kernel, int *output) {
    // Iterate over the output dimensions (1298x1298)
    for (int y = 0; y < OUT_SIZE; y++) {
        for (int x = 0; x < OUT_SIZE; x++) {
            // Initialize the result value for this position
            output[y * OUT_SIZE + x] = 0;
            // Iterate over the kernel (3x3)
            for (int ky = 0; ky < KERNEL_SIZE; ky++) {
                for (int kx = 0; kx < KERNEL_SIZE; kx++) {
                    // Compute the indices in the 1D arrays for both image and kernel
                    int image_index = (y + ky) * INPUT_SIZE + (x + kx);
                    int kernel_index = ky * KERNEL_SIZE + kx;

                    // Perform the convolution operation
                    output[y * OUT_SIZE + x] += image[image_index] * kernel[kernel_index];
                }
            }
        }
    }
}

void MatrixGen(uint32_t* image, uint32_t rows, uint32_t cols){
    for (uint32_t i = 0; i < rows; i++){
        for (uint32_t j = 0; j < cols; j++){
            image[i * cols + j] = (rand()%255) +1;
            }
    }
}

void PrintMatrix(uint32_t* image, uint32_t rows, uint32_t cols){
    for (uint32_t i = 0; i < rows; i++){
        for (uint32_t j = 0; j < cols; j++){
            printf("%d ",image[i * cols + j]);
            }
        printf("\r\n");
    }
    printf("\r\n");
}

void Compare(uint32_t* ACC_OUT, uint32_t* CPU_OUT, uint32_t rows, uint32_t cols, int* errCnt){
    uint32_t result = 0, expected = 0;

    *errCnt = 0;

    printf("Comparing for dims: %u x %u\r\n", rows, cols);

        for (uint32_t i = 0; i < rows; i++) {
        for (uint32_t j = 0; j < cols; j++) {

            result = ACC_OUT[i*cols + j];
            expected = CPU_OUT[i*cols + j];

            if (result != expected) {

                if (*errCnt < 10) {
                    printf("%u, %u != %u\r\n", i*cols+j, result, expected);
                }
                *errCnt += 1;
            }
        }
    }

    if (*errCnt == 0)
    {
        printf("TEST PASS! ALL OUTPUT ARE VERIFIED!");
    }
    else{
        printf("TEST FAIL!");

    }
}
