#include "convaccelerator.hpp"
#include <hls_stream.h>
#include <ap_int.h>
#include <iostream>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <hls_vector.h>
#include <hls_stream.h>


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

void MatrixGen(int* image, uint32_t rows, uint32_t cols){
    for (uint32_t i = 0; i < rows; i++){
        for (uint32_t j = 0; j < cols; j++){
            image[i * cols + j] = (rand()%255) +1;
            }
    }
}

void PrintMatrix(int* image, uint32_t rows, uint32_t cols){
    for (uint32_t i = 0; i < rows; i++){
        for (uint32_t j = 0; j < cols; j++){
            printf("%d ",image[i * cols + j]);
            }
        printf("\r\n");
    }
    printf("\r\n");
}

void Compare(int* ACC_OUT, int* CPU_OUT, uint32_t rows, uint32_t cols, int* errCnt){
    int result = 0, expected = 0;

    *errCnt = 0;

    printf("Comparing result's dimension: %u x %u\r\n", rows, cols);

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
        printf("TEST PASS! ALL OUTPUT ARE VERIFIED!\n");
    }
    else{
        printf("TEST FAIL!\n");

    }
}

// 2nd Attempt : DATAFLOW + TILING + VECTORIZE

// Function to compute the convolution for a subimage using streams
void ConvolveStream(
    hls::stream<int> &input_stream,
    int kernel[3][3],
    hls::stream<int> &output_stream)
{

    // Buffer to hold a 3x3 section of the image
    int window[3][TILE_SIZE];
    #pragma HLS ARRAY_PARTITION variable=window complete dim=1

    // Fill the window buffer initially
    fill_window:for (int i = 0; i < 3; i++) {
        for (int j = 0; j < TILE_SIZE; j++) {
            #pragma HLS PIPELINE II=1
            window[i][j] = input_stream.read();
        }
    }

    // Iterate through each pixel in the subimage excluding the border
    iterate_all_pixels:for (int y = 0; y < TILE_SIZE - 2; y++) {
        for (int x = 0; x < TILE_SIZE - 2; x++) {
            #pragma HLS PIPELINE II=1
            int sum = 0;

            // Apply the 3x3 convolution kernel
            compute:for (int ky = 0; ky < 3; ky++) {
                for (int kx = 0; kx < 3; kx++) {
                    #pragma HLS UNROLL
                    sum += window[ky][x + kx] * kernel[ky][kx];
                }
            }

            // Write the result to the output stream
            output_stream.write(sum);
        }
    }
}


extern "C" {
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
    hls::stream<int> &output10_stream)
{
    #pragma HLS INTERFACE axis port=input0_stream
    #pragma HLS INTERFACE axis port=input1_stream
    #pragma HLS INTERFACE axis port=input2_stream
    #pragma HLS INTERFACE axis port=input3_stream
    #pragma HLS INTERFACE axis port=input4_stream
    #pragma HLS INTERFACE axis port=input5_stream
    #pragma HLS INTERFACE axis port=input6_stream
    #pragma HLS INTERFACE axis port=input7_stream
    #pragma HLS INTERFACE axis port=input8_stream
    #pragma HLS INTERFACE axis port=input9_stream
    #pragma HLS INTERFACE axis port=input10_stream

    #pragma HLS INTERFACE axis port=output0_stream
    #pragma HLS INTERFACE axis port=output1_stream
    #pragma HLS INTERFACE axis port=output2_stream
    #pragma HLS INTERFACE axis port=output3_stream
    #pragma HLS INTERFACE axis port=output4_stream
    #pragma HLS INTERFACE axis port=output5_stream
    #pragma HLS INTERFACE axis port=output6_stream
    #pragma HLS INTERFACE axis port=output7_stream
    #pragma HLS INTERFACE axis port=output8_stream
    #pragma HLS INTERFACE axis port=output9_stream
    #pragma HLS INTERFACE axis port=output10_stream

    #pragma HLS INTERFACE s_axilite port=return bundle=control

    // Define the 3x3 convolution kernel
    int kernel[3][3] = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };

    #pragma HLS DATAFLOW

    // Process each stream in parallel
    ConvolveStream(input0_stream, kernel, output0_stream);
    ConvolveStream(input1_stream, kernel, output1_stream);
    ConvolveStream(input2_stream, kernel, output2_stream);
    ConvolveStream(input3_stream, kernel, output3_stream);
    ConvolveStream(input4_stream, kernel, output4_stream);
    ConvolveStream(input5_stream, kernel, output5_stream);
    ConvolveStream(input6_stream, kernel, output6_stream);
    ConvolveStream(input7_stream, kernel, output7_stream);
    ConvolveStream(input8_stream, kernel, output8_stream);
    ConvolveStream(input9_stream, kernel, output9_stream);
    ConvolveStream(input10_stream, kernel, output10_stream);
}
}

