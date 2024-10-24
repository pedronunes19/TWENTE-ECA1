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

void Compare(int* ACC_OUT, int* CPU_OUT, uint32_t rows, uint32_t cols, int* errCnt) {
    int result = 0, expected = 0;

    *errCnt = 0;

    printf("Comparing result's dimension: %u x %u\r\n", rows, cols);

    for (uint32_t i = 0; i < rows; i++) {
        for (uint32_t j = 0; j < cols; j++) {
            result = ACC_OUT[i * cols + j];
            expected = CPU_OUT[i * cols + j];

            if (result != expected) {
                if (*errCnt < 10) {
                    printf("Mismatch at index (%u, %u) -> ACC_OUT: %d, CPU_OUT: %d\n", i, j, result, expected);
                }
                (*errCnt)++;
            }
        }
    }

    if (*errCnt == 0) {
        printf("TEST PASS! ALL OUTPUT ARE VERIFIED!\n");
    } else {
        printf("TEST FAIL! Number of mismatches: %d\n", *errCnt);
    }
}


// 1st Attempt : Stream full image, depth 1300x1300 = 1690000

////// Define the window structure to hold a 3x3 pixel neighborhood
//struct window {
//    int pix[KERNEL_SIZE][KERNEL_SIZE];
//};
//
//// Function to read image data from memory into a stream
//void MemRead(
//    const int *image,
//    hls::stream<int> &pixel_stream)
//{
//    read_image: for (int n = 0; n < INPUT_SIZE * INPUT_SIZE; n++) {
//        #pragma HLS PIPELINE II=1
//        int pix = image[n];
//        pixel_stream.write(pix);
//
//        // Debug: Print first 10 pixels read
//        if (n < 10) {
//            printf("MemRead: pixel[%d] = %d\n", n, pix);
//        }
//    }
//}
//
//// Function to write image data from a stream back to memory
//void MemWrite(
//    hls::stream<int> &pixel_stream,
//    int *output)
//{
//    write_image: for (int i = 0; i < OUT_SIZE; i++) {
//        for (int j = 0; j < OUT_SIZE; j++) {
//            #pragma HLS PIPELINE II=1
//            if (!pixel_stream.empty()) {
//                int pix = pixel_stream.read();
//                output[i * OUT_SIZE + j] = pix;
//
//                // Debug: Print first 10 output values written
//                if (i * OUT_SIZE + j < 10) {
//                    printf("MemWrite: output[%d] = %d\n", i * OUT_SIZE + j, pix);
//                }
//            }
//        }
//    }
//}
//
//// Function to generate a sliding window over the input image
//void WindowGenerator(
//    hls::stream<int> &pixel_stream,
//    hls::stream<window> &window_stream)
//{
//    // Line buffers for 2 lines (since KERNEL_SIZE is 3)
//    int LineBuffer[KERNEL_SIZE - 1][INPUT_SIZE];
//#pragma HLS ARRAY_PARTITION variable=LineBuffer dim=1 complete
//
//    // Sliding window
//    window Window;
//
//    // Initialize line buffers to zero
//    for (int i = 0; i < KERNEL_SIZE - 1; i++) {
//        for (int j = 0; j < INPUT_SIZE; j++) {
//            LineBuffer[i][j] = 0;
//        }
//    }
//
//    // Initialize the window to zero to avoid leftover values
//    for (int i = 0; i < KERNEL_SIZE; i++) {
//        for (int j = 0; j < KERNEL_SIZE; j++) {
//            Window.pix[i][j] = 0;
//        }
//    }
//
//    unsigned col_ptr = 0;
//
//    // Iterate through the image to fill the window and line buffer
//    update_window: for (int y = 0; y < INPUT_SIZE; y++) {
//        for (int x = 0; x < INPUT_SIZE; x++) {
//            #pragma HLS PIPELINE II=1
//
//            int new_pixel = pixel_stream.read();
//
//            // Shift window horizontally
//            for (int i = 0; i < KERNEL_SIZE; i++) {
//                for (int j = 0; j < KERNEL_SIZE - 1; j++) {
//                    Window.pix[i][j] = Window.pix[i][j + 1];
//                }
//                Window.pix[i][KERNEL_SIZE - 1] = (i < KERNEL_SIZE - 1) ? LineBuffer[i][col_ptr] : new_pixel;
//            }
//
//            // Update line buffers after reading new pixel
//            if (y < INPUT_SIZE - 1) {
//                LineBuffer[0][col_ptr] = LineBuffer[1][col_ptr];
//                LineBuffer[1][col_ptr] = new_pixel;
//            }
//
//            // Output the window only if it contains valid data (no padding)
//            if (y >= KERNEL_SIZE - 1 && x >= KERNEL_SIZE - 1) {
//                window_stream.write(Window);
//
//                // Debug: Print the first few windows to verify correctness
//                if (y == KERNEL_SIZE - 1 && x < 10) {
//                    printf("WindowGenerator: Window at [%d, %d]:\n", y, x);
//                    for (int i = 0; i < KERNEL_SIZE; i++) {
//                        for (int j = 0; j < KERNEL_SIZE; j++) {
//                            printf("%d ", Window.pix[i][j]);
//                        }
//                        printf("\n");
//                    }
//                }
//            }
//
//            // Update column pointer
//            col_ptr = (col_ptr + 1) % INPUT_SIZE;
//        }
//    }
//}
//
//// Function to apply a convolution filter using a sliding window
//void ConvolutionFilter(
//    hls::stream<window> &window_stream,
//    hls::stream<int> &pixel_stream)
//{
//    int coeffs[KERNEL_SIZE][KERNEL_SIZE] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
//    #pragma HLS ARRAY_PARTITION variable=coeffs complete dim=0
//
//    apply_filter: for (int y = 0; y < OUT_SIZE; y++) {
//        for (int x = 0; x < OUT_SIZE; x++) {
//            #pragma HLS PIPELINE II=1
//
//            int sum = 0;
//
//            // Read the window
//            window w = window_stream.read();
//
//            // Apply convolution
//            for (int row = 0; row < KERNEL_SIZE; row++) {
//                for (int col = 0; col < KERNEL_SIZE; col++) {
//                    sum += w.pix[row][col] * coeffs[row][col];
//                }
//            }
//
//            pixel_stream.write(sum);
//
//            // Debug: Print the first few convolution results
//            if (y == 0 && x < 10) {
//                printf("ConvolutionFilter: Output at [%d, %d] = %d\n", y, x, sum);
//            }
//        }
//    }
//}
//
//// Top-level function
//extern "C" {
//void FixedConvolution(
//    int *image,
//    int *output)
//{
//#pragma HLS INTERFACE m_axi port=image offset=slave bundle=gmem0 depth=INPUT_SIZE*INPUT_SIZE
//#pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem0 depth=OUT_SIZE*OUT_SIZE
////#pragma HLS INTERFACE s_axilite port=return bundle=control
//
//    #pragma HLS DATAFLOW
//
//    hls::stream<int, 64> pixel_stream;
//    hls::stream<window, 64> window_stream;
//    hls::stream<int, 64> output_stream;
//
//    // Call the functions
//    MemRead(image, pixel_stream);
//    WindowGenerator(pixel_stream, window_stream);
//    ConvolutionFilter(window_stream, output_stream);
//    MemWrite(output_stream, output);
//}
//}

// 2nd Attempt : DATAFLOW + TILING + VECTORIZE

////Define the window structure to hold a 3x3 pixel neighborhood
//struct window {
//    int pix[KERNEL_SIZE][KERNEL_SIZE];
//};
//
//// Function to read image data from memory into a stream for each subimage
//void MemRead(
//    const int *subimage,
//    hls::stream<int> &pixel_stream)
//{
//    read_image: for (int n = 0; n < TILE_SIZE * TILE_SIZE; n++) {
//        #pragma HLS PIPELINE II=1
//        int pix = subimage[n];
//        pixel_stream.write(pix);
//
//        // Debug: Print first 10 pixels read
////        if (n < 10) {
////            printf("MemRead: pixel[%d] = %d\n", n, pix);
////        }
//    }
//}
//
//// Function to write image data from a stream back to memory for each subimage
//void MemWrite(
//    hls::stream<int> &pixel_stream,
//    int *suboutput)
//{
//    write_image: for (int i = 0; i < TILE_H-2; i++) {
//        for (int j = 0; j < TILE_W-2; j++) {
//            #pragma HLS PIPELINE II=1
//            if (!pixel_stream.empty()) {
//                int pix = pixel_stream.read();
//                suboutput[i * (TILE_SIZE-2) + j] = pix;
//
//                // Debug: Print first 10 output values written
////                if (i * (TILE_SIZE-2) + j < 10) {
////                    printf("MemWrite: output[%d] = %d\n", i * (TILE_SIZE-2) + j, pix);
////                }
//            }
//        }
//    }
//}
//
//// Function to generate a sliding window over the input image
//void WindowGenerator(
//    hls::stream<int> &pixel_stream,
//    hls::stream<window> &window_stream)
//{
//    // Line buffers for 2 lines (since KERNEL_SIZE is 3)
//    int LineBuffer[KERNEL_SIZE - 1][TILE_SIZE];
//#pragma HLS ARRAY_PARTITION variable=LineBuffer dim=1 complete
//
//    // Sliding window
//    window Window;
//
//    // Initialize line buffers to zero
//    for (int i = 0; i < KERNEL_SIZE - 1; i++) {
//        for (int j = 0; j < TILE_SIZE; j++) {
//            LineBuffer[i][j] = 0;
//        }
//    }
//
//    // Initialize the window to zero to avoid leftover values
//    for (int i = 0; i < KERNEL_SIZE; i++) {
//        for (int j = 0; j < KERNEL_SIZE; j++) {
//            Window.pix[i][j] = 0;
//        }
//    }
//
//    unsigned col_ptr = 0;
//
//    // Iterate through the image to fill the window and line buffer
//    update_window: for (int y = 0; y < TILE_SIZE; y++) {
//        for (int x = 0; x < TILE_SIZE; x++) {
//            #pragma HLS PIPELINE II=1
//
//            int new_pixel = pixel_stream.read();
//
//            // Shift window horizontally
//            for (int i = 0; i < KERNEL_SIZE; i++) {
//                for (int j = 0; j < KERNEL_SIZE - 1; j++) {
//                    Window.pix[i][j] = Window.pix[i][j + 1];
//                }
//                Window.pix[i][KERNEL_SIZE - 1] = (i < KERNEL_SIZE - 1) ? LineBuffer[i][col_ptr] : new_pixel;
//            }
//
//            // Update line buffers after reading new pixel
//            if (y < TILE_SIZE - 1) {
//                LineBuffer[0][col_ptr] = LineBuffer[1][col_ptr];
//                LineBuffer[1][col_ptr] = new_pixel;
//            }
//
//            // Output the window only if it contains valid data (no padding)
//            if (y >= KERNEL_SIZE - 1 && x >= KERNEL_SIZE - 1) {
//                window_stream.write(Window);
//
////                // Debug: Print the first few windows to verify correctness
////                if (y == KERNEL_SIZE - 1 && x < 10) {
////                    printf("WindowGenerator: Window at [%d, %d]:\n", y, x);
////                    for (int i = 0; i < KERNEL_SIZE; i++) {
////                        for (int j = 0; j < KERNEL_SIZE; j++) {
////                            printf("%d ", Window.pix[i][j]);
////                        }
////                        printf("\n");
////                    }
////                }
//            }
//
//            // Update column pointer
//            col_ptr = (col_ptr + 1) % TILE_SIZE;
//        }
//    }
//}
//
//// Function to apply a convolution filter using a sliding window
//void ConvolutionFilter(
//    hls::stream<window> &window_stream,
//    hls::stream<int> &pixel_stream)
//{
//    int coeffs[KERNEL_SIZE][KERNEL_SIZE] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
//    #pragma HLS ARRAY_PARTITION variable=coeffs complete dim=0
//
//    apply_filter: for (int y = 0; y < TILE_SIZE-2; y++) {
//        for (int x = 0; x < TILE_SIZE-2; x++) {
//            #pragma HLS PIPELINE II=1
//
//            int sum = 0;
//
//            // Read the window
//            window w = window_stream.read();
//
//            // Apply convolution
//            for (int row = 0; row < KERNEL_SIZE; row++) {
//                for (int col = 0; col < KERNEL_SIZE; col++) {
//                    sum += w.pix[row][col] * coeffs[row][col];
//                }
//            }
//
//            pixel_stream.write(sum);
//
////            // Debug: Print the first few convolution results
////            if (y == 0 && x < 10) {
////                printf("ConvolutionFilter: Output at [%d, %d] = %d\n", y, x, sum);
////            }
//        }
//    }
//}
//// Top-level function for multiple subimages
//extern "C" {
//void convolve_all(
//    int *subimage0, int *subimage1, int *subimage2,
//    int *subimage3, int *subimage4,
//	int *subimage5,
//    int *output0, int *output1, int *output2, int *output3,
//    int *output4, int *output5)
//{
//#pragma HLS INTERFACE m_axi port=subimage0 offset=slave bundle=gmem0 depth=TILE_SIZE*TILE_SIZE
//#pragma HLS INTERFACE m_axi port=subimage1 offset=slave bundle=gmem1 depth=TILE_SIZE*TILE_SIZE
//#pragma HLS INTERFACE m_axi port=subimage2 offset=slave bundle=gmem2 depth=TILE_SIZE*TILE_SIZE
//#pragma HLS INTERFACE m_axi port=subimage3 offset=slave bundle=gmem3 depth=TILE_SIZE*TILE_SIZE
//#pragma HLS INTERFACE m_axi port=subimage4 offset=slave bundle=gmem4 depth=TILE_SIZE*TILE_SIZE
//#pragma HLS INTERFACE m_axi port=subimage5 offset=slave bundle=gmem5 depth=TILE_SIZE*TILE_SIZE
//
//#pragma HLS INTERFACE m_axi port=output0 offset=slave bundle=gmem0 depth=(TILE_SIZE - 2)*(TILE_SIZE - 2)
//#pragma HLS INTERFACE m_axi port=output1 offset=slave bundle=gmem1 depth=(TILE_SIZE - 2)*(TILE_SIZE - 2)
//#pragma HLS INTERFACE m_axi port=output2 offset=slave bundle=gmem2 depth=(TILE_SIZE - 2)*(TILE_SIZE - 2)
//#pragma HLS INTERFACE m_axi port=output3 offset=slave bundle=gmem3 depth=(TILE_SIZE - 2)*(TILE_SIZE - 2)
//#pragma HLS INTERFACE m_axi port=output4 offset=slave bundle=gmem4 depth=(TILE_SIZE - 2)*(TILE_SIZE - 2)
//#pragma HLS INTERFACE m_axi port=output5 offset=slave bundle=gmem5 depth=(TILE_SIZE - 2)*(TILE_SIZE - 2)
//
//    #pragma HLS DATAFLOW
//
//    hls::stream<int, 64> pixel_stream[11];
//    hls::stream<window, 64> window_stream[11];
//    hls::stream<int, 64> output_stream[11];
//
//    // Read and process each subimage
//    MemRead(subimage0, pixel_stream[0]);
//    WindowGenerator(pixel_stream[0], window_stream[0]);
//    ConvolutionFilter(window_stream[0], output_stream[0]);
//    MemWrite(output_stream[0], output0);
//
//    MemRead(subimage1, pixel_stream[1]);
//    WindowGenerator(pixel_stream[1], window_stream[1]);
//    ConvolutionFilter(window_stream[1], output_stream[1]);
//    MemWrite(output_stream[1], output1);
//
//    MemRead(subimage2, pixel_stream[2]);
//    WindowGenerator(pixel_stream[2], window_stream[2]);
//    ConvolutionFilter(window_stream[2], output_stream[2]);
//    MemWrite(output_stream[2], output2);
//
//    MemRead(subimage3, pixel_stream[3]);
//	WindowGenerator(pixel_stream[3], window_stream[3]);
//	ConvolutionFilter(window_stream[3], output_stream[3]);
//	MemWrite(output_stream[3], output3);
//
//	MemRead(subimage4, pixel_stream[4]);
//	WindowGenerator(pixel_stream[4], window_stream[4]);
//	ConvolutionFilter(window_stream[4], output_stream[4]);
//	MemWrite(output_stream[4], output4);
//
//	MemRead(subimage5, pixel_stream[5]);
//	WindowGenerator(pixel_stream[5], window_stream[5]);
//	ConvolutionFilter(window_stream[5], output_stream[5]);
//	MemWrite(output_stream[5], output5);
//
//}
//}



///#### 3rd attempt using wider data bits (128 bit)
// // Function to read image data from memory into a stream for each subimage
// void MemRead(
//     const uintX_t *subimage,  // Input data pointer (128-bit packed data)
//     hls::stream<ap_int<32>> &pixel_stream,  // Output stream of 32-bit integers
//     int vSize  // Number of 128-bit chunks to read (TILE_H * TILE_W / 4)
// ) {
//     #pragma HLS INTERFACE m_axi port=subimage offset=slave bundle=gmem depth=1024 max_read_burst_length=4 num_read_outstanding=2 max_widen_bitwidth=128
//     #pragma HLS INTERFACE s_axilite port=subimage bundle=control
//     #pragma HLS INTERFACE s_axilite port=vSize bundle=control
//     #pragma HLS INTERFACE s_axilite port=return bundle=control
//     #pragma HLS INTERFACE axis port=pixel_stream

// 	vSize = TILE_H*TILE_W/4;

//     // Read the entire subimage block into the stream in a continuous fashion.
//     read_image: for (int i = 0; i < vSize; i++) {
// 	#pragma HLS PIPELINE II=1
//         // Read 128 bits (4 integers) at a time
//         uintX_t data = subimage[i];

//         pixel_stream.write(data);
//     }
// }
