#include "convaccelerator.hpp"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <hls_stream.h>

// Function to copy a chunk (tile) from the main image
void CopyToTile(int* image, int* tile, uint32_t rows, uint32_t cols, size_t y_chunk, size_t x_chunk) {
    for (uint32_t y = 0; y < TILE_SIZE; y++) {
        for (uint32_t x = 0; x < TILE_SIZE; x++) {
            uint32_t global_row = y_chunk + y;
            uint32_t global_col = x_chunk + x;
            tile[y * TILE_SIZE + x] = image[global_row * cols + global_col];
        }
    }
}

// Function to read from output streams and construct the final output image
void ConstructOutputFromStreams(hls::stream<int> output_streams[11], int* acc_output, int rows, int cols) {
    size_t y_stride = TILE_SIZE - KERNEL_SIZE + 1;
    size_t x_stride = TILE_SIZE - KERNEL_SIZE + 1;
    int index = 0;

    // Iterate over chunks and fill acc_output
    for (size_t y_chunk = 0; y_chunk <= rows - TILE_SIZE; y_chunk += y_stride) {
        for (size_t x_chunk = 0; x_chunk <= cols - TILE_SIZE; x_chunk += x_stride) {
            if (index < 11) {
                // Read values from the stream and place them in the correct location
                for (int y = 0; y < y_stride; y++) {
                    for (int x = 0; x < x_stride; x++) {
                        if (!output_streams[index].empty()) {
                            int value = output_streams[index].read();
                            int global_row = y_chunk + y;
                            int global_col = x_chunk + x;
                            acc_output[global_row * cols + global_col] = value;
                        }
                    }
                }
                index++;
            }
        }
    }
}

int main() {

	    // Define a constant 3x3 kernel with values from 1 to 9
    int kernel[3][3] = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };

    // Allocate memory for the image and output
    int* image = (int*)malloc(INPUT_SIZE * INPUT_SIZE * sizeof(int));       // Input image: 1300x1300
    int* subimage[11];
    int* output = (int*)malloc(OUT_SIZE * OUT_SIZE * sizeof(int));  // Output matrix: 1298x1298
    int* acc_output = (int*)malloc(OUT_SIZE * OUT_SIZE * sizeof(int)); // Output matrix for accelerator

    // Allocate memory for each subimage
    for (int i = 0; i < 11; i++) {
        subimage[i] = (int*)malloc(TILE_SIZE * TILE_SIZE * sizeof(int));
        if (subimage[i] == NULL) {
            printf("Memory allocation failed for subimage %d!\n", i);
            return -1;
        }
    }

    // Check if memory allocation was successful
    if (image == NULL || kernel == NULL || output == NULL) {
        printf("Memory allocation failed!\n");
        return -1;
    }

    // Generate random matrices for image
    MatrixGen(image, INPUT_SIZE, INPUT_SIZE);

    // Split the image into 11 subimages, each 120x120
    // Define the stride based on the chunk size and kernel size
    size_t y_stride = TILE_SIZE - KERNEL_SIZE + 1;
    size_t x_stride = TILE_SIZE - KERNEL_SIZE + 1;

        // Iterate over all chunks in the image using the defined stride and store in subimages
        int index = 0;
        for (size_t y_chunk = 0; y_chunk <= INPUT_SIZE - TILE_SIZE; y_chunk += y_stride) {
            for (size_t x_chunk = 0; x_chunk <= INPUT_SIZE - TILE_SIZE; x_chunk += x_stride) {
                if (index < 11) { // Make sure we don't exceed the number of subimages available
                    // Store the chunk directly into the corresponding subimage
                    CopyToTile(image, subimage[index], INPUT_SIZE, INPUT_SIZE, y_chunk, x_chunk);
                    //printf("Stored tile %d at (y_chunk=%zu, x_chunk=%zu)\n", index, y_chunk, x_chunk);
                    index++;
                }
            }
        }

    // Print the generated image and kernel (optional)
    printf("Input Image:\n");
    PrintMatrix(image, 10, 10);

    printf("Kernel:\n");
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            printf("%d ", kernel[i][j]);
        }
        printf("\n");}
    printf("\n");

    int loop_count = 1;
    // Measure start time
    clock_t start = clock();
    for(int i = 0; i < loop_count; i++){
        // Perform the convolution
    CPUConvolution((int*)image, (int*)kernel, (int*)output);
    // Measure end time
    }
    clock_t end = clock();

    // Calculate elapsed time in seconds
    double exec_time = ((double)(end - start) / CLOCKS_PER_SEC) / loop_count;

    // Print the result of the convolution
    printf("Output Image:\n");
    PrintMatrix(output, 10, 10);

    // Print the time taken for convolution
    printf("Time taken for CPU-based convolution: %.3f milliseconds\n", exec_time*1000);


    // ********** Test: Convolution TILES ********** //

//        // Streams for each output of subimage
//            hls::stream<int> output_streams[11];
//            convolve_all(
//                subimage[0], subimage[1], subimage[2], subimage[3], subimage[4],
//                subimage[5], subimage[6], subimage[7], subimage[8], subimage[9],
//                subimage[10], output_streams[0], output_streams[1], output_streams[2],
//                output_streams[3], output_streams[4], output_streams[5], output_streams[6],
//                output_streams[7], output_streams[8], output_streams[9], output_streams[10]);
//
//            // Construct the final convolution result from the output streams
//            ConstructOutputFromStreams(output_streams, acc_output, OUT_SIZE, OUT_SIZE);


    // ********** End of Test ********** //

//    NaiveAccelerator((int*)image, (int*)kernel, (int*)acc_output);
//    TilingConvolution((int*)image, kernel, (int*)acc_output);

    ///******** Performing Tiling convolve *********/////
//    VTDFConvolution(image, kernel, acc_output);
//    int errorCount = 0;
//    Compare((int*)acc_output, (int*)output, OUT_SIZE, OUT_SIZE, &errorCount);

    // Free dynamically allocated memory
    for (int i = 0; i < 11; i++) {
        free(subimage[i]);
    }
    free(image);
    free(output);
    free(acc_output);
    return 0;
}
