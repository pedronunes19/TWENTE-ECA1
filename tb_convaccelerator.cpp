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

// Function to read from suboutput tiles and construct the final output image
void ConstructOutput(
    int* suboutputs[126], // Array of pointers to suboutputs
    int* acc_output,
    int output_rows,
    int output_cols)
{
    const int y_stride = TILE_SIZE - KERNEL_SIZE + 1;
    const int x_stride = TILE_SIZE - KERNEL_SIZE + 1;

    printf("Starting ConstructOutput with rows=%d, cols=%d\n", output_rows, output_cols);

    // Iterate over all suboutputs and fill the accumulated output
    int index = 0;
    for (int y_chunk = 0; y_chunk <= output_rows - y_stride; y_chunk += y_stride) {
        for (int x_chunk = 0; x_chunk <= output_cols - x_stride; x_chunk += x_stride) {
            if (index < 121) { // Only process the first 121 tiles
                // Get the current suboutput pointer
                int* current_suboutput = suboutputs[index];

                // Copy the values from the current suboutput tile into the correct position in the output
                for (int y = 0; y < y_stride && (y_chunk + y) < output_rows; y++) {
                    for (int x = 0; x < x_stride && (x_chunk + x) < output_cols; x++) {
                        int value = current_suboutput[y * x_stride + x];
                        int global_row = y_chunk + y;
                        int global_col = x_chunk + x;

                        // Place the value in the correct location in the accumulated output
                        acc_output[global_row * output_cols + global_col] = value;
                    }
                }
                index++;
            }
        }
    }

    printf("Completed ConstructOutput.\n");
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
    int* subimage[126];
    int* suboutput[126];
    int* output = (int*)malloc(OUT_SIZE * OUT_SIZE * sizeof(int));  // Output matrix: 1298x1298
    int* acc_output = (int*)malloc(OUT_SIZE * OUT_SIZE * sizeof(int)); // Output matrix for accelerator

    // Allocate memory for each subimage
    for (int i = 0; i < 126; i++) {
        subimage[i] = (int*)malloc(TILE_H * TILE_W * sizeof(int));
        suboutput[i] = (int*)malloc((TILE_H -2) * (TILE_W-2) * sizeof(int));
        if (subimage[i] == NULL || suboutput[i] == NULL) {
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

    // Print the generated image and kernel (optional)
    printf("Input Image:\n");
    PrintMatrix(image, 10, 10);

    // Split the image into 11 subimages, each 120x120
	// Define the stride based on the chunk size and kernel size
	size_t y_stride = TILE_SIZE - KERNEL_SIZE + 1;
	size_t x_stride = TILE_SIZE - KERNEL_SIZE + 1;

	// Iterate over all chunks in the image using the defined stride and store in subimages
	int index = 0;
	for (size_t y_chunk = 0; y_chunk <= INPUT_SIZE - TILE_SIZE; y_chunk += y_stride) {
		for (size_t x_chunk = 0; x_chunk <= INPUT_SIZE - TILE_SIZE; x_chunk += x_stride) {
			if (index < 121) { // Make sure we don't exceed the number of subimages available
				// Store the chunk directly into the corresponding subimage
				CopyToTile(image, subimage[index], INPUT_SIZE, INPUT_SIZE, y_chunk, x_chunk);
				//printf("Stored tile %d at (y_chunk=%zu, x_chunk=%zu)\n", index, y_chunk, x_chunk);
				index++;
			}
		}
	}

	PrintMatrix(subimage[0],10,10);

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


    // ********** TEST : FIXED_CONVOLUTION HARDWARE ********* //

//    FixedConvolution(image, acc_output);

    // ********** Test: TILES CONVOLUTION ********** //

    // Define constants for the tile size and strides
    for(int a=0;a < 121 ;a+=6){
    convolve_all(subimage[a], subimage[a+1], subimage[a+2], subimage[a+3], subimage[a+4], subimage[a+5],
    		suboutput[a],suboutput[a+1], suboutput[a+2], suboutput[a+3], suboutput[a+4], suboutput[a+5]);
    }

//    printf("Print suboutput... \n "); //debugging
//    PrintMatrix(suboutput[120],10,10);
//    PrintMatrix(suboutput[122],10,10);

    // Call ConstructOutput with all suboutputs
    ConstructOutput(suboutput, acc_output, OUT_SIZE, OUT_SIZE);

    PrintMatrix(acc_output,10,10);
    // ********** End of Test ********** //

    int errorCount = 0;
    Compare((int*)acc_output, (int*)output, OUT_SIZE, OUT_SIZE, &errorCount);

    // Free dynamically allocated memory
    for (int i = 0; i < 126; i++) {
        free(subimage[i]);
        free(suboutput[i]);
    }
    free(image);
    free(output);
    free(acc_output);
    return 0;
}
