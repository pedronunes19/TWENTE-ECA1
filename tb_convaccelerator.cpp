#include "convaccelerator.hpp"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main() {
    
    // Define a constant 3x3 kernel with values from 1 to 9
    int kernel[3][3] = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };

    // Allocate memory for the image and output
    uint32_t* image = (uint32_t*)malloc(INPUT_SIZE * INPUT_SIZE * sizeof(uint32_t));       // Input image: 1300x1300
    uint32_t* output = (uint32_t*)malloc(OUT_SIZE * OUT_SIZE * sizeof(uint32_t));  // Output matrix: 1298x1298
    uint32_t* acc_output = (uint32_t*)malloc(OUT_SIZE * OUT_SIZE * sizeof(uint32_t)); // Output matrix for accelerator

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

    printf("Kernel:\n");
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            printf("%d ", kernel[i][j]);
        }
        printf("\n");}
    printf("\n");

    int loop_count = 100;
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

    // Free dynamically allocated memory
    free(image);
    free(output);
    return 0;


}