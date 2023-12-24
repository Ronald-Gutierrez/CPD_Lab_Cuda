#include <iostream>
#include <opencv2/opencv.hpp>

#define CHANNELS 3 // 7 9

__global__
void colorToGreyscaleConversion(unsigned char* Pout, unsigned char* Pin, int width, int height) {
    int Col = threadIdx.x + blockIdx.x * blockDim.x;
    int Row = threadIdx.y + blockIdx.y * blockDim.y;

    if (Col < width && Row < height) {
        int greyOffset = Row * width + Col;
        int rgbOffset = greyOffset * CHANNELS;
        unsigned char r = Pin[rgbOffset];
        unsigned char g = Pin[rgbOffset + 1];
        unsigned char b = Pin[rgbOffset + 2];

        Pout[greyOffset] = 0.21f * r + 0.71f * g + 0.07f * b;
    }
}

int main() {
    cv::Mat inputImage = cv::imread("../cat.jpg");

    if (inputImage.empty()) {
        std::cerr << "Error: No se pudo cargar la imagen." << std::endl;
        return -1;
    }

    int width = inputImage.cols;
    int height = inputImage.rows;
    int size = width * height;

    unsigned char* h_input = inputImage.data;
    unsigned char* h_output = new unsigned char[size];

    unsigned char* d_input, * d_output;
    cudaMalloc(&d_input, sizeof(unsigned char) * size * CHANNELS);
    cudaMalloc(&d_output, sizeof(unsigned char) * size);

    cudaMemcpy(d_input, h_input, sizeof(unsigned char) * size * CHANNELS, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    colorToGreyscaleConversion << <gridSize, blockSize >> > (d_output, d_input, width, height);

    cudaMemcpy(h_output, d_output, sizeof(unsigned char) * size, cudaMemcpyDeviceToHost);

    cv::Mat outputImage(height, width, CV_8UC1, h_output);

    cv::imshow("Imagen Original", inputImage);
    cv::imshow("Imagen en Escala de Grises", outputImage);
    cv::waitKey(0);

    delete[] h_output;
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
