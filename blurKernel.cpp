#include <iostream>
#include <opencv2/opencv.hpp>

#define BLUR_SIZE 3

__global__ 
void blurKernel(unsigned char* in, unsigned char* out, int w, int h) {
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    if (Col < w && Row < h) {
        int pixValR = 0, pixValG = 0, pixValB = 0;
        int pixels = 0;
        for (int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE + 1; ++blurRow) {
            for (int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE + 1; ++blurCol) {
                int curRow = Row + blurRow;
                int curCol = Col + blurCol;
                if (curRow > -1 && curRow < h && curCol > -1 && curCol < w) {
                    pixValB += in[(curRow * w + curCol) * 3];
                    pixValG += in[(curRow * w + curCol) * 3 + 1];
                    pixValR += in[(curRow * w + curCol) * 3 + 2];
                    pixels++;
                }
            }
        }
        out[(Row * w + Col) * 3] = (unsigned char)(pixValB / pixels);
        out[(Row * w + Col) * 3 + 1] = (unsigned char)(pixValG / pixels);
        out[(Row * w + Col) * 3 + 2] = (unsigned char)(pixValR / pixels);
    }
}

int main() {
    cv::Mat inputImage = cv::imread("../cat.jpg");

    if (inputImage.empty()) {
        std::cerr << "Error: No se pudo cargar la imagen" << std::endl;
        return -1;
    }

    int width = inputImage.cols;
    int height = inputImage.rows;

    cv::Mat outputImage(height, width, CV_8UC3);

    unsigned char* d_input, * d_output;
    cudaMalloc((void**)&d_input, width * height * 3 * sizeof(unsigned char));
    cudaMalloc((void**)&d_output, width * height * 3 * sizeof(unsigned char));

    cudaMemcpy(d_input, inputImage.data, width * height * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize - 1) / blockSize, (height + blockSize - 1) / blockSize);

    blurKernel << <gridSize, blockSize >> > (d_input, d_output, width, height);

    cudaMemcpy(outputImage.data, d_output, width * height * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    cv::imshow("Input Image", inputImage);
    cv::imshow("Blurred Image", outputImage);
    cv::waitKey(0);

    return 0;
}
