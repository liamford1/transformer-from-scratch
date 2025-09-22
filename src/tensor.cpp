#include "tensor.h"
#include <iostream>

Tensor::Tensor(int rows, int cols) {
    this->rows = rows;
    this->cols = cols;
    this->data = new float[rows * cols];

    for (int i = 0; i < rows * cols; i++) {
        data[i] = 0.0f;
    }
}

Tensor::~Tensor() {
    delete[] data;
}

float Tensor::getValue(int row, int col) {
    return data[row * this->cols + col];
}

void Tensor::setValue(int row, int col, float value) {
    data[row * this->cols + col] = value;
}

void Tensor::display() {
    for (int i = 0; i < this->rows; i++) {
        for (int j = 0; j < this->cols; j++) {
            std::cout << getValue(i, j) << " ";
        }
        std::cout << std::endl;
    }
}