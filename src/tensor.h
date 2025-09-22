#ifndef TENSOR_H
#define TENSOR_H

class Tensor {
    private:
        float* data;
        int rows;
        int cols;
    public:
        Tensor(int rows, int cols);

        ~Tensor();

        float getValue(int row, int col);
        void setValue(int row, int col, float value);
        void display();
};

#endif
