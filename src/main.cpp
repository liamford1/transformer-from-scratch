#include "tensor.h"
#include <iostream>

int main() {
    Tensor t(3, 3);

    std::cout << "Inital Tensor: " << std::endl;
    t.display();

    std::cout << "Adding some values: " << std::endl;
    t.setValue(0, 0, 1);
    t.setValue(0, 1, 2);
    t.setValue(0, 2, 3);
    t.display();

    return 0;
}

