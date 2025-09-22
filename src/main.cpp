#include "tensor.h"
#include <iostream>

int main() {
    Tensor a(2, 3); 
    Tensor b(3, 2);
    
    
    a.setValue(0, 0, 1); a.setValue(0, 1, 2); a.setValue(0, 2, 3);
    a.setValue(1, 0, 4); a.setValue(1, 1, 5); a.setValue(1, 2, 6);
    
    b.setValue(0, 0, 7); b.setValue(0, 1, 8);
    b.setValue(1, 0, 9); b.setValue(1, 1, 10);
    b.setValue(2, 0, 11); b.setValue(2, 1, 12);
    
    std::cout << "Matrix A:" << std::endl;
    a.display();
    
    std::cout << "\nMatrix B:" << std::endl;
    b.display();
    
    std::cout << "\nA * B:" << std::endl;
    Tensor c = a.matmul(b);
    c.display();

    // Test addition
    std::cout << "\n=== Testing Addition ===" << std::endl;
    Tensor x(2, 2);
    Tensor y(2, 2);

    // Fill x
    x.setValue(0, 0, 1); x.setValue(0, 1, 2);
    x.setValue(1, 0, 3); x.setValue(1, 1, 4);

    // Fill y  
    y.setValue(0, 0, 5); y.setValue(0, 1, 6);
    y.setValue(1, 0, 7); y.setValue(1, 1, 8);

    std::cout << "Matrix X:" << std::endl;
    x.display();

    std::cout << "\nMatrix Y:" << std::endl;
    y.display();

    std::cout << "\nX + Y:" << std::endl;
    Tensor sum = x.add(y);
    sum.display();
    
    return 0;
}

