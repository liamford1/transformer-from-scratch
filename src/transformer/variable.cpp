#include "transformer/variable.h"
#include <cmath>
#include <iostream>
#include <algorithm>
#include <random>
#include <stdexcept>

Variable::Variable(const Tensor& data, bool requires_grad) 
    : data(data), requires_grad(requires_grad) {
    if (requires_grad) {
        if (data.getIs3D()) {
            grad = Tensor(data.getBatchSize(), data.getRows(), data.getCols());
        } else {
            grad = Tensor(data.getRows(), data.getCols());
        }
        grad.fill(0.0f);
    }
}

Variable::Variable(int rows, int cols, bool requires_grad) 
    : data(rows, cols), requires_grad(requires_grad) {
    if (requires_grad) {
        grad = Tensor(rows, cols);
        grad.fill(0.0f);
    }
}

Variable::Variable(int batch_size, int rows, int cols, bool requires_grad) 
    : data(batch_size, rows, cols), requires_grad(requires_grad) {
    if (requires_grad) {
        grad = Tensor(batch_size, rows, cols);
        grad.fill(0.0f);
    }
}

std::shared_ptr<Variable> Variable::create(const Tensor& data, bool requires_grad) {
    return std::make_shared<Variable>(data, requires_grad);
}

std::shared_ptr<Variable> Variable::create(int rows, int cols, bool requires_grad) {
    return std::make_shared<Variable>(rows, cols, requires_grad);
}

std::shared_ptr<Variable> Variable::create(int batch_size, int rows, int cols, bool requires_grad) {
    return std::make_shared<Variable>(batch_size, rows, cols, requires_grad);
}

std::shared_ptr<Variable> Variable::createOutput(const Tensor& result, bool needs_grad) const {
    return std::make_shared<Variable>(result, needs_grad);
}

std::shared_ptr<Variable> Variable::matmul(std::shared_ptr<Variable> other) const {
    data.assertValid("Variable::matmul(lhs)");
    other->data.assertValid("Variable::matmul(rhs)");

    Tensor result = this->data.matmul(other->data);
    bool needs_grad = this->requires_grad || other->requires_grad;
    
    auto output = createOutput(result, needs_grad);
    
    if (needs_grad) {
        auto self_ptr = std::const_pointer_cast<Variable>(
            std::static_pointer_cast<const Variable>(shared_from_this())
        );
        
        output->addChild(self_ptr);
        output->addChild(other);
        
        output->setBackwardFn([self_ptr, other, output]() {
            output->grad.assertValid("Variable::matmul(dOut)");
            self_ptr->data.assertValid("Variable::matmul(self.data)");
            other->data.assertValid("Variable::matmul(other.data)");
            if (self_ptr->requires_grad) {
                Tensor other_transposed = other->data.transpose();
                Tensor self_grad = output->grad.matmul(other_transposed);
                self_ptr->grad.add_inplace(self_grad);
            }
            if (other->requires_grad) {
                Tensor self_transposed = self_ptr->data.transpose();
                Tensor other_grad = self_transposed.matmul(output->grad);
                other->grad.add_inplace(other_grad);
            }
        });
    }
    return output;
}

std::shared_ptr<Variable> Variable::add(std::shared_ptr<Variable> other) const {
    data.assertValid("Variable::add(lhs)");
    other->data.assertValid("Variable::add(rhs)");

    Tensor result = this->data.add(other->data);
    bool needs_grad = this->requires_grad || other->requires_grad;
    auto output = createOutput(result, needs_grad);

    if (needs_grad) {
        auto self_ptr = std::const_pointer_cast<Variable>(
            std::static_pointer_cast<const Variable>(shared_from_this())
        );

        output->addChild(self_ptr);
        output->addChild(other);

        output->setBackwardFn([self_ptr, other, output]() {
            output->grad.assertValid("Variable::add(dOut)");

            const Tensor& x  = self_ptr->data;
            const Tensor& y  = other->data;
            const Tensor& dO = output->grad;

            auto reduce2D = [](const Tensor& g, int R, int C, bool br, bool bc) -> Tensor {
                if (!br && !bc && g.getRows() == R && g.getCols() == C && !g.getIs3D()) {
                    Tensor out(R, C);
                    float* dst = out.raw();
                    const float* src = g.raw();
                    for (int i = 0, n = R * C; i < n; ++i) dst[i] = src[i];
                    return out;
                }

                Tensor out(R, C); out.fill(0.0f);
                const int GR = g.getRows();
                const int GC = g.getCols();

                for (int i = 0; i < R; ++i) {
                    for (int j = 0; j < C; ++j) {
                        float s = 0.0f;
                        if (br && bc) {
                            for (int ii = 0; ii < GR; ++ii)
                                for (int jj = 0; jj < GC; ++jj)
                                    s += g.getValue(ii, jj);
                        } else if (br) {
                            for (int ii = 0; ii < GR; ++ii)
                                s += g.getValue(ii, j);
                        } else if (bc) {
                            for (int jj = 0; jj < GC; ++jj)
                                s += g.getValue(i, jj);
                        } else {
                            s = g.getValue(i, j);
                        }
                        out.setValue(i, j, s);
                    }
                }
                return out;
            };

            auto reduce3Dfrom2D = [](const Tensor& g3, int R, int C, bool br, bool bc) -> Tensor {
                Tensor out(R, C); out.fill(0.0f);
                const int B  = g3.getBatchSize();
                const int GR = g3.getRows();
                const int GC = g3.getCols();

                if (!br && !bc) {
                    for (int b = 0; b < B; ++b)
                        for (int i = 0; i < R; ++i)
                            for (int j = 0; j < C; ++j)
                                out.setValue(i, j, out.getValue(i, j) + g3.getValue(b, i, j));
                    return out;
                }

                for (int i = 0; i < R; ++i) {
                    for (int j = 0; j < C; ++j) {
                        float s = 0.0f;
                        if (br && bc) {
                            for (int b = 0; b < B; ++b)
                                for (int ii = 0; ii < GR; ++ii)
                                    for (int jj = 0; jj < GC; ++jj)
                                        s += g3.getValue(b, ii, jj);
                        } else if (br) { 
                            for (int b = 0; b < B; ++b)
                                for (int ii = 0; ii < GR; ++ii)
                                    s += g3.getValue(b, ii, j);
                        } else { 
                            for (int b = 0; b < B; ++b)
                                for (int jj = 0; jj < GC; ++jj)
                                    s += g3.getValue(b, i, jj);
                        }
                        out.setValue(i, j, s);
                    }
                }
                return out;
            };

            if (self_ptr->requires_grad) {
                if (!x.getIs3D() && !dO.getIs3D()) {
                    bool br = (x.getRows() == 1) && (dO.getRows() > 1);
                    bool bc = (x.getCols() == 1) && (dO.getCols() > 1);
                    Tensor dx = reduce2D(dO, x.getRows(), x.getCols(), br, bc);
                    self_ptr->grad.add_inplace(dx);
                } else if (x.getIs3D() && dO.getIs3D()) {
                    self_ptr->grad.add_inplace(dO);
                } else if (!x.getIs3D() && dO.getIs3D()) {
                    bool br = (x.getRows() == 1) && (dO.getRows() > 1);
                    bool bc = (x.getCols() == 1) && (dO.getCols() > 1);
                    Tensor dx = reduce3Dfrom2D(dO, x.getRows(), x.getCols(), br, bc);
                    self_ptr->grad.add_inplace(dx);
                } else {
                    throw std::runtime_error("add backward: unexpected (3D x, 2D dOut)");
                }
            }

            if (other->requires_grad) {
                const Tensor& yD = other->data;
                if (!yD.getIs3D() && !dO.getIs3D()) {
                    bool br = (yD.getRows() == 1) && (dO.getRows() > 1);
                    bool bc = (yD.getCols() == 1) && (dO.getCols() > 1);
                    Tensor dy = reduce2D(dO, yD.getRows(), yD.getCols(), br, bc);
                    other->grad.add_inplace(dy);
                } else if (yD.getIs3D() && dO.getIs3D()) {
                    other->grad.add_inplace(dO);
                } else if (!yD.getIs3D() && dO.getIs3D()) {
                    bool br = (yD.getRows() == 1) && (dO.getRows() > 1);
                    bool bc = (yD.getCols() == 1) && (dO.getCols() > 1);
                    Tensor dy = reduce3Dfrom2D(dO, yD.getRows(), yD.getCols(), br, bc);
                    other->grad.add_inplace(dy);
                } else {
                    throw std::runtime_error("add backward: unexpected (3D y, 2D dOut)");
                }
            }
        });
    }
    return output;
}


std::shared_ptr<Variable> Variable::scale(float factor) const {
    data.assertValid("Variable::scale(x)");

    Tensor result = this->data.scale(factor);
    auto output = createOutput(result, this->requires_grad);
    
    if (this->requires_grad) {
        auto self_ptr = std::const_pointer_cast<Variable>(
            std::static_pointer_cast<const Variable>(shared_from_this())
        );
        
        output->addChild(self_ptr);
        output->setBackwardFn([self_ptr, factor, output]() {
            output->grad.assertValid("Variable::scale(dOut)");
            if (self_ptr->requires_grad) {
                Tensor scaled_grad = output->grad.scale(factor);
                self_ptr->grad.add_inplace(scaled_grad);
            }
        });
    }
    return output;
}

std::shared_ptr<Variable> Variable::softmax() const {
    data.assertValid("Variable::softmax(x)");

    Tensor result = this->data.softmax();
    auto output = createOutput(result, this->requires_grad);
    
    if (this->requires_grad) {
        auto self_ptr = std::const_pointer_cast<Variable>(
            std::static_pointer_cast<const Variable>(shared_from_this())
        );
        
        output->addChild(self_ptr);
        output->setBackwardFn([self_ptr, result, output]() {
            result.assertValid("Variable::softmax(y)");
            output->grad.assertValid("Variable::softmax(dOut)");

            if (self_ptr->requires_grad) {
                if (result.getIs3D()) {
                    Tensor temp_grad(result.getBatchSize(), result.getRows(), result.getCols());
                    temp_grad.fill(0.0f);
                    
                    for (int b = 0; b < result.getBatchSize(); b++) {
                        for (int i = 0; i < result.getRows(); i++) {
                            float dot_product = 0.0f;
                            for (int j = 0; j < result.getCols(); j++) {
                                dot_product += result.getValue(b, i, j) * output->grad.getValue(b, i, j);
                            }
                            for (int j = 0; j < result.getCols(); j++) {
                                float softmax_grad = result.getValue(b, i, j) * 
                                    (output->grad.getValue(b, i, j) - dot_product);
                                temp_grad.setValue(b, i, j, softmax_grad);
                            }
                        }
                    }
                    self_ptr->grad.add_inplace(temp_grad);
                } else {
                    Tensor temp_grad(result.getRows(), result.getCols());
                    temp_grad.fill(0.0f);
                    
                    for (int i = 0; i < result.getRows(); i++) {
                        float dot_product = 0.0f;
                        for (int j = 0; j < result.getCols(); j++) {
                            dot_product += result.getValue(i, j) * output->grad.getValue(i, j);
                        }
                        for (int j = 0; j < result.getCols(); j++) {
                            float softmax_grad = result.getValue(i, j) * 
                                (output->grad.getValue(i, j) - dot_product);
                            temp_grad.setValue(i, j, softmax_grad);
                        }
                    }
                    self_ptr->grad.add_inplace(temp_grad);
                }
            }
        });
    }
    return output;
}

std::shared_ptr<Variable> Variable::cross_entropy_loss(std::shared_ptr<Variable> targets) const {
    data.assertValid("Variable::cross_entropy_loss(input)");
    targets->data.assertValid("Variable::cross_entropy_loss(targets)");
    
    if (!this->data.getIs3D() && !targets->data.getIs3D()) {
        Tensor loss_tensor(1, 1);
        float total_loss = 0.0f;
        
        if (targets->data.getCols() == 1) { 
            for (int i = 0; i < this->data.getRows(); i++) {
                int target_idx = static_cast<int>(targets->data.getValue(i, 0));
                if (target_idx >= 0 && target_idx < this->data.getCols()) {
                    float prob = std::max(this->data.getValue(i, target_idx), 1e-15f);
                    total_loss -= std::log(prob);
                }
            }
        } else {
            for (int i = 0; i < this->data.getRows(); i++) {
                for (int j = 0; j < this->data.getCols(); j++) {
                    if (targets->data.getValue(i, j) > 0.0f) {
                        float prob = std::max(this->data.getValue(i, j), 1e-15f);
                        total_loss -= targets->data.getValue(i, j) * std::log(prob);
                    }
                }
            }
        }
        total_loss /= this->data.getRows();
        loss_tensor.setValue(0, 0, total_loss);
        auto output = createOutput(loss_tensor, this->requires_grad || targets->requires_grad);
        
        if (output->requires_grad) {
            auto self_ptr = std::const_pointer_cast<Variable>(shared_from_this());
            output->addChild(self_ptr);
            output->addChild(targets);
            output->setBackwardFn([self_ptr, targets, output]() {
                if (self_ptr->requires_grad) {
                    if (targets->data.getCols() == 1) {
                        Tensor grad_tensor(self_ptr->data.getRows(), self_ptr->data.getCols());
                        grad_tensor.fill(0.0f);
                        float scale = 1.0f / self_ptr->data.getRows();
                        
                        for (int i = 0; i < self_ptr->data.getRows(); i++) {
                            int target_idx = static_cast<int>(targets->data.getValue(i, 0));
                            if (target_idx >= 0 && target_idx < self_ptr->data.getCols()) {
                                for (int j = 0; j < self_ptr->data.getCols(); j++) {
                                    float grad_val = self_ptr->data.getValue(i, j) * scale;
                                    if (j == target_idx) {
                                        grad_val -= scale;
                                    }
                                    grad_tensor.setValue(i, j, grad_val);
                                }
                            }
                        }
                        self_ptr->grad.add_inplace(grad_tensor);
                    } else {
                        Tensor diff = self_ptr->data.subtract(targets->data);
                        Tensor scaled_diff = diff.scale(1.0f / self_ptr->data.getRows());
                        self_ptr->grad.add_inplace(scaled_diff);
                    }
                }
            });
        }
        return output;
        
    } else if (this->data.getIs3D()) {
        Tensor loss_tensor(1, 1);
        float total_loss = 0.0f;
        int batch_size = this->data.getBatchSize();
        int seq_len = this->data.getRows();
        int total_elements = batch_size * seq_len;
        
        for (int b = 0; b < batch_size; b++) {
            for (int i = 0; i < seq_len; i++) {
                int target_idx = targets->data.getIs3D() ? static_cast<int>(targets->data.getValue(b, i, 0)) : static_cast<int>(targets->data.getValue(b, i));

                if (target_idx >= 0 && target_idx < this->data.getCols()) {
                    float prob = std::max(this->data.getValue(b, i, target_idx), 1e-15f);
                    total_loss -= std::log(prob);
                }
            }
        }
        
        total_loss /= total_elements;
        loss_tensor.setValue(0, 0, total_loss);
        auto output = createOutput(loss_tensor, this->requires_grad || targets->requires_grad);
        
        if (output->requires_grad) {
            auto self_ptr = std::const_pointer_cast<Variable>(shared_from_this());
            output->addChild(self_ptr);
            output->addChild(targets);
            
            output->setBackwardFn([self_ptr, targets, batch_size, seq_len, total_elements]() {
                if (self_ptr->requires_grad) {
                    Tensor grad_tensor(batch_size, seq_len, self_ptr->data.getCols());
                    grad_tensor.fill(0.0f);
                    float scale = 1.0f / total_elements;
                    
                    for (int b = 0; b < batch_size; b++) {
                        for (int i = 0; i < seq_len; i++) {
                            int target_idx = targets->data.getIs3D() ? static_cast<int>(targets->data.getValue(b, i, 0)) : static_cast<int>(targets->data.getValue(b, i));

                            if (target_idx >= 0 && target_idx < self_ptr->data.getCols()) {
                                for (int j = 0; j < self_ptr->data.getCols(); j++) {
                                    float grad_val = self_ptr->data.getValue(b, i, j) * scale;
                                    if (j == target_idx) {
                                        grad_val -= scale;
                                    }
                                    grad_tensor.setValue(b, i, j, grad_val);
                                }
                            }
                        }
                    }
                    self_ptr->grad.add_inplace(grad_tensor);
                }
            });
        }
        return output;
    } else {
        throw std::runtime_error("Unsupported tensor configuration for cross-entropy loss");
    }
}

std::shared_ptr<Variable> Variable::gelu() const {
    data.assertValid("Variable::gelu(x)");

    Tensor result = this->data;
    for (int i = 0; i < result.numel(); i++) {
        float x = result.raw()[i];
        float cube = x * x * x;
        float gelu_val = 0.5f * x * (1.0f + std::tanh(0.79788456f * (x + 0.044715f * cube)));
        result.raw()[i] = gelu_val;
    }
    auto output = createOutput(result, this->requires_grad);
    
    if (this->requires_grad) {
        auto self_ptr = std::const_pointer_cast<Variable>(shared_from_this());
        output->addChild(self_ptr);
        output->setBackwardFn([self_ptr, output]() {
            output->grad.assertValid("Variable::gelu(dOut)");

            if (self_ptr->requires_grad) {
                Tensor grad_tensor = self_ptr->data.getIs3D() ? Tensor(self_ptr->data.getBatchSize(), self_ptr->data.getRows(), self_ptr->data.getCols()) : Tensor(self_ptr->data.getRows(), self_ptr->data.getCols());

                grad_tensor.fill(0.0f);
                for (int i = 0; i < self_ptr->data.numel(); i++) {
                    float x = self_ptr->data.raw()[i];
                    float cube = x * x * x;
                    float tanh_val = std::tanh(0.79788456f * (x + 0.044715f * cube));
                    float sech_sq = 1.0f - tanh_val * tanh_val;
                    float grad_val = 0.5f * (1.0f + tanh_val + x * sech_sq * 0.79788456f * (1.0f + 3.0f * 0.044715f * x * x));
                    grad_tensor.raw()[i] = grad_val * output->grad.raw()[i];
                }
                self_ptr->grad.add_inplace(grad_tensor);
            }
        });
    }
    return output;
}

std::shared_ptr<Variable> Variable::dropout(float dropout_rate, bool training) const {
    if (!training || dropout_rate == 0.0f) {
        return std::const_pointer_cast<Variable>(shared_from_this());
    }

    data.assertValid("Variable::dropout(x)");
    float scale = 1.0f / (1.0f - dropout_rate);
    Tensor mask = data.getIs3D() ? Tensor(data.getBatchSize(), data.getRows(), data.getCols()) : Tensor(data.getRows(), data.getCols());
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    
    for (int i = 0; i < mask.numel(); i++) {
        mask.raw()[i] = (dis(gen) < dropout_rate) ? 0.0f : scale;
    }

    Tensor result = this->data.elementwise(mask);
    auto output = createOutput(result, this->requires_grad);
    
    if (this->requires_grad) {
        auto self_ptr = std::const_pointer_cast<Variable>(shared_from_this());
        output->addChild(self_ptr);
        output->setBackwardFn([self_ptr, output, mask]() {
            if (self_ptr->requires_grad) {
                Tensor grad_tensor = output->grad.elementwise(mask);
                self_ptr->grad.add_inplace(grad_tensor);
            }
        });
    }
    return output;
}

std::shared_ptr<Variable> Variable::log_softmax() const {
    Tensor result = this->data;
    
    if (!result.getIs3D()) {
        for (int i = 0; i < result.getRows(); i++) {
            
            float max_val = result.getValue(i, 0);
            for (int j = 1; j < result.getCols(); j++) {
                max_val = std::max(max_val, result.getValue(i, j));
            }
            
            float sum_exp = 0.0f;
            for (int j = 0; j < result.getCols(); j++) {
                sum_exp += std::expf(result.getValue(i, j) - max_val);
            }
            float log_sum = std::logf(sum_exp) + max_val;
            
            for (int j = 0; j < result.getCols(); j++) {
                result.setValue(i, j, result.getValue(i, j) - log_sum);
            }
        }
    } else {
        for (int b = 0; b < result.getBatchSize(); b++) {
            for (int i = 0; i < result.getRows(); i++) {
                float max_val = result.getValue(b, i, 0);
                for (int j = 1; j < result.getCols(); j++) {
                    max_val = std::max(max_val, result.getValue(b, i, j));
                }
                
                float sum_exp = 0.0f;
                for (int j = 0; j < result.getCols(); j++) {
                    sum_exp += std::expf(result.getValue(b, i, j) - max_val);
                }
                float log_sum = std::logf(sum_exp) + max_val;
                
                for (int j = 0; j < result.getCols(); j++) {
                    result.setValue(b, i, j, result.getValue(b, i, j) - log_sum);
                }
            }
        }
    }
    
    auto output = createOutput(result, this->requires_grad);
    
    if (this->requires_grad) {
        auto self_ptr = std::const_pointer_cast<Variable>(shared_from_this());
        output->addChild(self_ptr);
        
        output->setBackwardFn([self_ptr, result, output]() {
            if (!result.getIs3D()) {
                Tensor grad(result.getRows(), result.getCols());
                for (int i = 0; i < result.getRows(); i++) {
                    float sum = 0.0f;
                    for (int j = 0; j < result.getCols(); j++) {
                        sum += output->grad.getValue(i, j);
                    }
                    for (int j = 0; j < result.getCols(); j++) {
                        float softmax_val = std::expf(result.getValue(i, j));
                        grad.setValue(i, j, output->grad.getValue(i, j) - softmax_val * sum);
                    }
                }
                self_ptr->grad.add_inplace(grad);
            } else {
                Tensor grad(result.getBatchSize(), result.getRows(), result.getCols());
                for (int b = 0; b < result.getBatchSize(); b++) {
                    for (int i = 0; i < result.getRows(); i++) {
                        float sum = 0.0f;
                        for (int j = 0; j < result.getCols(); j++) {
                            sum += output->grad.getValue(b, i, j);
                        }
                        for (int j = 0; j < result.getCols(); j++) {
                            float softmax_val = std::expf(result.getValue(b, i, j));
                            grad.setValue(b, i, j, output->grad.getValue(b, i, j) - softmax_val * sum);
                        }
                    }
                }
                self_ptr->grad.add_inplace(grad);
            }
        });
    }
    return output;
}

std::shared_ptr<Variable> Variable::nll_loss(std::shared_ptr<Variable> targets) const {
    if (!this->data.getIs3D()) {
        float total_loss = 0.0f;
        int n = this->data.getRows();
        
        for (int i = 0; i < n; i++) {
            int target_idx = static_cast<int>(targets->data.getValue(i, 0));
            if (target_idx >= 0 && target_idx < this->data.getCols()) {
                total_loss -= this->data.getValue(i, target_idx);
            }
        }
        total_loss /= n;
        
        Tensor loss_tensor(1, 1);
        loss_tensor.setValue(0, 0, total_loss);
        auto output = createOutput(loss_tensor, this->requires_grad);
        
        if (this->requires_grad) {
            auto self_ptr = std::const_pointer_cast<Variable>(shared_from_this());
            output->addChild(self_ptr);
            output->addChild(targets);
            
            output->setBackwardFn([self_ptr, targets, n]() {
                Tensor grad(self_ptr->data.getRows(), self_ptr->data.getCols());
                grad.fill(0.0f);
                float scale = -1.0f / n;
                
                for (int i = 0; i < n; i++) {
                    int target_idx = static_cast<int>(targets->data.getValue(i, 0));
                    if (target_idx >= 0 && target_idx < self_ptr->data.getCols()) {
                        grad.setValue(i, target_idx, scale);
                    }
                }
                self_ptr->grad.add_inplace(grad);
            });
        }
        return output;
    } else {
        int batch_size = this->data.getBatchSize();
        int seq_len = this->data.getRows();
        int total = batch_size * seq_len;
        float total_loss = 0.0f;
        
        for (int b = 0; b < batch_size; b++) {
            for (int i = 0; i < seq_len; i++) {
                int target_idx = static_cast<int>(targets->data.getValue(b, i, 0));
                if (target_idx >= 0 && target_idx < this->data.getCols()) {
                    total_loss -= this->data.getValue(b, i, target_idx);
                }
            }
        }
        total_loss /= total;
        
        Tensor loss_tensor(1, 1);
        loss_tensor.setValue(0, 0, total_loss);
        auto output = createOutput(loss_tensor, this->requires_grad);
        
        if (this->requires_grad) {
            auto self_ptr = std::const_pointer_cast<Variable>(shared_from_this());
            output->addChild(self_ptr);
            output->addChild(targets);
            
            output->setBackwardFn([self_ptr, targets, batch_size, seq_len, total]() {
                Tensor grad(batch_size, seq_len, self_ptr->data.getCols());
                grad.fill(0.0f);
                float scale = -1.0f / total;
                
                for (int b = 0; b < batch_size; b++) {
                    for (int i = 0; i < seq_len; i++) {
                        int target_idx = static_cast<int>(targets->data.getValue(b, i, 0));
                        if (target_idx >= 0 && target_idx < self_ptr->data.getCols()) {
                            grad.setValue(b, i, target_idx, scale);
                        }
                    }
                }
                self_ptr->grad.add_inplace(grad);
            });
        }
        return output;
    }
}

void Variable::topologicalSort(std::vector<std::shared_ptr<Variable>>& sorted, std::unordered_set<Variable*>& visited) const {
    if (visited.find(const_cast<Variable*>(this)) != visited.end()) {
        return;
    }
    
    visited.insert(const_cast<Variable*>(this));
    
    for (const auto& child : children) {
        child->topologicalSort(sorted, visited);
    }
    
    sorted.push_back(std::const_pointer_cast<Variable>(
        std::static_pointer_cast<const Variable>(shared_from_this())
    ));
}

void Variable::backward() {
    if (!requires_grad) {
        std::cerr << "Warning: backward() called on Variable that doesn't require grad" << std::endl;
        return;
    }
    
    if (data.numel() != 1) {
        throw std::runtime_error("Variable::backward(): output must be scalar to auto-seed dOut=1. ""For non-scalars, provide an explicit upstream gradient.");
    }
    grad.fill(1.0f);
    
    std::vector<std::shared_ptr<Variable>> sorted;
    std::unordered_set<Variable*> visited;
    topologicalSort(sorted, visited);
    
    for (auto it = sorted.rbegin(); it != sorted.rend(); ++it) {
        if ((*it)->backward_fn) {
            (*it)->backward_fn();
        }
    }
}

void Variable::zeroGrad() {
    if (requires_grad) {
        grad.fill(0.0f);
    }
}

void Variable::release_graph() {
    for (auto& child : children) {
        if (child) {
            child->release_graph();
        }
    }

    children.clear();
    backward_fn = nullptr;
}