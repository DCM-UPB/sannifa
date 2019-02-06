#include <torch/torch.h>
#include <iostream>

int main() {
    for (int i=0; i<500000; ++i) {
        auto X = torch::ones({2,}, torch::requires_grad(true));
        auto Y = torch::dot(X,X);
        std::cout << "Output" << std::endl << Y << std::endl;

        Y.backward(c10::nullopt, true, true);
        auto Xgrad = X.grad().clone(); // holds my first deriv
        std::cout << "First derivatives" << std::endl << Xgrad << std::endl;

        for (int j=0; j<2; ++j) {
            X.grad().zero_();
            Xgrad[j].backward(c10::nullopt, (j<1) ? true : false, false);
            std::cout << "Second derivatives [" << j << "]" << std::endl << X.grad() << std::endl;
        }
        Y.detach();
        Xgrad.detach();
        X.grad().detach();
        X.detach();
    }
}
