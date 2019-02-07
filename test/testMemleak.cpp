#include <torch/torch.h>

int main() {
    for (int i=0; i<500000; ++i) {
        auto X = torch::ones({1,}, torch::requires_grad(true));
        auto Y = X*X;
        Y.backward(c10::nullopt, true, true);
        X.grad()[0].backward(c10::nullopt, false, false);
        X.grad().detach_(); // this prevents the leak
    }
}
