#include <torch/torch.h>
#include <iostream>

#include "sannifa/TorchNetwork.hpp"
#include "PropagateBenchmark.hpp"

struct Model: public torch::nn::Module {

    // Constructor
    Model(const int insize, const int hsize, const int outsize)
    {
        // construct and register your layers
        in = register_module("in",torch::nn::Linear(insize,hsize));
        h = register_module("h",torch::nn::Linear(hsize,hsize));
        out = register_module("out",torch::nn::Linear(hsize,outsize));
        this->to(torch::kFloat64);
    }

    Model(const Model &other)
    {
        in = register_module("in", other.in);
        h = register_module("h", other.h);
        out = register_module("out", other.out);
        //this->to(torch::kFloat64);
    }

    // the forward operation (how data will flow from layer to layer)
    torch::Tensor forward(torch::Tensor X)
    {
        // let's pass relu 
        X = torch::sigmoid(in->forward(X));
        X = torch::sigmoid(h->forward(X));
        X = torch::sigmoid(out->forward(X));
        
        // return the output
        return X;
    }

    const std::shared_ptr<Model> clone(c10::optional<c10::Device>&)
    {
        std::shared_ptr<Model> ptr (new Model((*this)));
        return ptr;
    }

    torch::nn::Linear in{nullptr},h{nullptr},out{nullptr};

};


int main() {
    using namespace std;

    // benchmark
    torch::nn::AnyModule anymodel(Model(48, 96, 1));
    TorchNetwork wrapper(anymodel, 48, 1);

    std::vector<int> nstepsVec {200000, 25000, 1500, 1000};
    auto times = benchmarkDerivatives(&wrapper, nstepsVec, true);

    reportTimes(times);
}
