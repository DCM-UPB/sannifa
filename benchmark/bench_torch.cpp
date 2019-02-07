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

    // we create two identical wrappers to benchmark
    // separate derivative branches more easily
    torch::nn::AnyModule anymodel1(Model(48, 96, 1));
    TorchNetwork wrapper1(anymodel1, 48, 1);

    torch::nn::AnyModule anymodel2(Model(48, 96, 1));
    TorchNetwork wrapper2(anymodel2, 48, 1);

    // benchmark
    cout << endl << "--- Benchmark with libtorch backend ---" << endl;
    std::array<int, 6> nsteps {250000, 25000, 2500, 2500, 2500, 1000};
    auto times = benchmarkDerivatives(&wrapper1, &wrapper2, nsteps, true);

    reportTimes(times, nsteps);
}
