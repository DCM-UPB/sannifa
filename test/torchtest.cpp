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
    
    torch::nn::AnyModule anymodel(Model(2, 2, 1));
    TorchNetwork wrapper(anymodel, 2, 1);
    auto model = wrapper.getTorchNN()->get<Model>();

    cout << model << endl;
    for (auto buffer : model.buffers()) {
        cout << "buffer " << buffer << endl;
    }
    for (auto children : model.children()) {
        cout << "children " << children << endl;
    }
    for (auto parameter : model.parameters()) {
        cout << "parameter " << parameter << endl;
    }
    auto in = torch::ones({2,}, torch::dtype(torch::kFloat64));
    auto out = model.forward(in);

    int ninput = model.in->options.in_;
    cout << "ninput " << ninput << endl;
    cout << "input: " << in << endl;

    int noutput = model.out->options.out_;
    cout << "noutput " << noutput << endl;
    cout << "output: " << out << endl;
    cout << endl;

    // the same with TorchNetwork wrapper
    cout << "--- Wrapper ---" << endl;

    double input[2];    
    input[0] = 1.;
    input[1] = 1.;

    wrapper.evaluate(input);
    cout << "input: " << "1 1" << endl;
    cout << "output: " << wrapper.getOutput(0) << endl;

    input[0] = 0.5;
    input[1] = -1.;
    wrapper.evaluate(input); 
    cout << "input: " << "0.5 -1" << endl;
    cout << "output: " << wrapper.getOutput(0) << endl;

    torch::nn::AnyModule anymodel2(Model(48, 96, 1));
    TorchNetwork wrapper2(anymodel2, 48, 1);
    propagateBenchmark(&wrapper, 50000);
}
