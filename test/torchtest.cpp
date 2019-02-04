#include <torch/torch.h>
#include <iostream>

#include "sannifa/TorchNetwork.hpp"

#include "PropagateBenchmark.hpp"
#include "checkDerivatives.hpp"

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
    
    torch::nn::AnyModule anymodel(Model(2, 2, 3));
    TorchNetwork wrapper(anymodel, 2, 3);
    auto model = wrapper.getTorchNN()->get<Model>();

    
    auto in = torch::ones({2,}, torch::dtype(torch::kFloat64));
    auto out = model.forward(in);

    cout << "Raw object input/output:" << endl;
    cout << "input:" << endl << in << endl;
    cout << "output:" << endl << out << endl;
    cout << endl;

    // the same with TorchNetwork wrapper
    cout << endl << "Using wrapper object:" << endl;
    cout << "dimensions: "
         << wrapper.getNInput() << " "
         << wrapper.getNOutput() << " "
         << wrapper.getNVariationalParameters() << endl;

    double input[2];    
    input[0] = 1.;
    input[1] = 1.;

    wrapper.evaluate(input);
    cout << "input: " << "1 1" << endl;
    cout << "output: " << wrapper.getOutput(0) << " " << wrapper.getOutput(1) << " " << wrapper.getOutput(2) << endl;

    // derivative check
    cout << endl << "Derivative check...";
    torch::nn::AnyModule anymodel2(Model(2, 5, 2));
    TorchNetwork wrapper2(anymodel2, 2, 2);
    wrapper2.enableFirstDerivative();
    wrapper2.enableSecondDerivative();
    wrapper2.enableVariationalFirstDerivative();
    checkDerivatives(&wrapper2, 0.0001);
    cout << " Passed." << endl;

    cout << endl << "Benchmark...";
    torch::nn::AnyModule anymodel3(Model(48, 96, 1));
    TorchNetwork wrapper3(anymodel3, 48, 1);
    propagateBenchmark(&wrapper3, 100000);
    cout << " Done." << endl;
}
