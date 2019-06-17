#include <torch/torch.h>
#include <iostream>

#include "sannifa/PyTorchWrapper.hpp"

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

    std::shared_ptr<Model> clone(const c10::optional<c10::Device>&)
    {
        std::shared_ptr<Model> ptr (new Model((*this)));
        return ptr;
    }

    torch::nn::Linear in{nullptr},h{nullptr},out{nullptr};

};

int main() {
    using namespace std;
    
    torch::nn::AnyModule anymodel(Model(2, 2, 3));
    PyTorchWrapper wrapper(anymodel, 2, 3);
    auto model = wrapper.getTorchNN().get<Model>();
    
    auto in = torch::ones({2,}, torch::dtype(torch::kFloat64));
    auto out = model.forward(in);

    cout << "Raw object input/output:" << endl;
    cout << "input:" << endl << in << endl;
    cout << "output:" << endl << out << endl;
    cout << endl;

    // the same with PyTorchWrapper wrapper
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
    torch::nn::AnyModule anymodel2(Model(2, 4, 2));
    PyTorchWrapper wrapper2(anymodel2, 2, 2);
    DerivativeOptions dopt;
    dopt.d1 = true; dopt.d2 = true; dopt.vd1 = true;
    wrapper2.enableDerivatives(dopt);

    wrapper2.printInfo(true);
    cout << endl << "Derivative check..." << endl;
    checkDerivatives(&wrapper2, 0.0001);
    cout << "Passed." << endl;

    wrapper2.saveToFile("torch.out");
}
