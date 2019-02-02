#include <torch/torch.h>
#include <iostream>

#include "sannifa/TorchNetwork.hpp"

struct Model : torch::nn::Module {

    // Constructor
    Model() {
        // construct and register your layers
        in = register_module("in",torch::nn::Linear(2,4));
        h = register_module("h",torch::nn::Linear(4,4));
        out = register_module("out",torch::nn::Linear(4,1));
    }

    // the forward operation (how data will flow from layer to layer)
    torch::Tensor forward(torch::Tensor X){
        // let's pass relu 
        X = torch::relu(in->forward(X));
        X = torch::relu(h->forward(X));
        X = torch::sigmoid(out->forward(X));
        
        // return the output
        return X;
    }

    const std::shared_ptr<Model> clone(c10::optional<c10::Device>&)
    {
        std::shared_ptr<Model> ptr (new Model());
        torch::serialize::OutputArchive archive;
        //ptr->load_state_dict(this->state_dict());
        return ptr;
    }

    torch::nn::Linear in{nullptr},h{nullptr},out{nullptr};
};


int main() {
    using namespace std;
    
    Model model;
    
    auto in = torch::rand({2,});
    auto out = model.forward(in);

    cout << model << endl;
    cout << in << endl;
    cout << out << endl;

    int ninput = model.in->options.in_;
    cout << "ninput " << ninput << endl;
    int noutput = model.out->options.out_;
    cout << "noutput " << noutput << endl;

    torch::nn::AnyModule anymodel(Model{});
    TorchNetwork wrapper(anymodel, ninput, noutput);
    auto containedModel = (*wrapper.getTorchNN()).get<Model>();

    cout << containedModel << endl;
    for (auto buffer : containedModel.buffers()) {
        cout << "buffer " << buffer << endl;
    }
    for (auto children : containedModel.children()) {
        cout << "children " << children << endl;
    }
    for (auto parameter : containedModel.parameters()) {
        cout << "parameter " << parameter << endl;
    }

    wrapper.setInput(0, 1.);
    wrapper.setInput(1, -1.);
    wrapper.propagate();
    wrapper.propagate();
    //auto out = model.forward(in);
}
