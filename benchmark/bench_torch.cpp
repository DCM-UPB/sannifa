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

    std::array<torch::nn::AnyModule, 3> ffnnList {
        torch::nn::AnyModule(Model(6, 12, 1)),
        torch::nn::AnyModule(Model(24, 48, 1)),
        torch::nn::AnyModule(Model(96, 192, 1))
    };

    std::array<int, 3> ninputList {6, 24, 96};

    std::array<array<int, 6>, 3> nstepsList {
        std::array<int, 6>({150000, 30000, 4000, 15000, 15000, 3000}),
        std::array<int, 6>({120000, 30000, 1200, 12000, 12000, 1000}),
        std::array<int, 6>({100000, 20000, 250, 7500, 7500, 250})
    };

/*
    // fast mode
    for (size_t i=0; i<nstepsList.size(); ++i) {
        for (size_t j=0; j<nstepsList[i].size(); ++j) {
            nstepsList[i][j] /= 10;
        }
    }
*/

    std::array<std::string, 3> nameList {
        "small (6x12x12x1)",
        "medium (24x48x48x1)",
        "large (96x192x192x1)"
    };

    cout << endl << "--- Benchmark with libtorch backend ---" << endl;
    for (int i=0; i<3; ++i) {
        // we create two identical wrappers to benchmark
        // separate derivative branches more easily
        TorchNetwork wrapper1(ffnnList[i], ninputList[i], 1);
        TorchNetwork wrapper2(ffnnList[i], ninputList[i], 1);

        cout << endl << "Benchmarking " << nameList[i] << " FFNN with "
             << wrapper1.getNVariationalParameters() << " weights..." << endl;

        // benchmark
        auto times = benchmarkDerivatives(&wrapper1, &wrapper2, nstepsList[i], true);   

        reportTimes(times, nstepsList[i]);
    }
}
