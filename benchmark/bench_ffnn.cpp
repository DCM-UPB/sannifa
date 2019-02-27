#include <iostream>

#include "ffnn/net/FeedForwardNeuralNetwork.hpp"
#include "sannifa/FFNNetwork.hpp"

#include "PropagateBenchmark.hpp"

int main() {
    using namespace std;

    std::array<FeedForwardNeuralNetwork *, 3> ffnnList;
    ffnnList[0] = new FeedForwardNeuralNetwork(7, 13, 2); // small ffnn scalar output
    ffnnList[0]->pushHiddenLayer(13);
    ffnnList[1] = new FeedForwardNeuralNetwork(25, 49, 2); // medium ffnn scalar output
    ffnnList[1]->pushHiddenLayer(49);
    ffnnList[2] = new FeedForwardNeuralNetwork(97, 193, 2); // large ffnn scalar output
    ffnnList[2]->pushHiddenLayer(193);

    std::array<array<int, 6>, 3> nstepsList {
        std::array<int, 6>({4000000, 2000000, 1000000, 200000, 200000, 200000}),
        std::array<int, 6>({700000, 70000, 35000, 3500, 3500, 3500}),
        std::array<int, 6>({75000, 1000, 500, 50, 50, 50})
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

    cout << endl << "--- Benchmark with libffnn backend ---" << endl;
    for (int i=0; i<3; ++i) {
        ffnnList[i]->connectFFNN();
        ffnnList[i]->assignVariationalParameters();

        // we create two identical wrappers to benchmark
        // separate derivative branches more easily
        FFNNetwork wrapper1(ffnnList[i]);
        FFNNetwork wrapper2(ffnnList[i]);

        cout << endl << "Benchmarking " << nameList[i] << " FFNN with "
             << wrapper1.getNVariationalParameters() << " weights..." << endl;

        // benchmark
        auto times = benchmarkDerivatives(&wrapper1, &wrapper2, nstepsList[i], true);
    
        reportTimes(times, nstepsList[i]);

        delete ffnnList[i]; // no longer needed
    }
}
