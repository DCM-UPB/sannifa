#include <iostream>

#include "FeedForwardNeuralNetwork.hpp"
#include "sannifa/FFNNetwork.hpp"

#include "PropagateBenchmark.hpp"

int main() {
    using namespace std;

    // benchmark
    FeedForwardNeuralNetwork ffnn(49, 97, 2);
    ffnn.pushHiddenLayer(97);
    ffnn.connectFFNN();
    ffnn.assignVariationalParameters();

    FFNNetwork wrapper(&ffnn);

    std::vector<int> nstepsVec {200000, 25000, 10000, 1000};
    auto times = benchmarkDerivatives(&wrapper, nstepsVec, true);
    
    reportTimes(times);
}
