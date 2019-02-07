#include <iostream>

#include "FeedForwardNeuralNetwork.hpp"
#include "sannifa/FFNNetwork.hpp"

#include "PropagateBenchmark.hpp"

int main() {
    using namespace std;

    FeedForwardNeuralNetwork ffnn(49, 97, 2);
    ffnn.pushHiddenLayer(97);
    ffnn.connectFFNN();
    ffnn.assignVariationalParameters();

    // we create two identical wrappers to benchmark
    // separate derivative branches more easily
    FFNNetwork wrapper1(&ffnn);
    FFNNetwork wrapper2(&ffnn);

    // benchmark
    cout << endl << "--- Benchmark with libffnn backend ---" << endl;
    std::array<int, 6> nsteps {250000, 25000, 2500, 2500, 2500, 1000};
    auto times = benchmarkDerivatives(&wrapper1, &wrapper2, nsteps, true);
    
    reportTimes(times, nsteps);
}
