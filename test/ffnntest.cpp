#include <iostream>

#include "FeedForwardNeuralNetwork.hpp"
#include "sannifa/FFNNetwork.hpp"

#include "PropagateBenchmark.hpp"
#include "checkDerivatives.hpp"

int main() {
    using namespace std;

    // derivative check
    FeedForwardNeuralNetwork ffnn(3, 5, 3);
    ffnn.pushHiddenLayer(4);
    ffnn.connectFFNN();
    ffnn.assignVariationalParameters();
    ffnn.addCrossSecondDerivativeSubstrate();
    
    FFNNetwork wrapper(&ffnn);
    checkDerivatives(&wrapper, 0.0001);

    // benchmark
    FeedForwardNeuralNetwork ffnn2(49, 97, 2);
    ffnn2.pushHiddenLayer(97);
    ffnn2.connectFFNN();
    ffnn2.assignVariationalParameters();

    FFNNetwork wrapper2(&ffnn2);
    propagateBenchmark(&wrapper2, 1000);
}
