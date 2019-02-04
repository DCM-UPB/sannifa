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

    FFNNetwork wrapper(&ffnn);
    propagateBenchmark(&wrapper, 50000);
}
