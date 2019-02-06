#include <iostream>

#include "FeedForwardNeuralNetwork.hpp"
#include "sannifa/FFNNetwork.hpp"

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
}
