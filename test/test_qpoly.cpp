#include <iostream>

#include "qnets/poly/FeedForwardNeuralNetwork.hpp"
#include "sannifa/QPolyWrapper.hpp"

#include "checkDerivatives.hpp"

int main() {
    using namespace std;

    // derivative check
    FeedForwardNeuralNetwork ffnn(3, 5, 3);
    ffnn.pushHiddenLayer(5);
    ffnn.connectFFNN();
    ffnn.assignVariationalParameters();
    
    QPolyWrapper wrapper(ffnn);
    DerivativeOptions dopt;
    dopt.d1 = true; dopt.d2 = true;
    dopt.vd1 = true; dopt.cd1 = true; dopt.cd2 = true;
    wrapper.enableDerivatives(dopt);
    
    wrapper.printInfo(true);
    cout << endl << "Derivative check..." << endl;
    checkDerivatives(&wrapper, 0.0001);
    cout << "Passed." << endl;

    wrapper.saveToFile("ffnn.out");
    QPolyWrapper wrapper2("ffnn.out");
    wrapper2.printInfo(true);
}
