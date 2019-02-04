#ifndef FFN_NETWORK
#define FFN_NETWORK

#include "sannifa/ANNFunctionInterface.hpp"
#include "FeedForwardNeuralNetwork.hpp"

// ANNFunctionInterface wrapper around DCM-UPB/FeedForwardNeuralNetwork
class FFNNetwork: public ANNFunctionInterface
{
private:
    FeedForwardNeuralNetwork * _bareFFNN = nullptr; // the network used for simple evaluation (no derivatives)
    FeedForwardNeuralNetwork * _derivFFNN = nullptr; // and the one that also computes derivatives
    bool _flag_deriv = false; // was the last evaluation with derivFFNN?

public:
    FFNNetwork(FeedForwardNeuralNetwork * ffnn); // we keep a just a copy of the ffnn object
    ~FFNNetwork(); // and delete the copy here

    FeedForwardNeuralNetwork * getBareFFNN();
    FeedForwardNeuralNetwork * getDerivFFNN();

    double getVariationalParameter(const int ivp);
    void getVariationalParameters(double * vp);
    void setVariationalParameter(const int ivp, const double vp);
    void setVariationalParameters(const double * vp);

    void enableFirstDerivative();
    void enableSecondDerivative();
    void enableVariationalFirstDerivative();
    void enableCrossFirstDerivative();
    void enableCrossSecondDerivative();

    void evaluate(const double * in, const bool flag_deriv = false);

    void getOutput(double * out);
    double getOutput(const int i);

    void getFirstDerivative(double ** d1);
    void getFirstDerivative(const int iout, double * d1);
    double getFirstDerivative(const int iout, const int i1d);

    void getSecondDerivative(double ** d2);
    void getSecondDerivative(const int iout, double * d2);
    double getSecondDerivative(const int iout, const int i2d);

    void getVariationalFirstDerivative(double ** vd1);
    void getVariationalFirstDerivative(const int iout, double * vd1);
    double getVariationalFirstDerivative(const int iout, const int iv1d);

    void getCrossFirstDerivative(double *** d1vd1);
    void getCrossFirstDerivative(const int iout, double ** d1vd1);
    double getCrossFirstDerivative(const int iout, const int i1d, const int iv1d);

    void getCrossSecondDerivative(double *** d2vd1);
    void getCrossSecondDerivative(const int iout, double ** d2vd1);
    double getCrossSecondDerivative(const int iout, const int i2d, const int iv1d);
};


#endif
