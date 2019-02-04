#ifndef TORCH_NETWORK
#define TORCH_NETWORK

#include "sannifa/ANNFunctionInterface.hpp"
#include <torch/torch.h>

// ANNFunctionInterface wrapper around pytorch models
class TorchNetwork: public ANNFunctionInterface
{
protected:
    torch::nn::AnyModule _torchNN;
    double * _currentOutput = nullptr;

    void _evaluate(const double * in, const bool flag_deriv);

public:
    TorchNetwork(const torch::nn::AnyModule &torchNN, const int ninput, const int noutput); // we keep just a copy of the torch module struct
    ~TorchNetwork();

    torch::nn::AnyModule * getTorchNN();;

    double getVariationalParameter(const int ivp);
    void getVariationalParameters(double * vp);
    void setVariationalParameter(const int ivp, const double vp);
    void setVariationalParameters(const double * vp);

    void enableFirstDerivative();
    void enableSecondDerivative();
    void enableVariationalFirstDerivative();
    void enableCrossFirstDerivative();
    void enableCrossSecondDerivative();

    void evaluate(const double * in);
    void evaluateWithDerivatives(const double * in);

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
