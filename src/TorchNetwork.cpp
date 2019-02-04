#include "sannifa/TorchNetwork.hpp"

#include <iostream>

TorchNetwork::TorchNetwork(const torch::nn::AnyModule &torchNN, const int ninput, const int noutput): ANNFunctionInterface(ninput, noutput, 0)
{
    _torchNN = torchNN.clone();

    _currentOutput = new double[_noutput];
    for (int i=0; i<_noutput; ++i) _currentOutput[i] = 0;
}

TorchNetwork::~TorchNetwork()
{
    delete [] _currentOutput;
}

torch::nn::AnyModule * TorchNetwork::getTorchNN()
{
    return &_torchNN;
}


double TorchNetwork::getVariationalParameter(const int ivp)
{
    return 0.;
}

void TorchNetwork::getVariationalParameters(double * vp)
{
    return;
}

void TorchNetwork::setVariationalParameter(const int ivp, const double vp)
{
    return;
}

void TorchNetwork::setVariationalParameters(const double * vp)
{
    return;
}


void TorchNetwork::enableFirstDerivative()
{
    return;
}

void TorchNetwork::enableSecondDerivative()
{
    return;
}

void TorchNetwork::enableVariationalFirstDerivative()
{
    return;
}

void TorchNetwork::enableCrossFirstDerivative()
{
    return;
}

void TorchNetwork::enableCrossSecondDerivative()
{
    return;
}


void TorchNetwork::_evaluate(const double * in, const bool flag_deriv)
{
    // propagate and compute output (if flag_deriv, incl. recording for autodiff)
    double inCopy[_ninput]; // de-const the input
    for (int i=0; i<_ninput; ++i) {inCopy[i] = in[i];}

    torch::Tensor inputTensor = torch::from_blob(inCopy, {_ninput},
        torch::requires_grad(flag_deriv).dtype(torch::kFloat64));
    auto outputTensor = _torchNN.forward(inputTensor);

    auto outputAccessor = outputTensor.accessor<double,1>();
    for (int i=0; i<outputAccessor.size(0); ++i) {
        _currentOutput[i] = outputAccessor[i];
    }
}

void TorchNetwork::evaluate(const double * in)
{
    this->_evaluate(in, false);
}

void TorchNetwork::evaluateWithDerivatives(const double * in)
{
    this->_evaluate(in, true);
}

void TorchNetwork::getOutput(double * out)
{
    for (int i=0; i<_noutput; ++i)  out[i] = _currentOutput[i];
}

double TorchNetwork::getOutput(const int i)
{
    return _currentOutput[i];
}

void TorchNetwork::getFirstDerivative(double ** d1)
{
    return;
}

void TorchNetwork::getFirstDerivative(const int iout, double * d1)
{
    return;
}

double TorchNetwork::getFirstDerivative(const int iout, const int i1d)
{
    return 0.;
}

void TorchNetwork::getSecondDerivative(double ** d2)
{
    return;
}

void TorchNetwork::getSecondDerivative(const int iout, double * d2)
{
    return;
}

double TorchNetwork::getSecondDerivative(const int iout, const int i2d)
{
    return 0.;
}

void TorchNetwork::getVariationalFirstDerivative(double ** vd1)
{
    return;
}

void TorchNetwork::getVariationalFirstDerivative(const int iout, double * vd1)
{
    return;
}

double TorchNetwork::getVariationalFirstDerivative(const int iout, const int iv1d)
{
    return 0.;
}

void TorchNetwork::getCrossFirstDerivative(double *** d1vd1)
{
    return;
}

void TorchNetwork::getCrossFirstDerivative(const int iout, double ** d1vd1)
{
    return;
}

double TorchNetwork::getCrossFirstDerivative(const int iout, const int i1d, const int iv1d)
{
    return 0.;
}

void TorchNetwork::getCrossSecondDerivative(double *** d2vd1)
{
    return;
}

void TorchNetwork::getCrossSecondDerivative(const int iout, double ** d2vd1)
{
    return;
}

double TorchNetwork::getCrossSecondDerivative(const int iout, const int i2d, const int iv1d)
{
    return 0.;
}
