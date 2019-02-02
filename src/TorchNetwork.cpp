#include "sannifa/TorchNetwork.hpp"

#include <iostream>

TorchNetwork::TorchNetwork(const torch::nn::AnyModule &torchNN, const int ninput, const int noutput)
{
    _torchNN = torchNN.clone();
    _ninput = ninput;
    _noutput = noutput;
    //_ninput = torchNN.in->options.in_;
    //_noutput = torchNN.out->options.out_;
    _currentInput = new double[_ninput];
}

TorchNetwork::~TorchNetwork()
{
    delete [] _currentInput;
}

torch::nn::AnyModule * TorchNetwork::getTorchNN()
{
    return &_torchNN;
}

int TorchNetwork::getNInput()
{
    return _ninput;
}
int TorchNetwork::getNOutput()
{
    return _noutput;
}

bool TorchNetwork::hasFirstDerivative()
{
    return false;
}

bool TorchNetwork::hasSecondDerivative()
{
    return false;
}

bool TorchNetwork::hasVariationalFirstDerivative()
{
    return false;
}

bool TorchNetwork::hasCrossFirstDerivative()
{
    return false;
}

bool TorchNetwork::hasCrossSecondDerivative(){
    return false;
}


int TorchNetwork::getNVariationalParameters()
{
    return 0;
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


void TorchNetwork::setInput(const double * in)
{
    for (int i=0; i<_ninput; ++i) {
        _currentInput[i] = in[i];
    }
}

void TorchNetwork::setInput(const int i, const double in)
{
    _currentInput[i] = in;
}


void TorchNetwork::propagate()
{
    torch::Tensor inputTensor = torch::from_blob(_currentInput, {_ninput});
    //auto inputTensor = torch::zeros(_ninput, torch::dtype(torch::kFloat64));
    auto outputTensor = _torchNN.forward(inputTensor);
    std::cout << inputTensor << std::endl;
    std::cout << outputTensor << std::endl;
}


void TorchNetwork::getOutput(double * out)
{
    return;
}

double TorchNetwork::getOutput(const int i)
{
    return 0.;
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
