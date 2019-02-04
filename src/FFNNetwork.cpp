#include "sannifa/FFNNetwork.hpp"

FFNNetwork::FFNNetwork(FeedForwardNeuralNetwork * ffnn): ANNFunctionInterface(ffnn->getNInput(), ffnn->getNOutput(), ffnn->getNVariationalParameters())
{
    _bareFFNN = new FeedForwardNeuralNetwork(ffnn);
    _derivFFNN = new FeedForwardNeuralNetwork(ffnn);
}

FFNNetwork::~FFNNetwork()
{
    delete _bareFFNN;
    delete _derivFFNN;
}

FeedForwardNeuralNetwork * FFNNetwork::getBareFFNN()
{
    return _bareFFNN;
}

FeedForwardNeuralNetwork * FFNNetwork::getDerivFFNN()
{
    return _derivFFNN;
}


double FFNNetwork::getVariationalParameter(const int ivp)
{
    return _bareFFNN->getVariationalParameter(ivp);
}

void FFNNetwork::getVariationalParameters(double * vp)
{
    _bareFFNN->getVariationalParameter(vp);
}

void FFNNetwork::setVariationalParameter(const int ivp, const double vp)
{
    _bareFFNN->setVariationalParameter(ivp, vp);
    _derivFFNN->setVariationalParameter(ivp, vp);
}

void FFNNetwork::setVariationalParameters(const double * vp)
{
    _bareFFNN->setVariationalParameter(vp);
    _derivFFNN->setVariationalParameter(vp);
}


void FFNNetwork::enableFirstDerivative()
{
    _derivFFNN->addFirstDerivativeSubstrate();
    _enableFirstDerivative();
}

void FFNNetwork::enableSecondDerivative()
{
    _derivFFNN->addSecondDerivativeSubstrate();
    _enableSecondDerivative();
}

void FFNNetwork::enableVariationalFirstDerivative()
{
    _derivFFNN->addVariationalFirstDerivativeSubstrate();
    _enableVariationalFirstDerivative();
}

void FFNNetwork::enableCrossFirstDerivative()
{
    _derivFFNN->addCrossFirstDerivativeSubstrate();
    _enableCrossFirstDerivative();
}

void FFNNetwork::enableCrossSecondDerivative()
{
    _derivFFNN->addCrossSecondDerivativeSubstrate();
    _enableCrossSecondDerivative();
}

void FFNNetwork::evaluate(const double * in, const bool flag_deriv)
{
    FeedForwardNeuralNetwork * ffnnToUse = flag_deriv ? _derivFFNN : _bareFFNN;
    ffnnToUse->setInput(in);
    ffnnToUse->FFPropagate();
    _flag_deriv = flag_deriv;
}


void FFNNetwork::getOutput(double * out)
{
    if (_flag_deriv){
        _derivFFNN->getOutput(out);
    }
    else {
        _bareFFNN->getOutput(out);
    }
}

double FFNNetwork::getOutput(const int i)
{
    if (_flag_deriv){
        return _derivFFNN->getOutput(i);
    }
    else {
        return  _bareFFNN->getOutput(i);
    }
}

void FFNNetwork::getFirstDerivative(double ** d1)
{
    _derivFFNN->getFirstDerivative(d1);
}

void FFNNetwork::getFirstDerivative(const int iout, double * d1)
{
    _derivFFNN->getFirstDerivative(iout, d1);
}

double FFNNetwork::getFirstDerivative(const int iout, const int i1d)
{
    return _derivFFNN->getFirstDerivative(iout, i1d);
}

void FFNNetwork::getSecondDerivative(double ** d2)
{
    _derivFFNN->getSecondDerivative(d2);
}

void FFNNetwork::getSecondDerivative(const int iout, double * d2)
{
    _derivFFNN->getSecondDerivative(iout, d2);
}

double FFNNetwork::getSecondDerivative(const int iout, const int i2d)
{
    return _derivFFNN->getSecondDerivative(iout, i2d);
}

void FFNNetwork::getVariationalFirstDerivative(double ** vd1)
{
    _derivFFNN->getVariationalFirstDerivative(vd1);
}

void FFNNetwork::getVariationalFirstDerivative(const int iout, double * vd1)
{
    _derivFFNN->getVariationalFirstDerivative(iout, vd1);
}

double FFNNetwork::getVariationalFirstDerivative(const int iout, const int iv1d)
{
    return _derivFFNN->getVariationalFirstDerivative(iout, iv1d);
}

void FFNNetwork::getCrossFirstDerivative(double *** d1vd1)
{
    _derivFFNN->getCrossFirstDerivative(d1vd1);
}

void FFNNetwork::getCrossFirstDerivative(const int iout, double ** d1vd1)
{
    _derivFFNN->getCrossFirstDerivative(iout, d1vd1);
}

double FFNNetwork::getCrossFirstDerivative(const int iout, const int i1d, const int iv1d)
{
    return _derivFFNN->getCrossFirstDerivative(iout, i1d, iv1d);
}

void FFNNetwork::getCrossSecondDerivative(double *** d2vd1)
{
    _derivFFNN->getCrossSecondDerivative(d2vd1);
}

void FFNNetwork::getCrossSecondDerivative(const int iout, double ** d2vd1)
{
    _derivFFNN->getCrossSecondDerivative(iout, d2vd1);
}

double FFNNetwork::getCrossSecondDerivative(const int iout, const int i2d, const int iv1d)
{
    return _derivFFNN->getCrossSecondDerivative(iout, i2d, iv1d);
}
