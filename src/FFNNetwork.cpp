#include "sannifa/FFNNetwork.hpp"

FFNNetwork::FFNNetwork(FeedForwardNeuralNetwork * ffnn)
{
    _ffnn = new FeedForwardNeuralNetwork(ffnn);
}

FFNNetwork::~FFNNetwork()
{
    delete _ffnn;
}

FeedForwardNeuralNetwork * FFNNetwork::getFFNN()
{
    return _ffnn;
}

int FFNNetwork::getNInput()
{
    return _ffnn->getNInput();
}
int FFNNetwork::getNOutput()
{
    return _ffnn->getNOutput();
}

bool FFNNetwork::hasFirstDerivative()
{
    return _ffnn->hasFirstDerivativeSubstrate();
}

bool FFNNetwork::hasSecondDerivative()
{
    return _ffnn->hasSecondDerivativeSubstrate();
}

bool FFNNetwork::hasVariationalFirstDerivative()
{
    return _ffnn->hasVariationalFirstDerivativeSubstrate();
}

bool FFNNetwork::hasCrossFirstDerivative()
{
    return _ffnn->hasCrossFirstDerivativeSubstrate();
}

bool FFNNetwork::hasCrossSecondDerivative(){
    return _ffnn->hasCrossSecondDerivativeSubstrate();
}


int FFNNetwork::getNVariationalParameters()
{
    return _ffnn->getNVariationalParameters();
}

double FFNNetwork::getVariationalParameter(const int ivp)
{
    return _ffnn->getVariationalParameter(ivp);
}

void FFNNetwork::getVariationalParameters(double * vp)
{
    _ffnn->getVariationalParameter(vp);
}

void FFNNetwork::setVariationalParameter(const int ivp, const double vp)
{
    _ffnn->setVariationalParameter(ivp, vp);
}

void FFNNetwork::setVariationalParameters(const double * vp)
{
    _ffnn->setVariationalParameter(vp);
}


void FFNNetwork::enableFirstDerivative()
{
    _ffnn->addFirstDerivativeSubstrate();
}

void FFNNetwork::enableSecondDerivative()
{
    _ffnn->addSecondDerivativeSubstrate();
}

void FFNNetwork::enableVariationalFirstDerivative()
{
    _ffnn->addVariationalFirstDerivativeSubstrate();
}

void FFNNetwork::enableCrossFirstDerivative()
{
    _ffnn->addCrossFirstDerivativeSubstrate();
}

void FFNNetwork::enableCrossSecondDerivative()
{
    _ffnn->addCrossSecondDerivativeSubstrate();
}


void FFNNetwork::setInput(const double * in)
{
    _ffnn->setInput(in);
}

void FFNNetwork::setInput(const int i, const double in)
{
    _ffnn->setInput(i, in);
}


void FFNNetwork::propagate()
{
    _ffnn->FFPropagate();
}


void FFNNetwork::getOutput(double * out)
{
    _ffnn->getOutput(out);
}

double FFNNetwork::getOutput(const int i)
{
    return _ffnn->getOutput(i);
}

void FFNNetwork::getFirstDerivative(double ** d1)
{
    _ffnn->getFirstDerivative(d1);
}

void FFNNetwork::getFirstDerivative(const int iout, double * d1)
{
    _ffnn->getFirstDerivative(iout, d1);
}

double FFNNetwork::getFirstDerivative(const int iout, const int i1d)
{
    return _ffnn->getFirstDerivative(iout, i1d);
}

void FFNNetwork::getSecondDerivative(double ** d2)
{
    _ffnn->getSecondDerivative(d2);
}

void FFNNetwork::getSecondDerivative(const int iout, double * d2)
{
    _ffnn->getSecondDerivative(iout, d2);
}

double FFNNetwork::getSecondDerivative(const int iout, const int i2d)
{
    return _ffnn->getSecondDerivative(iout, i2d);
}

void FFNNetwork::getVariationalFirstDerivative(double ** vd1)
{
    _ffnn->getVariationalFirstDerivative(vd1);
}

void FFNNetwork::getVariationalFirstDerivative(const int iout, double * vd1)
{
    _ffnn->getVariationalFirstDerivative(iout, vd1);
}

double FFNNetwork::getVariationalFirstDerivative(const int iout, const int iv1d)
{
    return _ffnn->getVariationalFirstDerivative(iout, iv1d);
}

void FFNNetwork::getCrossFirstDerivative(double *** d1vd1)
{
    _ffnn->getCrossFirstDerivative(d1vd1);
}

void FFNNetwork::getCrossFirstDerivative(const int iout, double ** d1vd1)
{
    _ffnn->getCrossFirstDerivative(iout, d1vd1);
}

double FFNNetwork::getCrossFirstDerivative(const int iout, const int i1d, const int iv1d)
{
    return _ffnn->getCrossFirstDerivative(iout, i1d, iv1d);
}

void FFNNetwork::getCrossSecondDerivative(double *** d2vd1)
{
    _ffnn->getCrossSecondDerivative(d2vd1);
}

void FFNNetwork::getCrossSecondDerivative(const int iout, double ** d2vd1)
{
    _ffnn->getCrossSecondDerivative(iout, d2vd1);
}

double FFNNetwork::getCrossSecondDerivative(const int iout, const int i2d, const int iv1d)
{
    return _ffnn->getCrossSecondDerivative(iout, i2d, iv1d);
}
