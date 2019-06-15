#include "sannifa/QPolyNetwork.hpp"

// internal helpers

std::array<int, 3> loadDimensions(const std::string &filename)
{
    const char * fname_char = filename.c_str();
    FeedForwardNeuralNetwork ffnn(fname_char);
    std::array<int, 3> ret {ffnn.getNInput(), ffnn.getNOutput(), ffnn.getNVariationalParameters()};
    return ret;
}

// class methods

QPolyNetwork::QPolyNetwork(FeedForwardNeuralNetwork * ffnn): ANNFunctionInterface(ffnn->getNInput(), ffnn->getNOutput(), ffnn->getNVariationalParameters())
{
    _bareFFNN = new FeedForwardNeuralNetwork(ffnn);
    _derivFFNN = new FeedForwardNeuralNetwork(ffnn);
}

QPolyNetwork::QPolyNetwork(const std::string &filename): ANNFunctionInterface(loadDimensions(filename))
{
    const char * fname_char = filename.c_str();
    _bareFFNN = new FeedForwardNeuralNetwork(fname_char);
    _derivFFNN = new FeedForwardNeuralNetwork(fname_char);
}

QPolyNetwork::~QPolyNetwork()
{
    delete _bareFFNN;
    delete _derivFFNN;
}

FeedForwardNeuralNetwork * QPolyNetwork::getBareFFNN()
{
    return _bareFFNN;
}

FeedForwardNeuralNetwork * QPolyNetwork::getDerivFFNN()
{
    return _derivFFNN;
}

void QPolyNetwork::saveToFile(const std::string &filename)
{
    const char * fname_char = filename.c_str();
    _bareFFNN->storeOnFile(fname_char, true);
}

void QPolyNetwork::printInfo(const bool verbose)
{
    ANNFunctionInterface::printInfo(verbose);
    /*if (verbose) { // to be done later
      }*/
}


double QPolyNetwork::getVariationalParameter(const int ivp)
{
    return _bareFFNN->getVariationalParameter(ivp);
}

void QPolyNetwork::getVariationalParameters(double * vp)
{
    _bareFFNN->getVariationalParameter(vp);
}

void QPolyNetwork::setVariationalParameter(const int ivp, const double vp)
{
    _bareFFNN->setVariationalParameter(ivp, vp);
    _derivFFNN->setVariationalParameter(ivp, vp);
}

void QPolyNetwork::setVariationalParameters(const double * vp)
{
    _bareFFNN->setVariationalParameter(vp);
    _derivFFNN->setVariationalParameter(vp);
}


void QPolyNetwork::enableFirstDerivative()
{
    _derivFFNN->addFirstDerivativeSubstrate();
    _enableFirstDerivative();
}

void QPolyNetwork::enableSecondDerivative()
{
    _derivFFNN->addSecondDerivativeSubstrate();
    _enableSecondDerivative();
}

void QPolyNetwork::enableVariationalFirstDerivative()
{
    _derivFFNN->addVariationalFirstDerivativeSubstrate();
    _enableVariationalFirstDerivative();
}

void QPolyNetwork::enableCrossFirstDerivative()
{
    _derivFFNN->addCrossFirstDerivativeSubstrate();
    _enableCrossFirstDerivative();
}

void QPolyNetwork::enableCrossSecondDerivative()
{
    _derivFFNN->addCrossSecondDerivativeSubstrate();
    _enableCrossSecondDerivative();
}

void QPolyNetwork::evaluate(const double * in, const bool flag_deriv)
{
    FeedForwardNeuralNetwork * ffnnToUse = flag_deriv ? _derivFFNN : _bareFFNN;
    ffnnToUse->setInput(in);
    ffnnToUse->FFPropagate();
    _flag_deriv = flag_deriv;
}


void QPolyNetwork::getOutput(double * out)
{
    if (_flag_deriv){
        _derivFFNN->getOutput(out);
    }
    else {
        _bareFFNN->getOutput(out);
    }
}

double QPolyNetwork::getOutput(const int i)
{
    if (_flag_deriv){
        return _derivFFNN->getOutput(i);
    }
    else {
        return  _bareFFNN->getOutput(i);
    }
}

void QPolyNetwork::getFirstDerivative(double ** d1)
{
    _derivFFNN->getFirstDerivative(d1);
}

void QPolyNetwork::getFirstDerivative(const int iout, double * d1)
{
    _derivFFNN->getFirstDerivative(iout, d1);
}

double QPolyNetwork::getFirstDerivative(const int iout, const int i1d)
{
    return _derivFFNN->getFirstDerivative(iout, i1d);
}

void QPolyNetwork::getSecondDerivative(double ** d2)
{
    _derivFFNN->getSecondDerivative(d2);
}

void QPolyNetwork::getSecondDerivative(const int iout, double * d2)
{
    _derivFFNN->getSecondDerivative(iout, d2);
}

double QPolyNetwork::getSecondDerivative(const int iout, const int i2d)
{
    return _derivFFNN->getSecondDerivative(iout, i2d);
}

void QPolyNetwork::getVariationalFirstDerivative(double ** vd1)
{
    _derivFFNN->getVariationalFirstDerivative(vd1);
}

void QPolyNetwork::getVariationalFirstDerivative(const int iout, double * vd1)
{
    _derivFFNN->getVariationalFirstDerivative(iout, vd1);
}

double QPolyNetwork::getVariationalFirstDerivative(const int iout, const int iv1d)
{
    return _derivFFNN->getVariationalFirstDerivative(iout, iv1d);
}

void QPolyNetwork::getCrossFirstDerivative(double *** d1vd1)
{
    _derivFFNN->getCrossFirstDerivative(d1vd1);
}

void QPolyNetwork::getCrossFirstDerivative(const int iout, double ** d1vd1)
{
    _derivFFNN->getCrossFirstDerivative(iout, d1vd1);
}

double QPolyNetwork::getCrossFirstDerivative(const int iout, const int i1d, const int iv1d)
{
    return _derivFFNN->getCrossFirstDerivative(iout, i1d, iv1d);
}

void QPolyNetwork::getCrossSecondDerivative(double *** d2vd1)
{
    _derivFFNN->getCrossSecondDerivative(d2vd1);
}

void QPolyNetwork::getCrossSecondDerivative(const int iout, double ** d2vd1)
{
    _derivFFNN->getCrossSecondDerivative(iout, d2vd1);
}

double QPolyNetwork::getCrossSecondDerivative(const int iout, const int i2d, const int iv1d)
{
    return _derivFFNN->getCrossSecondDerivative(iout, i2d, iv1d);
}
