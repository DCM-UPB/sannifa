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

QPolyNetwork::QPolyNetwork(const FeedForwardNeuralNetwork &ffnn): ANNFunctionInterface(ffnn.getNInput(), ffnn.getNOutput(), ffnn.getNVariationalParameters())
{
    _bareFFNN = new FeedForwardNeuralNetwork(&ffnn);
    _derivFFNN = new FeedForwardNeuralNetwork(&ffnn);
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


void QPolyNetwork::_evaluate(const double in[], const bool flag_deriv)
{
    FeedForwardNeuralNetwork * const ffnnToUse = flag_deriv ? _derivFFNN : _bareFFNN;
    ffnnToUse->setInput(in);
    ffnnToUse->FFPropagate();
    _flag_deriv = flag_deriv;
}


void QPolyNetwork::saveToFile(const std::string &filename) const
{
    const char * fname_char = filename.c_str();
    _bareFFNN->storeOnFile(fname_char, true);
}

void QPolyNetwork::printInfo(const bool verbose) const
{
    ANNFunctionInterface::printInfo(verbose);
    /*if (verbose) { // to be done later
      }*/
}

void QPolyNetwork::setVariationalParameter(const int ivp, const double vp)
{
    _bareFFNN->setVariationalParameter(ivp, vp);
    _derivFFNN->setVariationalParameter(ivp, vp);
}

void QPolyNetwork::setVariationalParameters(const double vp[])
{
    _bareFFNN->setVariationalParameter(vp);
    _derivFFNN->setVariationalParameter(vp);
}


void QPolyNetwork::getOutput(double out[]) const
{
    if (_flag_deriv){
        _derivFFNN->getOutput(out);
    }
    else {
        _bareFFNN->getOutput(out);
    }
}

double QPolyNetwork::getOutput(const int i) const
{
    if (_flag_deriv){
        return _derivFFNN->getOutput(i);
    }
    else {
        return  _bareFFNN->getOutput(i);
    }
}

void QPolyNetwork::getFirstDerivative(double d1[]) const
{
    for (int iout=0; iout<_noutput; ++iout) { _derivFFNN->getFirstDerivative(iout, d1 + iout*_ninput); }
}

void QPolyNetwork::getSecondDerivative(double d2[]) const
{
    for (int iout=0; iout<_noutput; ++iout) { _derivFFNN->getSecondDerivative(iout, d2 + iout*_ninput); }
}

void QPolyNetwork::getVariationalFirstDerivative(double vd1[]) const
{
    for (int iout=0; iout<_noutput; ++iout) { _derivFFNN->getVariationalFirstDerivative(iout, vd1 + iout*_nvpar); }
}

void QPolyNetwork::getCrossFirstDerivative(double d1vd1[]) const
{
    for (int iout=0; iout<_noutput; ++iout) {
        this->getCrossFirstDerivative(iout, d1vd1 + iout*_ninput*_nvpar);
    }
}

void QPolyNetwork::getCrossFirstDerivative(int iout, double d1vd1[]) const
{
    for (int i1d = 0; i1d < _ninput; ++i1d) {
        _derivFFNN->getCrossFirstDerivative(iout, i1d, d1vd1 + i1d*_nvpar);
    }
}

void QPolyNetwork::getCrossSecondDerivative(double d2vd1[]) const
{
    for (int iout=0; iout<_noutput; ++iout) {
        this->getCrossSecondDerivative(iout, d2vd1 + iout*_ninput*_nvpar);
    }
}

void QPolyNetwork::getCrossSecondDerivative(int iout, double d2vd1[]) const
{
    for (int i2d = 0; i2d < _ninput; ++i2d) {
        _derivFFNN->getCrossSecondDerivative(iout, i2d, d2vd1 + i2d*_nvpar);
    }
}