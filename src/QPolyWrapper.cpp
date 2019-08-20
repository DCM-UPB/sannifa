#include "sannifa/QPolyWrapper.hpp"

#include <exception>

// internal helpers

std::array<int, 4> loadDimensions(const std::string &filename)
{
    const char * fname_char = filename.c_str();
    FeedForwardNeuralNetwork ffnn(fname_char);
    std::array<int, 4> ret {ffnn.getNInput(), ffnn.getNInput(), ffnn.getNOutput(), ffnn.getNVariationalParameters()};
    return ret;
}

// class methods

QPolyWrapper::QPolyWrapper(const FeedForwardNeuralNetwork &ffnn): Sannifa(ffnn.getNInput(), ffnn.getNInput(), ffnn.getNOutput(), ffnn.getNVariationalParameters())
{
    _bareFFNN = new FeedForwardNeuralNetwork(&ffnn);
    _derivFFNN = new FeedForwardNeuralNetwork(&ffnn);
}

QPolyWrapper::QPolyWrapper(const std::string &filename): Sannifa(loadDimensions(filename))
{
    const char * fname_char = filename.c_str();
    _bareFFNN = new FeedForwardNeuralNetwork(fname_char);
    _derivFFNN = new FeedForwardNeuralNetwork(fname_char);
}

QPolyWrapper::~QPolyWrapper()
{
    delete _bareFFNN;
    delete _derivFFNN;
}


void QPolyWrapper::_evaluate(const double in[], const bool flag_deriv)
{
    FeedForwardNeuralNetwork * const ffnnToUse = flag_deriv ? _derivFFNN : _bareFFNN;
    ffnnToUse->setInput(in);
    ffnnToUse->FFPropagate();
    _flag_deriv = flag_deriv;
}

void QPolyWrapper::_evaluateDerived(const double in[], const double orig_d1[], const double orig_d2[], const bool flag_deriv)
{
    throw std::runtime_error("[QPolyWrapper::_evaluate] Non-original input feed not supported.");
}

void QPolyWrapper::saveToFile(const std::string &filename) const
{
    const char * fname_char = filename.c_str();
    _bareFFNN->storeOnFile(fname_char, true);
}

void QPolyWrapper::printInfo(const bool verbose) const
{
    Sannifa::printInfo(verbose);
    /*if (verbose) { // to be done later
      }*/
}

void QPolyWrapper::setVariationalParameter(const int ivp, const double vp)
{
    _bareFFNN->setVariationalParameter(ivp, vp);
    _derivFFNN->setVariationalParameter(ivp, vp);
}

void QPolyWrapper::setVariationalParameters(const double vp[])
{
    _bareFFNN->setVariationalParameter(vp);
    _derivFFNN->setVariationalParameter(vp);
}


void QPolyWrapper::getOutput(double out[]) const
{
    if (_flag_deriv){
        _derivFFNN->getOutput(out);
    }
    else {
        _bareFFNN->getOutput(out);
    }
}

double QPolyWrapper::getOutput(const int i) const
{
    if (_flag_deriv){
        return _derivFFNN->getOutput(i);
    }
    else {
        return  _bareFFNN->getOutput(i);
    }
}

void QPolyWrapper::getFirstDerivative(double d1[]) const
{
    for (int iout=0; iout<_noutput; ++iout) { _derivFFNN->getFirstDerivative(iout, d1 + iout*_ninput); }
}

void QPolyWrapper::getSecondDerivative(double d2[]) const
{
    for (int iout=0; iout<_noutput; ++iout) { _derivFFNN->getSecondDerivative(iout, d2 + iout*_ninput); }
}

void QPolyWrapper::getVariationalFirstDerivative(double vd1[]) const
{
    for (int iout=0; iout<_noutput; ++iout) { _derivFFNN->getVariationalFirstDerivative(iout, vd1 + iout*_nvpar); }
}

void QPolyWrapper::getCrossFirstDerivative(double d1vd1[]) const
{
    for (int iout=0; iout<_noutput; ++iout) {
        this->getCrossFirstDerivative(iout, d1vd1 + iout*_ninput*_nvpar);
    }
}

void QPolyWrapper::getCrossFirstDerivative(int iout, double d1vd1[]) const
{
    for (int i1d = 0; i1d < _ninput; ++i1d) {
        _derivFFNN->getCrossFirstDerivative(iout, i1d, d1vd1 + i1d*_nvpar);
    }
}

void QPolyWrapper::getCrossSecondDerivative(double d2vd1[]) const
{
    for (int iout=0; iout<_noutput; ++iout) {
        this->getCrossSecondDerivative(iout, d2vd1 + iout*_ninput*_nvpar);
    }
}

void QPolyWrapper::getCrossSecondDerivative(int iout, double d2vd1[]) const
{
    for (int i2d = 0; i2d < _ninput; ++i2d) {
        _derivFFNN->getCrossSecondDerivative(iout, i2d, d2vd1 + i2d*_nvpar);
    }
}