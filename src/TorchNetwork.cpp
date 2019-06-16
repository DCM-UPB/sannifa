#include "sannifa/TorchNetwork.hpp"

#include <numeric>
#include <functional>
#include <algorithm>
#include <exception>

#include <iostream>

int countParameters(const torch::nn::AnyModule &model)
{
    int ret = 0;
    for (auto &parameterT : model.ptr()->parameters(true)) {
        ret += parameterT.numel();
    }
    return ret;
}

int copyTensorData(torch::Tensor &tensor, double out[])
{
    const int nvpar = tensor.numel();
    auto flat_tensor = tensor.view(-1);
    auto flat_accessor = flat_tensor.accessor<double, 1>();
    for (int i = 0; i < nvpar; ++i) {
        out[i] = flat_accessor[i];
    }
    return nvpar;
}

void TorchNetwork::_set_requires_grad(const bool requires_grad)
{
    for (at::Tensor &parameterT : _torchNN.ptr()->parameters(true)) {
        parameterT.set_requires_grad(requires_grad);
    }
}

void TorchNetwork::_storeVariationalDerivatives(const int iout, const bool flag_zero_grad)
{   // store computed variational derivatives
    int ivpar = 0;
    for (at::Tensor &parameterT : _torchNN.ptr()->parameters(true)) {
        ivpar += copyTensorData(parameterT.grad(), _currentVD1 + iout*_nvpar + ivpar);
    }
    if (flag_zero_grad) { _torchNN.ptr()->zero_grad(); } // zero out model's gradient storage
}


TorchNetwork::TorchNetwork(const torch::nn::AnyModule &torchNN, const int ninput, const int noutput):
        ANNFunctionInterface(ninput, noutput, countParameters(torchNN))
{
    _torchNN = torchNN.clone();

    // store conversion from flat parameter indexing
    // to nested vector<Tensor> model.parameters() indexing
    _vparIndex1 = new int[_nvpar];
    _vparIndex2 = new int[_nvpar];
    int ivpar = 0;
    int ivector = 0;
    for (auto &parameterT : _torchNN.ptr()->parameters(true)) {
        auto flat_tensor = parameterT.view(-1);
        auto flat_accessor = flat_tensor.accessor<double, 1>();
        for (int itensor = 0; itensor < flat_accessor.size(0); ++itensor) {
            _vparIndex1[ivpar] = ivector;
            _vparIndex2[ivpar] = itensor;
            ++ivpar;
        }
        ++ivector;
    }

    _currentOutput = new double[_noutput];
    for (int i = 0; i < _noutput; ++i) {
        _currentOutput[i] = 0.;
    }
}

TorchNetwork::~TorchNetwork()
{
    delete[] _vparIndex1;
    delete[] _vparIndex2;
    delete[] _currentOutput;

    if (this->hasFirstDerivative()) {
        delete[] _currentD1;
    }
    if (this->hasSecondDerivative()) {
        delete[] _currentD2;
    }
    if (this->hasVariationalFirstDerivative()) {
        delete[] _currentVD1;
    }
}

const torch::nn::AnyModule &TorchNetwork::getTorchNN() const
{
    return _torchNN;
}

void TorchNetwork::saveToFile(const std::string &filename) const
{
    torch::save(_torchNN.ptr(), filename);
}

void TorchNetwork::printInfo(const bool verbose) const
{
    using namespace std;
    ANNFunctionInterface::printInfo(verbose);
    if (verbose) {
        ostream &oStream = cout;
        oStream << endl;
        oStream << "PyTorch NN Info:" << endl;
        _torchNN.ptr()->pretty_print(oStream);
        oStream << endl;
    }
}


double TorchNetwork::getVariationalParameter(const int ivp) const
{
    auto flat_tensor = _torchNN.ptr()->parameters(true)[_vparIndex1[ivp]].view(-1);
    auto flat_accessor = flat_tensor.accessor<double, 1>();
    return flat_accessor[_vparIndex2[ivp]];
}

void TorchNetwork::getVariationalParameters(double vp[]) const
{
    int ivpar = 0;
    for (auto &parameterT : _torchNN.ptr()->parameters(true)) {
        auto flat_tensor = parameterT.view(-1);
        auto flat_accessor = flat_tensor.accessor<double, 1>();
        for (int i = 0; i < flat_accessor.size(0); ++i) {
            vp[ivpar] = flat_accessor[i];
            ++ivpar;
        }
    }
}

void TorchNetwork::setVariationalParameter(const int ivp, const double vp)
{
    auto flat_tensor = _torchNN.ptr()->parameters(true)[_vparIndex1[ivp]].view(-1);
    auto flat_accessor = flat_tensor.accessor<double, 1>();
    flat_accessor[_vparIndex2[ivp]] = vp;
}

void TorchNetwork::setVariationalParameters(const double vp[])
{
    int ivpar = 0;
    for (auto &parameterT : _torchNN.ptr()->parameters(true)) {
        auto flat_tensor = parameterT.view(-1);
        auto flat_accessor = flat_tensor.accessor<double, 1>();
        for (int i = 0; i < flat_accessor.size(0); ++i) {
            flat_accessor[i] = vp[ivpar];
            ++ivpar;
        }
    }
}


void TorchNetwork::_enableFirstDerivative()
{
    if (this->hasFirstDerivative()) { return; }
    _currentD1 = new double[_noutput*_ninput];
    std::fill(_currentD1, _currentD1 + _noutput*_ninput, 0.);
}

void TorchNetwork::_enableSecondDerivative()
{
    if (this->hasSecondDerivative()) { return; }
    _currentD2 = new double[_noutput*_ninput];
    std::fill(_currentD2, _currentD2 + _noutput*_ninput, 0.);
}

void TorchNetwork::_enableVariationalFirstDerivative()
{
    if (this->hasVariationalFirstDerivative()) { return; }
    _currentVD1 = new double[_noutput*_nvpar];
    std::fill(_currentVD1, _currentVD1 + _noutput*_nvpar, 0.);
}

void TorchNetwork::_enableCrossFirstDerivative()
{
    throw std::runtime_error("CrossFirstDerivative not implemented yet in TorchNetwork.");
}

void TorchNetwork::_enableCrossSecondDerivative()
{
    throw std::runtime_error("CrossSecondDerivative not implemented yet in TorchNetwork.");
}


void TorchNetwork::_evaluate(const double in[], const bool flag_deriv)
{    // propagate and compute output (if flag_deriv, incl. all enabled gradients)

    // control flags
    const bool doBackward = (flag_deriv && this->hasDerivatives());
    const bool doInputGrad = (flag_deriv && this->hasInputDerivatives());
    const bool doParamGrad = (flag_deriv && this->hasVariationalDerivatives());
    const bool doMultiBackward = (flag_deriv && (this->hasSecondDerivative() || this->hasCrossFirstDerivative()));

    // flag indicating separate input/parameter gradient calculations, for better performance
    const bool doSeparateGrad = (doParamGrad && doMultiBackward && !this->hasCrossFirstDerivative());

    if (doParamGrad) { // enable/disable parameters requires_grad
        this->_set_requires_grad(true);
        _torchNN.ptr()->zero_grad();
    }
    else {
        this->_set_requires_grad(false);
    }

    double inCopy[_ninput]; // de-const the input to use it in from_blob
    for (int i = 0; i < _ninput; ++i) {
        inCopy[i] = in[i];
    }
    auto inputTensor = torch::from_blob(inCopy, {_ninput}, torch::dtype(torch::kFloat64));

    if (!doSeparateGrad) { // we calculate everything based on a single forward
        if (doInputGrad) { inputTensor.set_requires_grad(true); }
        auto outputTensor = _torchNN.forward(inputTensor).view(-1);
        copyTensorData(outputTensor, _currentOutput);

        if (doBackward) {
            for (int i = 0; i < _noutput; ++i) {
                outputTensor[i].backward(torch::nullopt, true, doMultiBackward);

                if (this->hasVariationalFirstDerivative()) { // store variational derivative
                    _storeVariationalDerivatives(i, true); // which also calls zero_grad
                }

                if (this->hasFirstDerivative()) {
                    copyTensorData(inputTensor.grad(), _currentD1 + i*_ninput); // store first derivative
                }

                if (this->hasSecondDerivative()) {
                    auto d1Tensor = inputTensor.grad().clone();
                    for (int j = 0; j < _ninput; ++j) {  // compute hessian diagonal elements
                        inputTensor.grad().zero_();
                        d1Tensor[j].backward(torch::nullopt, true, false);
                        _currentD2[i*_ninput + j] = inputTensor.grad().accessor<double, 1>()[j]; // store second derivative
                    }
                }

                if (doInputGrad) { inputTensor.grad().zero_(); } // we zero here for the next output pass
            }
        }
    }
    else { // we do separate forward/backwards for parameters and input gradients
        inputTensor.set_requires_grad(false); // first pass without input gradients
        auto outputTensor = _torchNN.forward(inputTensor).view(-1);
        copyTensorData(outputTensor, _currentOutput); // store output here

        for (int i = 0; i < _noutput; ++i) {
            outputTensor[i].backward(torch::nullopt, true, false);
            _storeVariationalDerivatives(i, true); // stores gradients and calls zero_grad()
        }

        this->_set_requires_grad(false); // second pass without param gradients
        inputTensor.set_requires_grad(true); // but with input gradients
        outputTensor = _torchNN.forward(inputTensor).view(-1);

        for (int i = 0; i < _noutput; ++i) {
            outputTensor[i].backward(torch::nullopt, true, true);

            auto d1Tensor = inputTensor.grad().clone();
            if (this->hasFirstDerivative()) {
                copyTensorData(d1Tensor, _currentD1 + i*_ninput);
                inputTensor.grad().zero_(); // we zero here for the multi-backward passes
            }

            for (int j = 0; j < _ninput; ++j) {  // compute hessian diagonal elements
                d1Tensor[j].backward(torch::nullopt, true, false);
                _currentD2[i*_ninput + j] = inputTensor.grad().accessor<double, 1>()[j];
                inputTensor.grad().zero_(); // we zero here for the next input or output pass
            }
        }
        this->_set_requires_grad(true); // reenable to restore original state
    }

    if (doMultiBackward) { inputTensor.grad().detach_(); } // else we will leak memory heavily
}

void TorchNetwork::getOutput(double out[]) const
{
    for (int i = 0; i < _noutput; ++i) { out[i] = _currentOutput[i]; }
}

double TorchNetwork::getOutput(const int i) const
{
    return _currentOutput[i];
}

void TorchNetwork::getFirstDerivative(double d1[]) const
{
    std::copy(_currentD1, _currentD1 + _noutput*_ninput, d1);
}

void TorchNetwork::getFirstDerivative(const int iout, double d1[]) const
{
    for (int j = 0; j < _ninput; ++j) { d1[j] = _currentD1[iout*_ninput + j]; }
}

double TorchNetwork::getFirstDerivative(const int iout, const int i1d) const
{
    return _currentD1[iout*_ninput + i1d];
}

void TorchNetwork::getSecondDerivative(double d2[]) const
{
    std::copy(_currentD2, _currentD2 + _noutput*_ninput, d2);
}

void TorchNetwork::getSecondDerivative(const int iout, double d2[]) const
{
    for (int j = 0; j < _ninput; ++j) { d2[j] = _currentD2[iout*_ninput + j]; }
}

double TorchNetwork::getSecondDerivative(const int iout, const int i2d) const
{
    return _currentD2[iout*_ninput + i2d];
}

void TorchNetwork::getVariationalFirstDerivative(double vd1[]) const
{
    std::copy(_currentVD1, _currentVD1 + _noutput*_nvpar, vd1);
}

void TorchNetwork::getVariationalFirstDerivative(const int iout, double vd1[]) const
{
    for (int j = 0; j < _nvpar; ++j) { vd1[j] = _currentVD1[iout*_nvpar + j]; }
}

double TorchNetwork::getVariationalFirstDerivative(const int iout, const int iv1d) const
{
    return _currentVD1[iout*_nvpar + iv1d];
}

void TorchNetwork::getCrossFirstDerivative(double d1vd1[]) const {}
void TorchNetwork::getCrossFirstDerivative(const int iout, double d1vd1[]) const {}
double TorchNetwork::getCrossFirstDerivative(const int iout, const int i1d, const int iv1d) const { return 0.; }

void TorchNetwork::getCrossSecondDerivative(double d2vd1[]) const {}
void TorchNetwork::getCrossSecondDerivative(const int iout, double d2vd1[]) const {}
double TorchNetwork::getCrossSecondDerivative(const int iout, const int i2d, const int iv1d) const { return 0.; }
