#ifndef TORCH_NETWORK
#define TORCH_NETWORK

#include "sannifa/ANNFunctionInterface.hpp"
#include <torch/torch.h>

// ANNFunctionInterface wrapper around pytorch models
class TorchNetwork final: public ANNFunctionInterface
{
protected:
    torch::nn::AnyModule _torchNN; // any torch module to be wrapped

    // these will always be allocated
    int * _vparIndex1 = nullptr; // stores vpar location in torch model.parameters() vector
    int * _vparIndex2 = nullptr; // stores vpar location in flattened tensor (the vparIndex1 element of model.parameters() vector)
    double * _currentOutput = nullptr; // stores last output

    // these will be allocated on demand (enableDerivatives)
    double * _currentD1 = nullptr; // stores last coordinate derivatives
    double * _currentD2 = nullptr; // stores last coordinate second derivatives
    double * _currentVD1 = nullptr; // stores last parameter derivatives

    // call set requires_grad flag on _torchNN.ptr()->parameters()
    void _set_requires_grad(bool requires_grad);

    // shortcut to store parameters gradients in _currentVD1 and zero out afterwards
    void _storeVariationalDerivatives(int iout, bool flag_zero_grad = true); // iout == output index

    void _enableFirstDerivative() final;
    void _enableSecondDerivative() final;
    void _enableVariationalFirstDerivative() final;
    void _enableCrossFirstDerivative() final;
    void _enableCrossSecondDerivative() final;

    void _evaluate(const double in[], bool flag_deriv) final;

public:
    TorchNetwork(const torch::nn::AnyModule &torchNN, int ninput, int noutput); // we keep just a copy of the torch module
    //explicit TorchNetwork(const std::string &filename); // load from file, not implemented (load the model externally and use the normal constructor instead)

    ~TorchNetwork() final;

    const torch::nn::AnyModule &getTorchNN() const;

    void saveToFile(const std::string &filename) const final;

    void printInfo(bool verbose) const final; // add backend specific print, if verbose
    std::string getLibName() const final { return "libtorch"; }

    double getVariationalParameter(int ivp) const final;
    void getVariationalParameters(double vp[]) const final;
    void setVariationalParameter(int ivp, double vp) final;
    void setVariationalParameters(const double vp[]) final;

    void getOutput(double out[]) const final;
    double getOutput(int i) const final;

    void getFirstDerivative(double d1[]) const final;
    void getFirstDerivative(int iout, double d1[]) const final;
    double getFirstDerivative(int iout, int i1d) const final;

    void getSecondDerivative(double d2[]) const final;
    void getSecondDerivative(int iout, double d2[]) const final;
    double getSecondDerivative(int iout, int i2d) const final;

    void getVariationalFirstDerivative(double vd1[]) const final;
    void getVariationalFirstDerivative(int iout, double vd1[]) const final;
    double getVariationalFirstDerivative(int iout, int iv1d) const final;

    void getCrossFirstDerivative(double d1vd1[]) const final;
    void getCrossFirstDerivative(int iout, double d1vd1[]) const final;
    double getCrossFirstDerivative(int iout, int i1d, int iv1d) const final;

    void getCrossSecondDerivative(double d2vd1[]) const final;
    void getCrossSecondDerivative(int iout, double d2vd1[]) const final;
    double getCrossSecondDerivative(int iout, int i2d, int iv1d) const final;
};

#endif
