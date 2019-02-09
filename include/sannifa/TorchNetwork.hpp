#ifndef TORCH_NETWORK
#define TORCH_NETWORK

#include "sannifa/ANNFunctionInterface.hpp"
#include <torch/torch.h>

// ANNFunctionInterface wrapper around pytorch models
class TorchNetwork: public ANNFunctionInterface
{
protected:
    torch::nn::AnyModule _torchNN; // any torch module to be wrapped

    // these will always be allocated
    int * _vparIndex1 = nullptr; // stores vpar location in torch model.parameters() vector
    int * _vparIndex2 = nullptr; // stores vpar location in flattened tensor (the vparIndex1 element of model.parameters() vector)
    double * _currentOutput = nullptr; // stores last output

    // these will be allocated on demand (enableDerivatives)
    double ** _currentD1 = nullptr; // stores last coordinate derivatives
    double ** _currentD2 = nullptr; // stores last coordinate second derivatives
    double ** _currentVD1 = nullptr; // stores last parameter derivatives

    // call set requires_grad flag on _torchNN.ptr()->parameters()
    void _set_requires_grad(const bool requires_grad);

    // shortcut to store parameters gradients in _currentVD1 and zero out afterwards
    void _storeVariationalDerivatives(const int iout, const bool flag_zero_grad = true); // iout == output index

public:
    TorchNetwork(const torch::nn::AnyModule &torchNN, const int ninput, const int noutput); // we keep just a copy of the torch module
    //explicit TorchNetwork(const std::string &filename); // load from file, not implemented (load the model externally and use the normal constructor instead)

    ~TorchNetwork();

    torch::nn::AnyModule * getTorchNN();

    void saveToFile(const std::string &filename);

    void printInfo(const bool verbose = false); // add backend specific print, if verbose
    std::string getLibName(){return "libtorch";}

    double getVariationalParameter(const int ivp);
    void getVariationalParameters(double * vp);
    void setVariationalParameter(const int ivp, const double vp);
    void setVariationalParameters(const double * vp);

    void enableFirstDerivative();
    void enableSecondDerivative();
    void enableVariationalFirstDerivative();
    void enableCrossFirstDerivative();
    void enableCrossSecondDerivative();

    void evaluate(const double * in, const bool flag_deriv = false);

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
