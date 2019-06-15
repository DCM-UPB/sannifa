#ifndef QPOLY_NETWORK
#define QPOLY_NETWORK

#include "sannifa/ANNFunctionInterface.hpp"
#include "qnets/poly/FeedForwardNeuralNetwork.hpp"

// ANNFunctionInterface wrapper around DCM-UPB/QNets PolyNet
class QPolyNetwork final: public ANNFunctionInterface
{
private:
    FeedForwardNeuralNetwork * _bareFFNN = nullptr; // the network used for simple evaluation (no derivatives)
    FeedForwardNeuralNetwork * _derivFFNN = nullptr; // and the one that also computes derivatives
    bool _flag_deriv = false; // was the last evaluation with derivFFNN?

    void _enableFirstDerivative() final { _derivFFNN->addFirstDerivativeSubstrate(); }
    void _enableSecondDerivative() final { _derivFFNN->addSecondDerivativeSubstrate(); }
    void _enableVariationalFirstDerivative() final { _derivFFNN->addVariationalFirstDerivativeSubstrate(); }
    void _enableCrossFirstDerivative() final { _derivFFNN->addCrossFirstDerivativeSubstrate(); }
    void _enableCrossSecondDerivative() final { _derivFFNN->addCrossSecondDerivativeSubstrate(); }

public:
    explicit QPolyNetwork(const FeedForwardNeuralNetwork &ffnn); // we keep just a copy of the ffnn object
    explicit QPolyNetwork(const std::string &filename); // load from file

    ~QPolyNetwork() final; // and delete the copy here

    const FeedForwardNeuralNetwork &getBareFFNN() const { return *_bareFFNN; }
    const FeedForwardNeuralNetwork &getDerivFFNN() const { return *_derivFFNN; }

    void saveToFile(const std::string &filename) const final;

    void printInfo(bool verbose) const final; // add backend specific print, if verbose
    std::string getLibName() const final {return "libffnn";}

    double getVariationalParameter(int ivp) const final { return _bareFFNN->getVariationalParameter(ivp); }
    void getVariationalParameters(double vp[]) const final { _bareFFNN->getVariationalParameter(vp); }
    void setVariationalParameter(int ivp, double vp) final;
    void setVariationalParameters(const double vp[]) final;

    void evaluate(const double in[], bool flag_deriv) final;

    void getOutput(double out[]) const final;
    double getOutput(int i) const final;

    void getFirstDerivative(double d1[]) const final;
    void getFirstDerivative(int iout, double d1[]) const final { _derivFFNN->getFirstDerivative(iout, d1); }
    double getFirstDerivative(int iout, int i1d) const final { return _derivFFNN->getFirstDerivative(iout, i1d); }

    void getSecondDerivative(double d2[]) const final;
    void getSecondDerivative(int iout, double d2[]) const final { _derivFFNN->getSecondDerivative(iout, d2); }
    double getSecondDerivative(int iout, int i2d) const final { return _derivFFNN->getSecondDerivative(iout, i2d); }

    void getVariationalFirstDerivative(double vd1[]) const final;
    void getVariationalFirstDerivative(int iout, double vd1[]) const final { _derivFFNN->getVariationalFirstDerivative(iout, vd1); }
    double getVariationalFirstDerivative(int iout, int iv1d) const final { return _derivFFNN->getVariationalFirstDerivative(iout, iv1d); }

    void getCrossFirstDerivative(double d1vd1[]) const final;
    void getCrossFirstDerivative(int iout, double d1vd1[]) const final;
    double getCrossFirstDerivative(int iout, int i1d, int iv1d) const final { return _derivFFNN->getCrossFirstDerivative(iout, i1d, iv1d); }

    void getCrossSecondDerivative(double d2vd1[]) const final;
    void getCrossSecondDerivative(int iout, double d2vd1[]) const final;
    double getCrossSecondDerivative(int iout, int i2d, int iv1d) const final { return _derivFFNN->getCrossSecondDerivative(iout, i2d, iv1d); }
};


#endif
