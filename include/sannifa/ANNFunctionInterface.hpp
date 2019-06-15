#ifndef ANN_FUNCTION_INTERFACE
#define ANN_FUNCTION_INTERFACE

#include <string>
#include <array>

struct DerivativeOptions
{
    bool d1 = false;
    bool d2 = false;
    bool vd1 = false;
    bool cd1 = false;
    bool cd2 = false;
};

class ANNFunctionInterface
{
private:
    DerivativeOptions _dopt = DerivativeOptions();

protected:
    const int _ninput, _noutput, _nvpar;

    // child routines to setup appropriate derivative calculation
    virtual void _enableFirstDerivative() = 0;
    virtual void _enableSecondDerivative() = 0;
    virtual void _enableVariationalFirstDerivative() = 0;
    virtual void _enableCrossFirstDerivative() = 0;
    virtual void _enableCrossSecondDerivative() = 0;

public:
    ANNFunctionInterface(const int ninput, const int noutput, const int nvpar):
        _ninput(ninput), _noutput(noutput), _nvpar(nvpar) {}
    explicit ANNFunctionInterface(const std::array<int, 3> &dimensions):
        _ninput(dimensions[0]), _noutput(dimensions[1]), _nvpar(dimensions[2]) {}
    // if possible, the following constructor should be implemented in child:
    // ChildNetwork(const std::string &filename): ANNFunctionInterface(_loadDimensions(filename)) {...};
    virtual ~ANNFunctionInterface() = default;

    // --- store to file ---
    virtual void saveToFile(const std::string &filename) const = 0; // save to a loadable file

    // --- Get general information about the ANN-function
    virtual void printInfo(bool verbose) const; // can be overriden, but should still be called
    void printInfo() const { this->printInfo(false); } // default
    virtual std::string getLibName() const = 0; // should return a string that identifies the used backend lib

    int getNInput() const {return _ninput;}
    int getNOutput() const {return _noutput;}
    int getNVariationalParameters() const {return _nvpar;}

    bool hasDerivatives() const; // true if any of the below yields true
    bool hasInputDerivatives() const; // true if any input derivatives is present
    bool hasVariationalDerivatives() const; // true if any parameter derivatives is present
    bool hasFirstDerivative() const {return _dopt.d1;}
    bool hasSecondDerivative() const {return _dopt.d2;}
    bool hasVariationalFirstDerivative() const {return _dopt.vd1;}
    bool hasCrossFirstDerivative() const {return _dopt.cd1;}
    bool hasCrossSecondDerivative() const {return _dopt.cd2;}

    // --- Manage the variational parameters (which may contain a subset of network weights and/or other parameters)
    virtual double getVariationalParameter(int ivp) const = 0;
    virtual void getVariationalParameters(double vp[]) const = 0;
    virtual void setVariationalParameter(int ivp, double vp) = 0;
    virtual void setVariationalParameters(const double vp[]) = 0;

    // --- enable derivatives with respect to input, parameters or both
    // child impl is called to setup desired derivatives (if possible)
    void enableFirstDerivative(); // coordinates first derivatives
    void enableSecondDerivative(); // coordinates second derivatives
    void enableVariationalFirstDerivative(); // parameter first derivatives
    void enableCrossFirstDerivative(); // parameters first coordinates first derivatives
    void enableCrossSecondDerivative(); // parameters first coordinates second derivatives

    // shortcut for enabling multiple derivatives
    void enableDerivatives(const DerivativeOptions &doptToEnable);

    // --- Propagation
    // Routine for propagation
    virtual void evaluate(const double in[], bool flag_deriv) = 0;
    void evaluate(const double in[]) { this->evaluate(in, false); } // default

    // --- Get outputs
    // it remains to be decided by child classes
    // how to store and access the output
    virtual void getOutput(double out[]) const = 0;
    virtual double getOutput(int i) const = 0;

    virtual void getFirstDerivative(double d1[]) const = 0; // d1[noutput*ninput]
    virtual void getFirstDerivative(int iout, double d1[]) const = 0; // iout is the output index
    virtual double getFirstDerivative(int iout, int i1d) const = 0; // i1d the input index

    virtual void getSecondDerivative(double d2[]) const = 0; // d2[noutput*ninput]
    virtual void getSecondDerivative(int iout, double d2[]) const = 0;
    virtual double getSecondDerivative(int iout, int i2d) const = 0; // i2d the input index

    virtual void getVariationalFirstDerivative(double vd1[]) const = 0; // vd1[noutput*nvpar]
    virtual void getVariationalFirstDerivative(int iout, double vd1[]) const= 0;
    virtual double getVariationalFirstDerivative(int iout, int iv1d) const = 0; // iv1d the variational parameter index

    virtual void getCrossFirstDerivative(double d1vd1[]) const = 0; // d1vd1[noutput*ninput*nvpar]
    virtual void getCrossFirstDerivative(int iout, double d1vd1[]) const = 0;
    virtual double getCrossFirstDerivative(int iout, int i1d, int iv1d) const = 0;

    virtual void getCrossSecondDerivative(double d2vd1[]) const = 0; // d2vd1[noutput*ninput*nvpar]
    virtual void getCrossSecondDerivative(int iout, double d2vd1[]) const = 0;
    virtual double getCrossSecondDerivative(int iout, int i2d, int iv1d) const = 0;
};


#endif
