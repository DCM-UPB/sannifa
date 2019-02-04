#ifndef ANN_FUNCTION_INTERFACE
#define ANN_FUNCTION_INTERFACE

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

    void _enableFirstDerivative(){_dopt.d1 = true;}
    void _enableSecondDerivative(){_dopt.d2 = true;}
    void _enableVariationalFirstDerivative(){_dopt.vd1 = true;}
    void _enableCrossFirstDerivative(){_dopt.cd1 = true;}
    void _enableCrossSecondDerivative(){_dopt.cd2 = true;}

public:
    ANNFunctionInterface(const int ninput, const int noutput, const int nvpar):
        _ninput(ninput), _noutput(noutput), _nvpar(nvpar) {}
    virtual ~ANNFunctionInterface(){}

    // --- Get general information about the ANN-function
    int getNInput(){return _ninput;}
    int getNOutput(){return _noutput;}

    bool hasDerivatives(); // true if any of the below yields true
    bool hasInputDerivatives(); // true if any input derivatives is present
    bool hasVariationalDerivatives(); // true if any parameter derivatives is present
    bool hasFirstDerivative(){return _dopt.d1;}
    bool hasSecondDerivative(){return _dopt.d2;}
    bool hasVariationalFirstDerivative(){return _dopt.vd1;}
    bool hasCrossFirstDerivative(){return _dopt.cd1;}
    bool hasCrossSecondDerivative(){return _dopt.cd2;}

    // --- Manage the variational parameters (which may contain a subset of network weights and/or other parameters)
    int getNVariationalParameters(){return _nvpar;}
    virtual double getVariationalParameter(const int ivp) = 0;
    virtual void getVariationalParameters(double * vp) = 0;
    virtual void setVariationalParameter(const int ivp, const double vp) = 0;
    virtual void setVariationalParameters(const double * vp) = 0;

    // --- enable derivatives with respect to input, parameters or both
    // if the desired derivative is available, child implementation
    // should call the corresponding _enable method, else throw exception
    virtual void enableFirstDerivative() = 0;  // coordinates first derivatives
    virtual void enableSecondDerivative() = 0;  // coordinates second derivatives
    virtual void enableVariationalFirstDerivative() = 0;  // parameter first derivatives
    virtual void enableCrossFirstDerivative() = 0;  // parameters first coordinates first derivatives
    virtual void enableCrossSecondDerivative() = 0; // parameters first coordinates second derivatives

    // shortcut for enabling multiple derivatives
    void enableDerivatives(const DerivativeOptions &doptToEnable);

    // --- Propagation
    // Routine for propagation
    virtual void evaluate(const double * in, const bool flag_deriv = false) = 0;

    // --- Get outputs
    // it remains to be decided by child classes
    // how to store and access the output
    virtual void getOutput(double * out) = 0;
    virtual double getOutput(const int i) = 0;

    virtual void getFirstDerivative(double ** d1) = 0; // d1[noutput][ninput]
    virtual void getFirstDerivative(const int iout, double * d1) = 0; // iout is the output index
    virtual double getFirstDerivative(const int iout, const int i1d) = 0; // i1d the input index

    virtual void getSecondDerivative(double ** d2) = 0; // d2[noutput][ninput]
    virtual void getSecondDerivative(const int iout, double * d2) = 0;
    virtual double getSecondDerivative(const int iout, const int i2d) = 0; // i2d the input index

    virtual void getVariationalFirstDerivative(double ** vd1) = 0; // vd1[noutput][nvpar]
    virtual void getVariationalFirstDerivative(const int iout, double * vd1) = 0;
    virtual double getVariationalFirstDerivative(const int iout, const int iv1d) = 0; // iv1d the variational parameter index

    virtual void getCrossFirstDerivative(double *** d1vd1) = 0; // d1vd1[noutput][ninput][nvpar]
    virtual void getCrossFirstDerivative(const int iout, double ** d1vd1) = 0;
    virtual double getCrossFirstDerivative(const int iout, const int i1d, const int iv1d) = 0;

    virtual void getCrossSecondDerivative(double *** d2vd1) = 0; // d2vd1[noutput][ninput][nvpar]
    virtual void getCrossSecondDerivative(const int iout, double ** d2vd1) = 0;
    virtual double getCrossSecondDerivative(const int iout, const int i2d, const int iv1d) = 0;
};


#endif
