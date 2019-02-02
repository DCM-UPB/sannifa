#ifndef ANN_INTERFACE
#define ANN_INTERFACE

#include <cstddef>

class ANNInterface
{
public:
    virtual ~ANNInterface(){}

    // --- Get general information about the ANN-function
    virtual int getNInput() = 0;
    virtual int getNOutput() = 0;

    virtual bool hasFirstDerivative() = 0;
    virtual bool hasSecondDerivative() = 0;
    virtual bool hasVariationalFirstDerivative() = 0;
    virtual bool hasCrossFirstDerivative() = 0;
    virtual bool hasCrossSecondDerivative() = 0;

    // --- Manage the variational parameters (which may contain a subset of network weights and/or other parameters)
    virtual int getNVariationalParameters() = 0;
    virtual double getVariationalParameter(const int ivp) = 0;
    virtual void getVariationalParameters(double * vp) = 0;
    virtual void setVariationalParameter(const int ivp, const double vp) = 0;
    virtual void setVariationalParameters(const double * vp) = 0;

    // --- enable/disable derivatives with respect to input
    virtual void enableFirstDerivative() = 0;  // coordinates first derivatives
    //virtual void disableFirstDerivative() = 0;
    virtual void enableSecondDerivative() = 0;  // coordinates second derivatives
    //virtual void disableSecondDerivative() = 0;

    // --- enable/disable derivatives with respect to variational parameters
    virtual void enableVariationalFirstDerivative() = 0;  // variational first derivatives
    //virtual void disableVariationalFirstDerivative() = 0;

    // --- enable/disable derivatives with respect to both variational parameters (1st order) and input (1st and 2nd order)
    virtual void enableCrossFirstDerivative() = 0;  // cross first derivatives
    //virtual void disableCrossFirstDerivative() = 0;
    virtual void enableCrossSecondDerivative() = 0;  // cross second derivatives
    //virtual void disableCrossSecondDerivative() = 0;

    // shortcut for enabling multiple derivatives
    void enableDerivatives(const bool flag_d1 = false, const bool flag_d2 = false, const bool flag_vd1 = false, const bool flag_c1d = false, const bool flag_c2d = false);

    // Set input
    virtual void setInput(const double * in) = 0;
    virtual void setInput(const int i, const double in) = 0;

    // --- Computation
    virtual void propagate() = 0;

    // Shortcut for computation: set input and get all values and derivatives with one call.
    // If some derivatives are not supported (substrate missing) the values will be left unchanged.
    void evaluate(const double * in, double * out, double ** d1 = NULL, double ** d2 = NULL, double ** vd1 = NULL);

    // --- Get outputs
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
