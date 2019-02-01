#ifndef ANN_INTERFACE
#define ANN_INTERFACE

#include <cstddef>

class ANNInterface
{
public:
    virtual ~ANNInterface();

    // --- Get information about the NN structure
    virtual int getNInput();
    virtual int getNOutput();

    virtual bool isConnected();
    virtual bool hasFirstDerivative();
    virtual bool hasSecondDerivative();
    virtual bool hasVariationalFirstDerivative();
    virtual bool hasCrossFirstDerivative();
    virtual bool hasCrossSecondDerivative();

    // --- Connect the neural network
    virtual void connect();
    virtual void disconnect();

    // --- Manage the variational parameters (which may contain a subset of beta and/or non-beta parameters),
    //     which exist only after that they are assigned to actual parameters in the network (e.g. betas)
    virtual void assignVariationalParameters(const int starting_layer_index = 0); // make weight parameters variational, starting from starting_layer (if the network is layered)
    virtual int getNVariationalParameters();
    virtual double getVariationalParameter(const int ivp);
    virtual void getVariationalParameters(double * vp);
    virtual void setVariationalParameter(const int ivp, const double vp);
    virtual void setVariationalParameters(const double * vp);


    // --- enable/disable derivatives with respect to input
    virtual void enableFirstDerivative();  // coordinates first derivatives
    //virtual void disableFirstDerivative();
    virtual void enableSecondDerivative();  // coordinates second derivatives
    //virtual void disableSecondDerivative();

    // --- enable/disable derivatives with respect to variational parameters
    virtual void enableVariationalFirstDerivative();  // variational first derivatives
    //virtual void disableVariationalFirstDerivative();

    // --- enable/disable derivatives with respect to both variational parameters (1st order) and input (1st and 2nd order)
    virtual void enableCrossFirstDerivative();  // cross first derivatives
    //virtual void disableCrossFirstDerivative();
    virtual void enableCrossSecondDerivative();  // cross second derivatives
    //virtual void disableCrossSecondDerivative();

    // shortcut for enabling multiple derivatives
    void enableDerivatives(const bool flag_d1 = false, const bool flag_d2 = false, const bool flag_vd1 = false, const bool flag_c1d = false, const bool flag_c2d = false);

    // Set input
    virtual void setInput(const double * in);
    virtual void setInput(const int i, const double in);

    // --- Computation
    virtual void propagate();

    // Shortcut for computation: set input and get all values and derivatives with one call.
    // If some derivatives are not supported (substrate missing) the values will be left unchanged.
    void evaluate(const double * in, double * out, double ** d1 = NULL, double ** d2 = NULL, double ** vd1 = NULL);

    // --- Get outputs
    virtual void getOutput(double * out);
    virtual double getOutput(const int i);

    virtual void getFirstDerivative(double ** d1); // d1[noutput][ninput]
    virtual void getFirstDerivative(const int iout, double * d1); // iout is the output index
    virtual double getFirstDerivative(const int iout, const int i1d); // i1d the input index

    virtual void getSecondDerivative(double ** d2); // d2[noutput][ninput]
    virtual void getSecondDerivative(const int iout, double * d2);
    virtual double getSecondDerivative(const int iout, const int i2d); // i2d the input index

    virtual void getVariationalFirstDerivative(double ** vd1); // vd1[noutput][nvpar]
    virtual void getVariationalFirstDerivative(const int iout, double * vd1);
    virtual double getVariationalFirstDerivative(const int iout, const int iv1d); // iv1d the variational parameter index

    virtual void getCrossFirstDerivative(double *** d1vd1); // d1vd1[noutput][ninput][nvpar]
    virtual void getCrossFirstDerivative(const int iout, double ** d1vd1);
    virtual double getCrossFirstDerivative(const int iout, const int i1d, const int iv1d);

    virtual void getCrossSecondDerivative(double *** d2vd1); // d2vd1[noutput][ninput][nvpar]
    virtual void getCrossSecondDerivative(const int iout, double ** d2vd1);
    virtual double getCrossSecondDerivative(const int iout, const int i2d, const int iv1d);
};


#endif
