#ifndef SANNIFA_QTEMPLWRAPPER_HPP
#define SANNIFA_QTEMPLWRAPPER_HPP

#include "sannifa/Sannifa.hpp"
#include "qnets/templ/TemplNet.hpp"

#include <algorithm>
#include <exception>

// Sannifa wrapper around DCM-UPB/QNets TemplNet
template <class TNet>
class QTemplWrapper final: public Sannifa
{
private:
    bool _flag_deriv = false; // was the last evaluation with derivFFNN?

    void _enableFirstDerivative() final { nn->dflags = nn->dflags.OR(templ::StaticDFlags<templ::DerivConfig::D1>{}); }
    void _enableSecondDerivative() final { nn->dflags = nn->dflags.OR(templ::StaticDFlags<templ::DerivConfig::D12>{}); }
    void _enableVariationalFirstDerivative() final { nn->dflags = nn->dflags.OR(templ::StaticDFlags<templ::DerivConfig::VD1>{}); }
    void _enableCrossFirstDerivative() final { throw std::runtime_error("CrossFirstDerivative not implemented in QTemplWrapper."); }

    void _enableCrossSecondDerivative() final { throw std::runtime_error("CrossSecondDerivative not implemented in QTemplWrapper."); }

    void _evaluate(const double in[], bool flag_deriv) final
    {
        nn->setInput(in, in+TNet::ninput);
        nn->FFPropagate();
    }

public:
    TNet * const nn; // templnet can be safely public


    // Construct

    explicit QTemplWrapper(templ::DynamicDFlags init_dflags = templ::DynamicDFlags{TNet::dconf.dconf()}):
            Sannifa(TNet::ninput, TNet::noutput, TNet::nbeta,
                    DerivativeOptions{init_dflags.d1(), init_dflags.d2(), init_dflags.vd1(), false, false}),
            nn(new TNet(init_dflags)) {} // construct new nn

    explicit QTemplWrapper(const TNet &init_nn):
            Sannifa(TNet::ninput, TNet::noutput, TNet::nbeta,
                    DerivativeOptions{init_nn.hasD1(), init_nn.hasD2(), init_nn.hasVD1(), false, false}),
                    nn(new TNet(init_nn)) {} // we keep just a copy
    QTemplWrapper(const QTemplWrapper &other): QTemplWrapper(other.nn) { } // copy construct
    explicit QTemplWrapper(const std::string &filename) {} // load from file (not yet)

    ~QTemplWrapper() final { delete nn; }


    // Misc

    void saveToFile(const std::string &filename) const final {} // not yet

    void printInfo(bool verbose) const final // add backend specific print, if verbose
    {
        using namespace std;
        Sannifa::printInfo(verbose);
        if (verbose) {
            cout << endl;
            cout << "TemplNet Derivatives (Allowed / Enabled):" << endl;
            cout << "  d1:  " << (nn->allowsD1() ? "1" : "0") << " / " << (nn->hasD1() ? "1" : "0") << endl;
            cout << "  d2:  " << (nn->allowsD2() ? "1" : "0") << " / " << (nn->hasD2() ? "1" : "0") << endl;
            cout << "  vd1: " << (nn->allowsVD1() ? "1" : "0") << " / " << (nn->hasVD1() ? "1" : "0") << endl;
            cout << "  cd1: 0 / 0" << endl;
            cout << "  cd2: 0 / 0" << endl;
        }
    }
    std::string getLibName() const final {return "libqnets/templ";}


    // Access

    double getVariationalParameter(int ivp) const final { return nn->getBeta(ivp); }
    void getVariationalParameters(double vp[]) const final { nn->getBetas(vp, vp + TNet::nbeta); }
    void setVariationalParameter(int ivp, double vp) final { nn->setBeta(ivp, vp); }
    void setVariationalParameters(const double vp[]) final { nn->setBetas(vp, vp + TNet::nbeta); }

    void getOutput(double out[]) const final { std::copy(nn->getOutput().begin(), nn->getOutput().end(), out); }
    double getOutput(int i) const final { return nn->getOutput(i); }

    void getFirstDerivative(double d1[]) const final { std::copy(nn->getD1().begin(), nn->getD1().end(), d1); }
    void getFirstDerivative(int iout, double d1[]) const final { std::copy(nn->getD1().begin() + iout*TNet::ninput, nn->getD1().begin() + (iout+1)*TNet::ninput, d1); }
    double getFirstDerivative(int iout, int i1d) const final { return nn->getD1(iout, i1d); }

    void getSecondDerivative(double d2[]) const final { std::copy(nn->getD2().begin(), nn->getD2().end(), d2); }
    void getSecondDerivative(int iout, double d2[]) const final { std::copy(nn->getD2().begin() + iout*TNet::ninput, nn->getD2().begin() + (iout+1)*TNet::ninput, d2);  }
    double getSecondDerivative(int iout, int i2d) const final { return nn->getD2(iout, i2d); }

    void getVariationalFirstDerivative(double vd1[]) const final { std::copy(nn->getVD1().begin(), nn->getVD1().end(), vd1); }
    void getVariationalFirstDerivative(int iout, double vd1[]) const final { std::copy(nn->getVD1().begin() + iout*TNet::nbeta, nn->getVD1().begin() + (iout+1)*TNet::nbeta, vd1);  }
    double getVariationalFirstDerivative(int iout, int iv1d) const final { return nn->getVD1(iout, iv1d); }

    void getCrossFirstDerivative(double d1vd1[]) const final {}
    void getCrossFirstDerivative(int iout, double d1vd1[]) const final {}
    double getCrossFirstDerivative(int iout, int i1d, int iv1d) const final { return 0.; }

    void getCrossSecondDerivative(double d2vd1[]) const final {}
    void getCrossSecondDerivative(int iout, double d2vd1[]) const final {}
    double getCrossSecondDerivative(int iout, int i2d, int iv1d) const final { return 0.; }
};


#endif
