#include "sannifa/Sannifa.hpp"

#include <iostream>

void Sannifa::printInfo(const bool verbose) const
{   // verbose flag can be used in child overrides (default false!)
    using namespace std;

    cout << "--------------------" << endl;
    cout << "ANN-Function Summary" << endl;
    cout << "--------------------" << endl;
    cout << "Backend: " << this->getLibName() << endl;
    cout << endl;
    cout << "NInputs (orig) : " << this->getNOrigInput() << endl;
    cout << "NInputs (net)  : " << this->getNInput() << endl;
    cout << "NOutputs       : " << this->getNOutput() << endl;
    cout << "NParams        : " << this->getNVariationalParameters() << endl;
    cout << endl;
    cout << "Derivatives:" << endl;
    cout << "  d1:  " << (this->hasFirstDerivative() ? "yes" : "no") << endl;
    cout << "  d2:  " << (this->hasSecondDerivative() ? "yes" : "no") << endl;
    cout << "  vd1: " << (this->hasVariationalFirstDerivative() ? "yes" : "no") << endl;
    cout << "  cd1: " << (this->hasCrossFirstDerivative() ? "yes" : "no") << endl;
    cout << "  cd2: " << (this->hasCrossSecondDerivative() ? "yes" : "no") << endl;
}

bool Sannifa::hasDerivatives() const
{
    return (_dopt.d1 || _dopt.d2 || _dopt.vd1 || _dopt.cd1 || _dopt.cd2);
}

bool Sannifa::hasInputDerivatives() const
{
    return (_dopt.d1 || _dopt.d2);
}

bool Sannifa::hasVariationalDerivatives() const
{
    return (_dopt.vd1 || _dopt.cd1 || _dopt.cd2);
}

void Sannifa::enableDerivatives(const DerivativeOptions &doptToEnable)
{
    if (doptToEnable.d1) this->enableFirstDerivative();
    if (doptToEnable.d2) this->enableSecondDerivative();
    if (doptToEnable.vd1) this->enableVariationalFirstDerivative();
    if (doptToEnable.cd1) this->enableCrossFirstDerivative();
    if (doptToEnable.cd2) this->enableCrossSecondDerivative();
}

void Sannifa::enableFirstDerivative() // coordinates first derivatives
{
    this->_enableFirstDerivative();
    _dopt.d1 = true;
}

void Sannifa::enableSecondDerivative() // coordinates second derivatives
{
    this->_enableSecondDerivative();
    _dopt.d2 = true;
}

void Sannifa::enableVariationalFirstDerivative() // parameter first derivatives
{
    this->_enableVariationalFirstDerivative();
    _dopt.vd1 = true;
}

void Sannifa::enableCrossFirstDerivative() // parameters first coordinates first derivatives
{
    this->_enableCrossFirstDerivative();
    _dopt.cd1 = true;
}

void Sannifa::enableCrossSecondDerivative() // parameters first coordinates second derivatives
{
    this->_enableCrossSecondDerivative();
    _dopt.cd2 = true;
}
