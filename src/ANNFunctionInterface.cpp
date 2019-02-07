#include "sannifa/ANNFunctionInterface.hpp"

#include <iostream>

void ANNFunctionInterface::printInfo(const bool verbose)
{   // verbose flag can be used in child overrides (default false!)
    using namespace std;

    cout << "--------------------" << endl;
    cout << "ANN-Function Summary" << endl;
    cout << "--------------------" << endl;
    cout << "Backend: " << this->getLibName() << endl;
    cout << endl;
    cout << "NInputs  : " << this->getNInput() << endl;
    cout << "NOutputs : " << this->getNOutput() << endl;
    cout << "NParams  : " << this->getNVariationalParameters() << endl;
    cout << endl;
    cout << "Derivatives:" << endl;
    cout << "  d1:  " << (this->hasFirstDerivative() ? "yes" : "no") << endl;
    cout << "  d2:  " << (this->hasSecondDerivative() ? "yes" : "no") << endl;
    cout << "  vd1: " << (this->hasVariationalFirstDerivative() ? "yes" : "no") << endl;
    cout << "  cd1: " << (this->hasCrossFirstDerivative() ? "yes" : "no") << endl;
    cout << "  cd2: " << (this->hasCrossSecondDerivative() ? "yes" : "no") << endl;
}

bool ANNFunctionInterface::hasDerivatives()
{
    return (_dopt.d1 || _dopt.d2 || _dopt.vd1 || _dopt.cd1 || _dopt.cd2);
}

bool ANNFunctionInterface::hasInputDerivatives()
{
    return (_dopt.d1 || _dopt.d2);
}

bool ANNFunctionInterface::hasVariationalDerivatives()
{
    return (_dopt.vd1 || _dopt.cd1 || _dopt.cd2);
}

void ANNFunctionInterface::enableDerivatives(const DerivativeOptions &doptToEnable)
{
    if (doptToEnable.d1) this->enableFirstDerivative();
    if (doptToEnable.d2) this->enableSecondDerivative();
    if (doptToEnable.vd1) this->enableVariationalFirstDerivative();
    if (doptToEnable.cd1) this->enableCrossFirstDerivative();
    if (doptToEnable.cd2) this->enableCrossSecondDerivative();
}
