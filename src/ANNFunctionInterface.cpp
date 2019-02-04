#include "sannifa/ANNFunctionInterface.hpp"

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
