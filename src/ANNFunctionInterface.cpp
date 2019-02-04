#include "sannifa/ANNFunctionInterface.hpp"

bool ANNFunctionInterface::hasDerivatives()
{
    return (_dopt.d1 || _dopt.d2 || _dopt.v1d || _dopt.c1d || _dopt.c2d);
}

void ANNFunctionInterface::enableDerivatives(const DerivativeOptions &doptToEnable)
{
    if (doptToEnable.d1) this->enableFirstDerivative();
    if (doptToEnable.d2) this->enableSecondDerivative();
    if (doptToEnable.v1d) this->enableVariationalFirstDerivative();
    if (doptToEnable.c1d) this->enableCrossFirstDerivative();
    if (doptToEnable.c2d) this->enableCrossSecondDerivative();
}
