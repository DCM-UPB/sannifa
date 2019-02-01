#include "sannifa/ANNInterface.hpp"

void ANNInterface::evaluate(const double * in, double * out, double ** d1, double ** d2, double ** vd1){
    this->setInput(in);
    this->propagate();
    this->getOutput(out);

    if (this->hasFirstDerivative() && d1!=NULL) {
        this->getFirstDerivative(d1);
    }
    if (this->hasSecondDerivative() && d2!=NULL) {
        this->getSecondDerivative(d2);
    }
    if (this->hasVariationalFirstDerivative() && vd1!=NULL) {
        this->getVariationalFirstDerivative(vd1);
    }
}

void ANNInterface::enableDerivatives(const bool flag_d1, const bool flag_d2, const bool flag_vd1, const bool flag_c1d, const bool flag_c2d)
{
    if (flag_d1) this->enableFirstDerivative();
    if (flag_d2) this->enableSecondDerivative();
    if (flag_vd1) this->enableVariationalFirstDerivative();
    if (flag_c1d) this->enableCrossFirstDerivative();
    if (flag_c2d) this->enableCrossSecondDerivative();
}
