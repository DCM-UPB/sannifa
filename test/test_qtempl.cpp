#include <iostream>
#include <random>
#include <array>

#include "sannifa/QTemplWrapper.hpp"
#include "qnets/actf/Sigmoid.hpp"

#include "checkDerivatives.hpp"

int main() {
    using namespace std;
    using namespace templ;

    constexpr auto dconf = DerivConfig::D12_VD1; // "allocate" for derivatives
    using RealT = double;

    // Test Net (same dimensions as other tests)
    using L1Type = LayerConfig<4, actf::Sigmoid>;
    using L2Type = LayerConfig<4, actf::Sigmoid>;
    using L3Type = LayerConfig<2, actf::Sigmoid>;
    using NetType = TemplNet<RealT, dconf, 2, 2, L1Type, L2Type, L3Type>;
    QTemplWrapper<NetType> wrapper; // allocated derivs are enabled per default

    // generate some random betas
    random_device rdev;
    mt19937_64 rgen;
    uniform_real_distribution<double> rd;
    rgen = mt19937_64(rdev());
    rgen.seed(18984687);
    rd = uniform_real_distribution<double>(-sqrt(3.), sqrt(3.)); // uniform with variance 1

    for (int i=0; i<wrapper.getNVariationalParameters(); ++i) {
        wrapper.setVariationalParameter(i, rd(rgen));
    }


    // derivative check
    
    wrapper.printInfo(true);
    cout << endl << "Derivative check..." << endl;
    checkDerivatives(&wrapper, 0.0001);
    cout << "Passed." << endl;

    //wrapper.saveToFile("ffnn.out");
    //QPolyWrapper wrapper2("ffnn.out");
    //wrapper2.printInfo(true);
}
