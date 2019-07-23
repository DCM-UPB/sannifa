#include <iostream>
#include <random>
#include <array>
#include <string>

#include "sannifa/QTemplWrapper.hpp"
#include "qnets/actf/Sigmoid.hpp"

#include "PropagateBenchmark.hpp"

int main() {
    using namespace std;

    using namespace templ;

    // Setup

    const int yndim = 1;
    constexpr int xndim[3] = {6, 24, 96}, nhu1[3] = {12, 48, 192}, nhu2[3] = {12, 48, 192};

    constexpr auto dconf = DerivConfig::D12_VD1; // "allocate" for all derivatives
    constexpr DynamicDFlags dflags0(DerivConfig::OFF); // start with all derivs off

    using RealT = double;

    // Small Net
    using L1Type_s = LayerConfig<nhu1[0], actf::Sigmoid>;
    using L2Type_s = LayerConfig<nhu2[0], actf::Sigmoid>;
    using L3Type_s = LayerConfig<yndim, actf::Sigmoid>;
    using NetType_s = TemplNet<RealT, dconf, xndim[0], L1Type_s, L2Type_s, L3Type_s>;
    QTemplWrapper<NetType_s> wrapper_s(dflags0);

    // Medium Net
    using L1Type_m = LayerConfig<nhu1[1], actf::Sigmoid>;
    using L2Type_m = LayerConfig<nhu2[1], actf::Sigmoid>;
    using L3Type_m = LayerConfig<yndim, actf::Sigmoid>;
    using NetType_m = TemplNet<RealT, dconf, xndim[1], L1Type_m, L2Type_m, L3Type_m>;
    QTemplWrapper<NetType_m> wrapper_m(dflags0);

    // Large Net
    using L1Type_l = LayerConfig<nhu1[2], actf::Sigmoid>;
    using L2Type_l = LayerConfig<nhu2[2], actf::Sigmoid>;
    using L3Type_l = LayerConfig<yndim, actf::Sigmoid>;
    using NetType_l = TemplNet<RealT, dconf, xndim[2], L1Type_l, L2Type_l, L3Type_l>;
    QTemplWrapper<NetType_l> wrapper_l(dflags0);

    // generate some random betas
    random_device rdev;
    mt19937_64 rgen;
    uniform_real_distribution<double> rd;
    rgen = mt19937_64(rdev());
    rgen.seed(18984687);
    rd = uniform_real_distribution<double>(-sqrt(3.), sqrt(3.)); // uniform with variance 1

    for (int i=0; i<wrapper_s.getNVariationalParameters(); ++i) {
        wrapper_s.setVariationalParameter(i, rd(rgen));
    }
    for (int i=0; i<wrapper_m.getNVariationalParameters(); ++i) {
        wrapper_m.setVariationalParameter(i, rd(rgen));
    }
    for (int i=0; i<wrapper_l.getNVariationalParameters(); ++i) {
        wrapper_l.setVariationalParameter(i, rd(rgen));
    }

    // make identical copies of the wrappers (both copies are used later)
    QTemplWrapper<NetType_s> wrapper2_s(*wrapper_s.nn);
    QTemplWrapper<NetType_m> wrapper2_m(*wrapper_m.nn);
    QTemplWrapper<NetType_l> wrapper2_l(*wrapper_l.nn);

    std::array<Sannifa *, 3> nnList{&wrapper_s, &wrapper_m, &wrapper_l};
    std::array<Sannifa *, 3> nnList2{&wrapper2_s, &wrapper2_m, &wrapper2_l};

    std::array<array<int, 6>, 3> nstepsList {
        std::array<int, 6>({15000000, 10000000, 3000000, 10000000, 10000000, 3000000}),
        std::array<int, 6>({2000000, 1500000, 175000, 1000000, 1000000, 150000}),
        std::array<int, 6>({100000, 90000, 2500, 75000, 75000, 2500})
    };

/*
    // fast mode
    for (size_t i=0; i<nstepsList.size(); ++i) {
        for (size_t j=0; j<nstepsList[i].size(); ++j) {
            nstepsList[i][j] /= 10;
        }
    }
*/

    std::array<std::string, 3> nameList {
        "small (6x12x12x1)",
        "medium (24x48x48x1)",
        "large (96x192x192x1)"
    };

    cout << endl << "--- Benchmark with libqnets/templ backend ---" << endl;
    for (int i=0; i<3; ++i) {
        cout << endl << "Benchmarking " << nameList[i] << " FFNN with "
             << nnList[i]->getNVariationalParameters() << " weights..." << endl;

        // benchmark
        nnList[i]->printInfo(true);
        auto times = benchmarkDerivatives(nnList[i], nnList2[i], nstepsList[i], true);
    
        reportTimes(times, nstepsList[i]);
    }
}
