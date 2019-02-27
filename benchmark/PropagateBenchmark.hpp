#ifndef PROPAGATE_BENCHMARK
#define PROPAGATE_BENCHMARK

#include <exception>
#include <array>
#include <algorithm>
#include <functional>

#include "sannifa/ANNFunctionInterface.hpp"

#include "Timer.hpp"

double propagateBenchmark(ANNFunctionInterface * ann, const int nsteps, const bool flag_deriv = false)
{   // sample time per evaluation (forward + backward, if flag_deriv)
    Timer * timer = new Timer();
    double time;

    const int ninput = ann->getNInput();
    double inputv[ninput];
    
    timer->reset();
    for (int i=0; i<nsteps; ++i) {
        for (int j=0; j<ninput; ++j) {
            inputv[j] = rand()*(1.0 / RAND_MAX);
        }
        ann->evaluate(inputv, flag_deriv);
    }
    
    time = timer->elapsed();
    delete timer;
    
    return time;
}


std::array<double, 6> benchmarkDerivatives(ANNFunctionInterface * ann1, ANNFunctionInterface * ann2, const std::array<int, 6> &nsteps, const bool verbose = false)
{   // ! ann1 and ann2 should be identical (but different objects) !
    using namespace std;
    // expects an ann without any derivatives enabled
    if (ann1->hasDerivatives() || ann2->hasDerivatives()) {
        throw invalid_argument("To be benchmarked ANN should be without any enabled derivatives.");
    }

    array<double, 6> times;
    
    // no derivatives
    if (verbose) cout << endl << "Running benchmark (no deriv)..." << endl;
    times[0] = propagateBenchmark(ann1, nsteps[0], false);
    if (verbose) cout << "Done. (" << times[0] << " s / " << nsteps[0] << " steps)" << endl;

    // 1st deriv
    ann1->enableFirstDerivative();
    if (verbose) cout << endl << "Running benchmark (d1)..." << endl;
    times[1] = propagateBenchmark(ann1, nsteps[1], true);
    if (verbose) cout << "Done. (" << times[1] << " s / " << nsteps[1] << " steps)" << endl;

    // 1st+2nd deriv
    ann1->enableSecondDerivative();
    if (verbose) cout << endl << "Running benchmark (d1+d2)..." << endl;
    times[2] = propagateBenchmark(ann1, nsteps[2], true);
    if (verbose) cout << "Done. (" << times[2] << " s / " << nsteps[2] << " steps)" << endl;

    // 1st param deriv
    ann2->enableVariationalFirstDerivative();
    if (verbose) cout << endl << "Running benchmark (vd1)..." << endl;
    times[3] = propagateBenchmark(ann2, nsteps[3], true);
    if (verbose) cout << "Done. (" << times[3] << " s / " << nsteps[3] << " steps)" << endl;

    // 1st deriv + 1st param deriv
    ann2->enableFirstDerivative();
    if (verbose) cout << endl << "Running benchmark (d1+vd1)..." << endl;
    times[4] = propagateBenchmark(ann2, nsteps[4], true);
    if (verbose) cout << "Done. (" << times[4] << " s / " << nsteps[4] << " steps)" << endl;

    // 1st+2nd + 1st param deriv
    ann2->enableSecondDerivative();
    if (verbose) cout << endl << "Running benchmark (d1+d2+vd1)..." << endl;
    times[5] = propagateBenchmark(ann2, nsteps[5], true);
    if (verbose) cout << "Done. (" << times[5] << " s / " << nsteps[5] << " steps)" << endl;

    return times;
}

void reportTimes(std::array<double, 6> &times, std::array<int, 6> &nsteps) {
    using namespace std;

    std::array<double, 6> evalTimes;
    for (int i=0; i<6; ++i) evalTimes[i] = times[i]/nsteps[i]*1000.;

    cout << endl;
    cout << "Time per Evaluation" << endl;
    cout << "-------------------" << endl;
    cout << "noderiv  : " << evalTimes[0] << " ms" << endl;
    cout << "d1       : " << evalTimes[1] << " ms" << endl;
    cout << "d1+d2    : " << evalTimes[2] << " ms" << endl;
    cout << "vd1      : " << evalTimes[3] << " ms" << endl;
    cout << "d1+vd1   : " << evalTimes[4] << " ms" << endl;
    cout << "d1+d2+vd1: " << evalTimes[5] << " ms" << endl;
    cout << endl;
}

#endif
