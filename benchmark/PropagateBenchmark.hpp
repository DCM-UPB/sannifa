#ifndef PROPAGATE_BENCHMARK
#define PROPAGATE_BENCHMARK

#include <exception>

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
            inputv[j] = rand();
        }
        ann->evaluate(inputv, flag_deriv);
    }
    
    time = timer->elapsed();
    delete timer;
    
    return time/nsteps;
}


std::vector<double> benchmarkDerivatives(ANNFunctionInterface * ann, const std::vector<int> &nstepsVec, const bool verbose = false)
{   
    using namespace std;
    // expects an ann without any derivatives enabled
    if (ann->hasDerivatives()) {
        throw invalid_argument("To be benchmarked ANN should be without any enabled derivatives.");
    }

    vector<double> times;
    
    // no derivatives
    if (verbose) cout << endl << "Running benchmark (no deriv)..." << endl;
    times.push_back(propagateBenchmark(ann, nstepsVec[0], false));
    if (verbose) cout << "Done." << endl;

    // 1st deriv
    ann->enableFirstDerivative();
    if (verbose) cout << endl << "Running benchmark (d1)..." << endl;
    times.push_back(propagateBenchmark(ann, nstepsVec[1], true));
    if (verbose) cout << "Done." << endl;

    // 1st+2nd deriv
    ann->enableSecondDerivative();
    if (verbose) cout << endl << "Running benchmark (d1+d2)..." << endl;
    times.push_back(propagateBenchmark(ann, nstepsVec[2], true));
    if (verbose) cout << "Done." << endl;

    // 1st+2nd + 1st param deriv
    ann->enableVariationalFirstDerivative();
    if (verbose) cout << endl << "Running benchmark (d1+d2+vd1)..." << endl;
    times.push_back(propagateBenchmark(ann, nstepsVec[3], true));
    if (verbose) cout << "Done." << endl;

    return times;
}

void reportTimes(std::vector<double> &times) {
    using namespace std;

    cout << "Time per Evaluation" << endl;
    cout << "-------------------" << endl;
    cout << "noderiv  : " << times[0]*1000 << " ms" << endl;
    cout << "d1       : " << times[1]*1000 << " ms" << endl;
    cout << "d1+d2    : " << times[2]*1000 << " ms" << endl;
    cout << "d1+d2+vd1: " << times[3]*1000 << " ms" << endl;
}

#endif
