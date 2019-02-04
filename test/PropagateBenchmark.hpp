#ifndef PROPAGATE_BENCHMARK
#define PROPAGATE_BENCHMARK
 
double propagateBenchmark(ANNFunctionInterface * ann, const int nsteps)
{
   const int ninput = ann->getNInput();
   double inputv[ninput];
   for (int i=0; i<nsteps; ++i) {
       for (int j=0; j<ninput; ++j) {
           inputv[j] = rand();
       }
       ann->evaluate(inputv);
    }

    return 0.;
}

#endif
