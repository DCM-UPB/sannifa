#include <iostream>
#include <assert.h>
#include <cmath>
#include <vector>

#include "sannifa/Sannifa.hpp"

void checkDerivatives(Sannifa * ann, const double &TINY)
{
    using namespace std;

    double x[2] = {1.7, -0.2};
    const double dx = 0.0005;

    ann->evaluate(x);
    double fx = ann->getOutput(0);
    double fy = ann->getOutput(1);
    //cout << "fx = " << fx << endl;
    //cout << "fy = " << fy << endl;


    // --- first and second derivative in respect to first input

    if (ann->hasFirstDerivative() && ann->hasSecondDerivative()) {
        double x1[2];

        ann->evaluate(x, true);
        double anal_dfxdx = ann->getFirstDerivative(0, 0);
        double anal_dfydx = ann->getFirstDerivative(1, 0);
        double anal_d2fxdx2 = ann->getSecondDerivative(0, 0);
        double anal_d2fydx2 = ann->getSecondDerivative(1, 0);

        x1[0] = x[0] + dx;
        x1[1] = x[1];
        ann->evaluate(x1);
        double fx1 = ann->getOutput(0);
        double fy1 = ann->getOutput(1);
        //cout << "fx1 = " << fx1 << endl;
        //cout << "fy1 = " << fy1 << endl;

        x1[0] = x[0] - dx;
        x1[1] = x[1];
        ann->evaluate(x1);
        double fxm1 = ann->getOutput(0);
        double fym1 = ann->getOutput(1);
        //cout << "fxm1 = " << fxm1 << endl;
        //cout << "fym1 = " << fym1 << endl;
        //cout << endl;

        double num_dfxdx = (fx1-fx)/dx;
        double num_dfydx = (fy1-fy)/dx;

        //cout << "anal_dfxdx = " << anal_dfxdx << endl;
        //cout << "num_dfxdx = " << num_dfxdx << endl;
        //cout << endl;
        assert(abs( (anal_dfxdx-num_dfxdx) ) < TINY);

        //cout << "anal_dfydx = " << anal_dfydx << endl;
        //cout << "num_dfydx = " << num_dfydx << endl;
        //cout << endl;
        assert(abs( (anal_dfydx-num_dfydx) ) < TINY);

        double num_d2fxdx2 = (fx1-2.*fx+fxm1)/(dx*dx);
        double num_d2fydx2 = (fy1-2.*fy+fym1)/(dx*dx);

        //cout << "anal_d2fxdx2 = " << anal_d2fxdx2 << endl;
        //cout << "num_d2fxdx2 = " << num_d2fxdx2 << endl;
        //cout << endl;
        assert(abs( (anal_d2fxdx2-num_d2fxdx2) ) < TINY);

        //cout << "anal_d2fydx2 = " << anal_d2fydx2 << endl;
        //cout << "num_d2fydx2 = " << num_d2fydx2 << endl;
        //cout << endl;
        assert(abs( (anal_d2fydx2-num_d2fydx2) ) < TINY);


        // --- first and second derivative in respect to second input

        ann->evaluate(x, true);
        anal_dfxdx = ann->getFirstDerivative(0, 1);
        anal_dfydx = ann->getFirstDerivative(1, 1);
        anal_d2fxdx2 = ann->getSecondDerivative(0, 1);
        anal_d2fydx2 = ann->getSecondDerivative(1, 1);

        x1[0] = x[0];
        x1[1] = x[1] + dx;
        ann->evaluate(x1);
        fx1 = ann->getOutput(0);
        fy1 = ann->getOutput(1);
        //cout << "fx1 = " << fx1 << endl;
        //cout << "fy1 = " << fy1 << endl;

        x1[0] = x[0];
        x1[1] = x[1] - dx;
        ann->evaluate(x1);
        fxm1 = ann->getOutput(0);
        fym1 = ann->getOutput(1);
        //cout << "fxm1 = " << fxm1 << endl;
        //cout << "fym1 = " << fym1 << endl;
        //cout << endl;

        num_dfxdx = (fx1-fx)/dx;
        num_dfydx = (fy1-fy)/dx;

        //cout << "anal_dfxdx = " << anal_dfxdx << endl;
        //cout << "num_dfxdx = " << num_dfxdx << endl;
        //cout << endl;
        assert(abs( (anal_dfxdx-num_dfxdx) ) < TINY);

        //cout << "anal_dfydx = " << anal_dfydx << endl;
        //cout << "num_dfydx = " << num_dfydx << endl;
        //cout << endl;
        assert(abs( (anal_dfydx-num_dfydx) ) < TINY);

        num_d2fxdx2 = (fx1-2.*fx+fxm1)/(dx*dx);
        num_d2fydx2 = (fy1-2.*fy+fym1)/(dx*dx);

        //cout << "anal_d2fxdx2 = " << anal_d2fxdx2 << endl;
        //cout << "num_d2fxdx2 = " << num_d2fxdx2 << endl;
        //cout << endl;
        assert(abs( (anal_d2fxdx2-num_d2fxdx2) ) < TINY);

        //cout << "anal_d2fydx2 = " << anal_d2fydx2 << endl;
        //cout << "num_d2fydx2 = " << num_d2fydx2 << endl;
        //cout << endl;
        assert(abs( (anal_d2fydx2-num_d2fydx2) ) < TINY);
    }

    // --- variational derivative

    std::vector<double> anal_dfxdbeta(ann->getNVariationalParameters());
    std::vector<double> anal_dfydbeta(ann->getNVariationalParameters());

    if (ann->hasVariationalFirstDerivative()) {
        ann->evaluate(x, true);
        for (int i=0; i<ann->getNVariationalParameters(); ++i){
            anal_dfxdbeta[i] = ann->getVariationalFirstDerivative(0, i);
            anal_dfydbeta[i] = ann->getVariationalFirstDerivative(1, i);
        }

        for (int i=0; i<ann->getNVariationalParameters(); ++i){
            const double orig_vp = ann->getVariationalParameter(i);
            ann->setVariationalParameter(i, orig_vp+dx);
            ann->evaluate(x);
            const double fx1 = ann->getOutput(0);
            const double fy1 = ann->getOutput(1);

            const double num_dfxdbeta = (fx1-fx)/dx;
            const double num_dfydbeta = (fy1-fy)/dx;

            //cout << "i_beta = " << i << endl;
            //cout << "anal_dfxdbeta = " << anal_dfxdbeta[i] << endl;
            //cout << "num_dfxdbeta = " << num_dfxdbeta << endl;
            //cout << endl;
            assert( abs( (anal_dfxdbeta[i]-num_dfxdbeta) ) < TINY);

            //cout << "anal_dfydbeta = " << anal_dfydbeta[i] << endl;
            //cout << "num_dfydbeta = " << num_dfydbeta << endl;
            //cout << endl;
            assert( abs( (anal_dfydbeta[i]-num_dfydbeta) ) < TINY);

            ann->setVariationalParameter(i, orig_vp);
        }
    }


    // --- cross first derivatives
/*
    double ** anal_dfxdxdbeta = new double*[ann->getNInput()];
    for (int i=0; i<ann->getNInput(); ++i){
        anal_dfxdxdbeta[i] = new double[ann->getNVariationalParameters()];
    }
    double ** anal_dfydxdbeta = new double*[ann->getNInput()];
    for (int i=0; i<ann->getNInput(); ++i){
        anal_dfydxdbeta[i] = new double[ann->getNVariationalParameters()];
    }

    if (ann->hasCrossFirstDerivative()) {
        ann->evaluate(x, true);
        ann->getCrossFirstDerivative(0, anal_dfxdxdbeta);
        ann->getCrossFirstDerivative(1, anal_dfydxdbeta);

        for (int i1d=0; i1d<ann->getNInput(); ++i1d){
            for (int iv1d=0; iv1d<ann->getNVariationalParameters(); ++iv1d){
                const double orig_x = x[i1d];
                const double orig_vp = ann->getVariationalParameter(iv1d);

                ann->setInput(i1d, orig_x);
                ann->setVariationalParameter(iv1d, orig_vp+dx);
                ann->FFPropagate();
                const double fxdbeta = ann->getOutput(0);
                const double fydbeta = ann->getOutput(1);

                ann->setInput(i1d, orig_x+dx);
                ann->setVariationalParameter(iv1d, orig_vp);
                ann->FFPropagate();
                const double fxdx = ann->getOutput(0);
                const double fydx = ann->getOutput(1);

                ann->setInput(i1d, orig_x+dx);
                ann->setVariationalParameter(iv1d, orig_vp+dx);
                ann->FFPropagate();
                const double fxdxdbeta = ann->getOutput(0);
                const double fydxdbeta = ann->getOutput(1);

                const double num_dfxdxdbeta = (fxdxdbeta - fxdx - fxdbeta + fx)/(dx*dx);
                const double num_dfydxdbeta = (fydxdbeta - fydx - fydbeta + fy)/(dx*dx);

                // //cout << "anal_dfxdxdbeta[" << i1d << "][" << iv1d << "]    " << anal_dfxdxdbeta[i1d][iv1d] << endl;
                // //cout << " --- > num_dfxdxdbeta    " << num_dfxdxdbeta << endl << endl;
                assert(abs( (anal_dfxdxdbeta[i1d][iv1d]-num_dfxdxdbeta) ) < TINY);

                // //cout << "anal_dfydxdbeta[" << i1d << "][" << iv1d << "]    " << anal_dfydxdbeta[i1d][iv1d] << endl;
                // //cout << " --- > num_dfydxdbeta    " << num_dfydxdbeta << endl << endl;
                assert(abs( (anal_dfydxdbeta[i1d][iv1d]-num_dfydxdbeta) ) < TINY);

                ann->setInput(i1d, orig_x);
                ann->setVariationalParameter(iv1d, orig_vp);
            }
        }
    }

    // --- cross second derivatives

    double ** anal_dfxdx2dbeta = new double*[ann->getNInput()];
    for (int i=0; i<ann->getNInput(); ++i){
        anal_dfxdx2dbeta[i] = new double[ann->getNVariationalParameters()];
    }
    double ** anal_dfydx2dbeta = new double*[ann->getNInput()];
    for (int i=0; i<ann->getNInput(); ++i){
        anal_dfydx2dbeta[i] = new double[ann->getNVariationalParameters()];
    }

    if (ann->hasCrossSecondDerivative()) {
        ann->setInput(x);
        ann->FFPropagate();
        ann->getCrossSecondDerivative(0, anal_dfxdx2dbeta);
        ann->getCrossSecondDerivative(1, anal_dfydx2dbeta);

        for (int i2d=0; i2d<ann->getNInput(); ++i2d){
            for (int iv1d=0; iv1d<ann->getNVariationalParameters(); ++iv1d){
                const double orig_x = x[i2d];
                const double orig_vp = ann->getVariationalParameter(iv1d);

                ann->setInput(i2d, orig_x+dx);
                ann->setVariationalParameter(iv1d, orig_vp+dx);
                ann->FFPropagate();
                const double fxdxdbeta = ann->getOutput(0);
                const double fydxdbeta = ann->getOutput(1);

                ann->setInput(i2d, orig_x);
                ann->setVariationalParameter(iv1d, orig_vp+dx);
                ann->FFPropagate();
                const double fxdbeta = ann->getOutput(0);
                const double fydbeta = ann->getOutput(1);

                ann->setInput(i2d, orig_x-dx);
                ann->setVariationalParameter(iv1d, orig_vp+dx);
                ann->FFPropagate();
                const double fxmdxdbeta = ann->getOutput(0);
                const double fymdxdbeta = ann->getOutput(1);

                ann->setInput(i2d, orig_x+dx);
                ann->setVariationalParameter(iv1d, orig_vp);
                ann->FFPropagate();
                const double fxdx = ann->getOutput(0);
                const double fydx = ann->getOutput(1);

                ann->setInput(i2d, orig_x-dx);
                ann->setVariationalParameter(iv1d, orig_vp);
                ann->FFPropagate();
                const double fxmdx = ann->getOutput(0);
                const double fymdx = ann->getOutput(1);

                const double num_dfxdx2dbeta = (fxdxdbeta - 2.*fxdbeta + fxmdxdbeta - fxdx + 2.*fx - fxmdx)/(dx*dx*dx);
                const double num_dfydx2dbeta = (fydxdbeta - 2.*fydbeta + fymdxdbeta - fydx + 2.*fy - fymdx)/(dx*dx*dx);

                // //cout << "anal_dfx2dxdbeta[" << i2d << "][" << iv1d << "]    " << anal_dfxdx2dbeta[i2d][iv1d] << endl;
                // //cout << " --- > num_dfxdx2dbeta    " << num_dfxdx2dbeta << endl << endl;
                assert(abs( (anal_dfxdx2dbeta[i2d][iv1d]-num_dfxdx2dbeta) ) < TINY);

                // //cout << "anal_dfydx2dbeta[" << i2d << "][" << iv1d << "]    " << anal_dfydx2dbeta[i2d][iv1d] << endl;
                // //cout << " --- > num_dfydx2dbeta    " << num_dfydx2dbeta << endl << endl;
                assert(abs( (anal_dfydx2dbeta[i2d][iv1d]-num_dfydx2dbeta) ) < TINY);

                ann->setInput(i2d, orig_x);
                ann->setVariationalParameter(iv1d, orig_vp);
            }
        }
    }

    // free resources
    for (int i=0; i<ann->getNInput(); ++i){
        delete[] anal_dfxdxdbeta[i];
        delete[] anal_dfydxdbeta[i];
        delete[] anal_dfxdx2dbeta[i];
        delete[] anal_dfydx2dbeta[i];
    }
    delete[] anal_dfxdxdbeta;
    delete[] anal_dfydxdbeta;
    delete[] anal_dfxdx2dbeta;
    delete[] anal_dfydx2dbeta;
*/

}
