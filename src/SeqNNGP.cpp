#include "SeqNNGP.h"
#include "tree.h"
#include "utils.h"

#include <chrono>
#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace pyNNGP {
    SeqNNGP::SeqNNGP(const double* y, const double* X, const double* coords,
        int p, int n, int nNeighbors, CovModel& cm, const double tausqr) :
        _p(p), _n(n), _m(nNeighbors), _nIndx(_m*(_m+1)/2+(_n-_m-1)*_m),
        _y(y, _n), _X(X, _n, _p), _coords(coords, _n, _p),
        _XtX(_X.transpose()*_X),
        _cm(cm), _tausqr(tausqr),
        _gen(_rd())
        {
            // build the neighbor index
            nnIndx.resize(_nIndx);
            nnDist.resize(_nIndx);
            nnIndxLU.resize(2*_n);
            _w.resize(_n);

            std::cout << "Finding neighbors" << '\n';
            auto start = std::chrono::high_resolution_clock::now();
            mkNNIndxTree0(_n, _m, _coords.data(), &nnIndx[0], &nnDist[0], &nnIndxLU[0]);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff = end-start;
            std::cout << "duration = " << diff.count() << "s" << '\n';

            std::cout << "Building neighbors of neighbors index" << '\n';
            start = std::chrono::high_resolution_clock::now();
            mkUIndx();
            mkUIIndx();
            end = std::chrono::high_resolution_clock::now();
            diff = end-start;
            std::cout << "duration = " << diff.count() << "s" << '\n';

            _B.resize(_nIndx);
            _F.resize(_n);
            _BCand.resize(_nIndx);
            _FCand.resize(_n);
            _c.resize(_nIndx);
            std::cout << "Making CD" << '\n';
            start = std::chrono::high_resolution_clock::now();
            mkCD();
            end = std::chrono::high_resolution_clock::now();
            diff = end-start;
            std::cout << "duration = " << diff.count() << "s" << '\n';

            // ExponentialCovModel covModel{5.0, 6.0};
            std::cout << "updating BF" << '\n';
            start = std::chrono::high_resolution_clock::now();
            updateBF();
            end = std::chrono::high_resolution_clock::now();
            diff = end-start;
            std::cout << "duration = " << diff.count() << "s" << '\n';

            // std::cout << "\n\n\nB = " << '\n';
            // for(int ii=0; ii<_nIndx; ii++) {
            //     std::cout << "  " << _B[ii] << '\n';
            // }
            //
            // std::cout << "\n\n\nF = " << '\n';
            // for(int ii=0; ii<_n; ii++) {
            //     std::cout << "  " << _F[ii] << '\n';
            // }
            // std::cout << "\n\n\n" << '\n';

            // For now...
            _beta = VectorXd(2);
            _beta << 2.346582, 1.888253;

            _tauSqIGa = 2.0;
            _tauSqIGb = 1.0;
            _sigmaSqrIGa = 2.0;
            _sigmaSqrIGb = 5.0;
            _sigmaSqr = 1.0;

            int nSamples=1;
            for(int s=0; s<nSamples; s++){
                updateW();
                updateBeta();
                updateTauSqr();
                updateSigmaSqr();
                updateTheta();
            }
        }

    void SeqNNGP::mkUIndx() {
        uIndx.resize(_nIndx);
        uIndxLU.resize(2*_n);

        for(int i=0, ell=0; i<_n; i++) {
            uIndxLU[i] = ell;
            int h=0;
            for(int j=0; j<_n; j++) {
                int iNNIndx, iNN;
                getNNIndx(j, _m, iNNIndx, iNN);
                // Go through each neighbor of j
                for(int k=0; k<iNN; k++) {
                    if(nnIndx[iNNIndx+k] == i) {
                        uIndx[ell+h] = j;
                        h++;
                    }
                }
            }
            ell += h;
            uIndxLU[_n+i] = h;
        }
    }

    void SeqNNGP::mkUIIndx() {
        uiIndx.resize(_nIndx);
        for(int i=0; i<_n; i++){
            for(int j=0; j<uIndxLU[_n+i]; j++) { //for each location that has i as a neighbor
                //index of a location that has i as a neighbor
                int k = uIndx[uIndxLU[i]+j];
                uiIndx[uIndxLU[i]+j] = which(i, &nnIndx[nnIndxLU[k]], nnIndxLU[_n+k]);
            }
        }
    }

    void SeqNNGP::mkCD() {
        // What is C?  Something to do with covariance...
        CIndx.resize(2*_n);
        int j=0;
        for(int i=0; i<_n; i++){ //zero should never be accessed
            j += nnIndxLU[_n+i]*nnIndxLU[_n+i];
            if(i == 0) {
                CIndx[_n+i] = 0;
                CIndx[i] = 0;
            } else {
                CIndx[_n+i] = nnIndxLU[_n+i]*nnIndxLU[_n+i]; // # of neighbors squared
                CIndx[i] = CIndx[_n+i-1] + CIndx[i-1]; // cumulative sum of above...
            }
        }

        _C.resize(j);
        _D.resize(j);
        for(int i=0; i<_n; i++) {
            for(int k=0; k<nnIndxLU[_n+i]; k++) {
                for(int ell=0; ell<=k; ell++) {
                    int i1 = nnIndx[nnIndxLU[i]+k];
                    int i2 = nnIndx[nnIndxLU[i]+ell];
                    _D[CIndx[i]+ell*nnIndxLU[_n+i]+k] = dist2(_coords.row(i1), _coords.row(i2));
                }
            }
        }
    }

    // Modified to ignore Matern covariance until I can figure out a way to get
    // a cyl_bessel_k compiled.
    void SeqNNGP::updateBF() {
        int k, ell;
        Eigen::Map<VectorXd> eigenB(&_B[0], _nIndx);
        Eigen::Map<VectorXd> eigenF(&_F[0], _n);

        #ifdef _OPENMP
        #pragma omp parallel for private(k, ell)
        #endif
        for(int i=0; i<_n; i++) {
            if(i>0) {
                // Construct C and c matrices that we'll Chosolve below
                // I think these are essentially the constituents of eq (3) of Datta++14
                for(k=0; k<nnIndxLU[_n+i]; k++) {
                    _c[nnIndxLU[i]+k] = _cm.cov(nnDist[nnIndxLU[i]+k]);
                    for(ell=0; ell<=k; ell++) {
                        _C[CIndx[i]+ell*nnIndxLU[_n+i]+k] =
                            _cm.cov(_D[CIndx[i]+ell*nnIndxLU[_n+i]+k]);
                    }
                }
                // Note symmetric, so shouldn't matter if I screw up row/col major here.
                const Eigen::Map<const MatrixXd> eigenC(&_C[CIndx[i]], nnIndxLU[i+_n], nnIndxLU[i+_n]);
                const Eigen::Map<const VectorXd> eigenc(&_c[nnIndxLU[i]], nnIndxLU[i+_n]);
                // Might be good to figure out how to use solveInPlace here.
                auto Blocal = eigenB.segment(nnIndxLU[i], nnIndxLU[_n+i]);
                Blocal = eigenC.llt().solve(eigenc);
                eigenF[i] = _cm.cov(0.0) - Blocal.dot(eigenc);
            } else {
                _B[i] = 0;
                _F[i] = _cm.cov(0.0);
            }
        }
    }

    void SeqNNGP::updateW() {
        for(int i=0; i<_n; i++) {
            double a = 0.0;
            double v = 0.0;
            if (uIndxLU[_n+i]>0) { //is i a neighbor for anybody
                for(int j=0; j<uIndxLU[_n+i]; j++) { //how many location have i as a neighbor
                    double b = 0.0;
                    // now the neighbors for the jth location who has i as a neighbor
                    int jj = uIndx[uIndxLU[i]+j]; //jj is the index of the jth location who has i as a neighbor
                    for(int k=0; k<nnIndxLU[_n+jj]; k++) { // these are the neighbors of the jjth location
                        int kk = nnIndx[nnIndxLU[jj]+k]; // kk is the index for the jth locations neighbors
                        if (kk != i) { //if the neighbor of jj is not i
                            b += _B[nnIndxLU[jj]+k] * _w[kk]; //covariance between jj and kk and the random effect of kk
                        }
                    }
                    a += _B[nnIndxLU[jj]+uiIndx[uIndxLU[i]+j]]*(_w[jj]-b)/_F[jj];
                    v += pow(_B[nnIndxLU[jj]+uiIndx[uIndxLU[i]+j]],2)/_F[jj];
                }
            }

            double e = 0.0;
            for(int j=0; j<nnIndxLU[_n+i]; j++){
                e += _B[nnIndxLU[i]+j] * _w[nnIndx[nnIndxLU[i]+j]];
            }
            double mu = _y[i] - _X.row(i).dot(_beta)/_tausqr + e/_F[i] + a;
            double var = 1.0/(1.0/_tausqr + 1.0/_F[i] + v);

            std::normal_distribution<> norm{mu*var, std::sqrt(var)};
            _w[i] = norm(_gen);
        }
        // For debugging
        // _w[0] =   1.508921;
        // _w[1] =   0.250724;
        // _w[2] =   2.137284;
        // _w[3] =  -1.005202;
        // _w[4] =  -2.574499;
    }

    void SeqNNGP::updateBeta() {
        VectorXd tmp_p{_X.transpose()*(_y-_w)/_tausqr};
        MatrixXd tmp_pp{_XtX/_tausqr};

        // May be more efficient ways to do this...
        VectorXd mean = tmp_pp.llt().solve(tmp_p);
        MatrixXd cov = tmp_pp.inverse();
        _beta = MVNorm(mean, cov)(_gen);

        // For debugging
        // _beta << 2.114505, 2.841327;
        // std::cout << "_beta = \n" << _beta << '\n';
    }

    void SeqNNGP::updateTauSqr() {
        VectorXd tmp_n = _y - _w - _X*_beta;
        std::gamma_distribution<> gamma{_tauSqIGa+_n/2.0, _tauSqIGb+0.5*tmp_n.squaredNorm()};
        _tausqr = 1.0/gamma(_gen);
    }

    void SeqNNGP::updateSigmaSqr() {
        double a = 0.0;

        #ifdef _OPENMP
        #pragma omp parallel for private (e, j, b) reduction(+:a, logDet)
        #endif
        for(int i=0; i<_n; i++){
            double b = _w[i];
            if(nnIndxLU[_n+i] > 0){
                double e = 0.0;
                for(int j=0; j<nnIndxLU[_n+i]; j++){
                    e += _B[nnIndxLU[i]+j]*_w[nnIndx[nnIndxLU[i]+j]];
                }
                b -= e;
            }
            a += b*b/_F[i];
        }

        std::gamma_distribution<> gamma{_sigmaSqrIGa+_n/2.0, _sigmaSqrIGb+0.5*a*_sigmaSqr};
        _sigmaSqr = 1.0/gamma(_gen);
    }

    void SeqNNGP::updateTheta() {
        // Knows about w, B, F, indexing arrays, priors, RNG, MH proposal parameters,
        //   scratch-space for proposed B, F, theta?, other arrays?  (C, D)

        updateBF();
        double a=0.0;
        double logDet=0.0;

        #ifdef _OPENMP
        #pragma omp parallel for private (e, j, b) reduction(+:a, logDet)
        #endif
        for(int i=0; i<_n; i++) {
            double b = _w[i];
            if(nnIndxLU[_n+i] > 0) {
                double e=0.0;
                for(int j=0; j<nnIndxLU[_n+i]; j++) {
                    e += _B[nnIndxLU[i]+j]*_w[nnIndx[nnIndxLU[i]+j]];
                }
                b -= e;
            }
            a += b*b/_F[i];
            logDet += log(_F[i]);
        }

        double logPostCurrent = -0.5*logDet - 0.5*a;
        // logPostCurrent += std::log(theta[phiIndx] - phiUnifa) + std::log(phiUnifb - theta[phiIndx]);
        // if(corName == "matern"){
        //     logPostCurrent += log(theta[nuIndx] - nuUnifa) + log(nuUnifb - theta[nuIndx]);
        // }

    }
    //
    //     //candidate
    //     phiCand = logitInv(rnorm(logit(theta[phiIndx], phiUnifa, phiUnifb), tuning[phiIndx]), phiUnifa, phiUnifb);
    //     if(corName == "matern"){
    //         nuCand = logitInv(rnorm(logit(theta[nuIndx], nuUnifa, nuUnifb), tuning[nuIndx]), nuUnifa, nuUnifb);
    //     }
    //
    //     updateBF(BCand, FCand, c, C, D, d, nnIndxLU, CIndx, n, theta[sigmaSqIndx], phiCand, nuCand, covModel, bk, nuUnifb);
    //
    //     a = 0;
    //     logDet = 0;
    //
    //     #ifdef _OPENMP
    //     #pragma omp parallel for private (e, j, b) reduction(+:a, logDet)
    //     #endif
    //     for(i = 0; i < n; i++){
    //         if(nnIndxLU[n+i] > 0){
    //             e = 0;
    //             for(j = 0; j < nnIndxLU[n+i]; j++){
    //                 e += BCand[nnIndxLU[i]+j]*w[nnIndx[nnIndxLU[i]+j]];
    //             }
    //             b = w[i] - e;
    //         }else{
    //             b = w[i];
    //         }
    //         a += b*b/FCand[i];
    //         logDet += log(FCand[i]);
    //     }
    //
    //     logPostCand = -0.5*logDet - 0.5*a;
    //     logPostCand += log(phiCand - phiUnifa) + log(phiUnifb - phiCand);
    //     if(corName == "matern"){
    //         logPostCand += log(nuCand - nuUnifa) + log(nuUnifb - nuCand);
    //     }
    //
    //     if(runif(0.0,1.0) <= exp(logPostCand - logPostCurrent)){
    //
    //         std::swap(BCand, B);
    //         std::swap(FCand, F);
    //
    //         theta[phiIndx] = phiCand;
    //         if(corName == "matern"){
    //             theta[nuIndx] = nuCand;
    //         }
    //
    //         accept++;
    //         batchAccept++;
    //     }
    // }
}
