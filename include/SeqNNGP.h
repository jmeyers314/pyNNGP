#ifndef NNGP_SeqNNGP_h
#define NNGP_SeqNNGP_h

#include "covModel.h"
#include <vector>
#include <random>

namespace pyNNGP {
    class SeqNNGP {
    public:
        SeqNNGP(const double* y, const double* X, const double* coords,
                int p, int n, int nNeighbors, const CovModel& cm, const double tausqr);

        std::vector<int> nnIndx;
        std::vector<int> nnIndxLU;
        std::vector<double> nnDist;
        std::vector<int> uIndx;
        std::vector<int> uIndxLU;
        std::vector<int> uiIndx;
        std::vector<int> CIndx;

    private:
        void mkUIndx();
        void mkUIIndx();
        void mkCD();
        void updateBF();
        void updateW();

        const double* _y;  // responses [n]
        const double* _X;  // predictors [n, p]
        const double* _coords; // [n, 2]

        const int _p;  // Number of indicators per input location
        const int _n;  // Number of input locations
        const int _m;  // Number of nearest neighbors
        const int _nIndx;  // Total number of neighbors (DAG edges)
        const double _tausqr;  // Measurement uncertainty

        std::vector<double> _B;
        std::vector<double> _F;
        std::vector<double> _BCand;
        std::vector<double> _FCand;
        std::vector<double> _c;
        std::vector<double> _C;
        std::vector<double> _D;
        std::vector<double> _w; // Latent GP samples?
        std::vector<double> _beta; // Unknown linear model coefficients

        std::random_device _rd;
        std::mt19937 _gen;

        const CovModel& _cm;
    };
}

#endif
