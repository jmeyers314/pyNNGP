#ifndef NNGP_SeqNNGP_h
#define NNGP_SeqNNGP_h

#include "covModel.h"
#include <vector>
#include <random>
#include <Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace pyNNGP {
    class SeqNNGP {
    public:
        SeqNNGP(const double* y, const double* X, const double* coords,
                int p, int n, int nNeighbors, const CovModel& cm, const double tausqr);

        std::vector<int> nnIndx;    // [nIndx]
        std::vector<int> nnIndxLU;  // [2*n]
        std::vector<double> nnDist; // [nIndx]
        std::vector<int> uIndx;     // [nIndx]
        std::vector<int> uIndxLU;   // [2*n]
        std::vector<int> uiIndx;    // [nIndx]
        std::vector<int> CIndx;     // [2*n]

    private:
        const int _p;  // Number of indicators per input location
        const int _n;  // Number of input locations
        const int _m;  // Number of nearest neighbors
        const int _nIndx;  // Total number of neighbors (DAG edges)

        const Eigen::Map<const VectorXd> _y;       // [n]
        const Eigen::Map<const MatrixXd> _X;       // [n, p]
        const Eigen::Map<const MatrixXd> _coords;  // [n, 2]
        const MatrixXd _XtX;

        const CovModel& _cm;

        double _tausqr;  // Measurement uncertainty

        std::random_device _rd;
        std::mt19937 _gen;

        std::vector<double> _B;      // [nIndx]
        std::vector<double> _F;      // [n]
        std::vector<double> _BCand;  // [nIndx]
        std::vector<double> _FCand;  // [n]
        std::vector<double> _c;      // [nIndx]
        std::vector<double> _C;      // [j?]
        std::vector<double> _D;      // [j?]
        VectorXd _w;                 // [n] Latent GP samples?
        VectorXd _beta;              // [p] Unknown linear model coefficients
        double _tauSqIGa;
        double _tauSqIGb;

        void mkUIndx();
        void mkUIIndx();
        void mkCD();
        void updateBF();
        void updateW();
        void updateBeta();
        void updateTauSqr();
    };
}

#endif
