#ifndef NNGP_SeqNNGP_h
#define NNGP_SeqNNGP_h

#include <vector>
#include <random>
#include <Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace pyNNGP {
    class CovModel;
    class SeqNNGP {
    public:
        SeqNNGP(const double* y, const double* X, const double* coords,
                int p, int n, int nNeighbors, CovModel& cm, const double tausqr);

        std::vector<int> nnIndx;    // [nIndx]
        std::vector<int> nnIndxLU;  // [2*n]
        std::vector<double> nnDist; // [nIndx]
        std::vector<int> uIndx;     // [nIndx]
        std::vector<int> uIndxLU;   // [2*n]
        std::vector<int> uiIndx;    // [nIndx]
        std::vector<int> CIndx;     // [2*n]

        const int p;  // Number of indicators per input location
        const int n;  // Number of input locations
        const int m;  // Number of nearest neighbors
        const int nIndx;  // Total number of neighbors (DAG edges)

        const Eigen::Map<const VectorXd> y;       // [n]
        const Eigen::Map<const MatrixXd> X;       // [n, p]
        const Eigen::Map<const MatrixXd> coords;  // [n, 2]
        const MatrixXd XtX;

        CovModel& cm;

        double tauSq;  // Measurement uncertainty

        std::random_device rd;
        std::mt19937 gen;

        std::vector<double> B;      // [nIndx]
        std::vector<double> F;      // [n]
        std::vector<double> Bcand;  // [nIndx]
        std::vector<double> Fcand;  // [n]
        std::vector<double> c;      // [nIndx]
        std::vector<double> C;      // [j?]
        std::vector<double> D;      // [j?]
        VectorXd w;                 // [n] Latent GP samples?
        VectorXd beta;              // [p] Unknown linear model coefficients
        double tauSqIGa;
        double tauSqIGb;

        void sample(int nSamples);

        void updateBF(double*, double*, CovModel&);

    private:
        void mkUIndx();
        void mkUIIndx();
        void mkCD();
        void updateW();
        void updateBeta();
        void updateTauSq();
    };

}

#endif
