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

        // Allocate our own memory for these
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

        // Use existing memory here (allocated in python-layer)
        const Eigen::Map<const VectorXd> y;       // [n]
        const Eigen::Map<const MatrixXd> Xt;      // [p, n]  ([n, p] in python)
        const Eigen::Map<const MatrixXd> coords;  // [2, n]  ([n, 2] in python)
        const MatrixXd XtX;

        CovModel& cm;  // Model for covariances
        double tauSq;  // Measurement uncertainty

        std::random_device rd;
        std::mt19937 gen;

        // These are mostly internal, but I'm too lazy for the moment to make them private
        // We allocate this memory ourselves.
        std::vector<double> B;      // [nIndx]
        std::vector<double> F;      // [n]
        std::vector<double> Bcand;  // [nIndx]
        std::vector<double> Fcand;  // [n]
        std::vector<double> c;      // [nIndx]
        std::vector<double> C;      // [~n*m*m]
        std::vector<double> D;      // [~n*m*m]
        VectorXd w;                 // [n] Latent GP samples
        VectorXd beta;              // [p] Unknown linear model coefficients
        double tauSqIGa;            // Priors for noise model
        double tauSqIGb;

        void sample(int nSamples);  // One Gibbs iteration

        // Use a particular covariance model to update given B and F vectors.
        void updateBF(double*, double*, CovModel&);
        void updateW();
        void updateBeta();
        void updateTauSq();

    private:
        void mkUIndx();
        void mkUIIndx();
        void mkCD();
    };

}

#endif
