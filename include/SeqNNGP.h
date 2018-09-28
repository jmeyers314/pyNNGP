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
        SeqNNGP(const double* _y, const double* _X, const double* _coords,
                int _p, int _n, int _m, CovModel& _cm, const double _tauSq);

        // Allocate our own memory for these
        // Nearest neighbors index.  Holds the indices of the neighbors of each node.
        std::vector<int> nnIndx;    // [nIndx]

        // Nearest neighbors ranges
        // Lower part holds starting index for each node
        // Upper part holds number of elements for each node
        std::vector<int> nnIndxLU;  // [2*n]

        // Distances between neighbors
        std::vector<double> nnDist; // [nIndx]

        // Reverse index.  Holds which nodes have n as a neighbor
        std::vector<int> uIndx;     // [nIndx]

        // Ranges for reverse index.
        std::vector<int> uIndxLU;   // [2*n]
        // Which neighbor is it?
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
