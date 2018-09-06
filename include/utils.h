#ifndef pyNNGP_Utils_h
#define pyNNGP_Utils_h

#include <vector>
#include <random>
#include <Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace pyNNGP {
    double dist2(double a1, double b1, double a2, double b2);
    double dist2(const VectorXd& a, const VectorXd& b);
    int which(int a, int *b, int n);

    template<typename... Args>
    void apply_permutation(std::vector<int>& p, Args*... args);

    template<typename... Args>
    void apply_permutation(std::vector<int>& p, Args&... args);

    void rsort_with_index(double* x, int* index, int N);

    double logit(double theta, double a, double b);
    double logitInv(double z, double a, double b);

    // Consider replacing the following with https://github.com/beniz/eigenmvn/blob/master/eigenmvn.h
    class MVNorm {
    public:
        MVNorm(VectorXd mu, MatrixXd cov) : _mu(mu), _transform(cov.llt().matrixL()) {}

        template<class Generator=std::mt19937>
        VectorXd operator()(Generator& g) {
            std::normal_distribution<> norm{0.0, 1.0};
            VectorXd result(_mu.size());
            for(int i=0; i<_mu.size(); i++)
                result[i] = norm(g);
            return _transform * result + _mu;
        }
    private:
        const VectorXd& _mu;
        const MatrixXd _transform;
    };
}

#endif
