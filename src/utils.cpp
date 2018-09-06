#include "utils.h"

#include <cmath>
#include <numeric>
#include <random>

namespace pyNNGP {
    double dist2(double a1, double a2, double b1, double b2) {
        return std::sqrt((a1-b1)*(a1-b1) + (a2-b2)*(a2-b2));
    }

    double dist2(const VectorXd& a, const VectorXd& b) {
        return (a-b).norm();
    }

    // variadic template utility for apply_permutation
    template<typename T>
    void swap_at(int idx1, int idx2, T& a) {
        std::swap(a[idx1], a[idx2]);
    }

    template<typename T, typename... Args>
    void swap_at(int idx1, int idx2, T& a, Args&... args) {
        swap_at(idx1, idx2, a);
        swap_at(idx1, idx2, args...);
    }

    // Apply permutation p to each item in parameter pack
    // Note, p gets sorted during the algorithm, so can only
    // be used once.
    template<typename... Args>
    void apply_permutation(std::vector<int>& p, Args&... args) {
        for (const auto& i : p) {
            int current = i;
            while (i != p[current]) {
                int next = p[current];
                swap_at(current, next, args...);
                p[current] = current;
                current = next;
            }
            p[current] = current;
        }
    }

    // Same as above, but accept pointer instead of reference.
    template<typename... Args>
    void apply_permutation(std::vector<int>& p, Args*... args) {
        for (const auto& i : p) {
            int current = i;
            while (i != p[current]) {
                int next = p[current];
                swap_at(current, next, args...);
                p[current] = current;
                current = next;
            }
            p[current] = current;
        }
    }

    void rsort_with_index(double* x, int* index, int N) {
        // sorts on x, applies same permutation to index
        std::vector<int> p(N, 0);
        std::iota(p.begin(), p.end(), 0);
        std::sort(p.begin(), p.end(),
            [&](int a, int b)
            {return x[a] < x[b];}
        );
        apply_permutation(p, x, index);
    }

    //which index of b equals a, where b is of length n
    int which(int a, int *b, int n){
        int i;
        for(int i=0; i<n; i++){
            if(a == b[i])
                return(i);
        }
        std::runtime_error("c++ error: which failed");
        return -9999;
    }

    double logit(double theta, double a, double b){
        return log((theta-a)/(b-theta));
    }

    double logitInv(double z, double a, double b){
        return b-(b-a)/(1+exp(z));
    }
}
