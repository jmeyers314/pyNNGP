#ifndef pyNNGP_Tree_h
#define pyNNGP_Tree_h

#include <Eigen/Dense>

using Eigen::MatrixXd;

namespace pyNNGP {
    struct Node {
        Node (int i);
        int index;  // which point I am
        Node *left = nullptr;
        Node *right = nullptr;
    };

    Node *miniInsert(Node *Tree, const MatrixXd& coords, int index, int d,int n);

    void get_nn(Node *Tree, int index, int d, const MatrixXd& coords, int n,
                double *nnDist, int *nnIndx, int iNNIndx, int iNN, int check);
    void getNNIndx(int i, int m, int& iNNIndx, int& iNN);

    void mkNNIndxTree0(const int n, const int m, const MatrixXd& coords,
                       int *nnIndx, double *nnDist, int *nnIndxLU);
}

#endif
