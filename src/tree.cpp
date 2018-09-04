#include "tree.h"
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace pyNNGP {

    double dist2(double a1, double a2, double b1, double b2) {
        return std::sqrt((a1-b1)*(a1-b1) + (a2-b2)*(a2-b2));
    }

    void apply_permutation(double* v, int* index, std::vector<int>& p, int N) {
        // apply permutation p to both v and index.
        using std::swap;  // permit ADL
        for (int i = 0; i < N; i++) {
            int current = i;
            while (i != p[current]) {
                int next = p[current];
                swap(v[current], v[next]);
                swap(index[current], index[next]);
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
        apply_permutation(x, index, p, N);
    }

    void getNNIndx(int i, int m, int& iNNIndx, int& iNN) {
        // return index into nnIndx array, nnDist array, ...
        // iNN is # of neighbors of i
        if(i == 0) {
            iNNIndx = 0; //this should never be accessed
            iNN = 0;
            return;
        } else if(i < m) {
            iNNIndx = (i*(i-1))/2;
            iNN = i;
            return;
        } else {
            iNNIndx = (m*(m-1))/2 + (i-m)*m;
            iNN = m;
            return;
        }
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

    Node::Node(int i) : index(i), left(nullptr), right(nullptr) {}

    Node* miniInsert(Node* Tree, const double* coords, int index, int d) {
        // 2D-tree
        if(!Tree) return new Node(index);

        if (coords[2*index+d] <= coords[2*Tree->index+d])
            Tree->left = miniInsert(Tree->left, coords, index, (d+1)%2);
        else
            Tree->right = miniInsert(Tree->right, coords, index, (d+1)%2);
        return Tree;
    }

    void get_nn(Node* Tree, int index, int d, const double* coords,
                double* nnDist, int* nnIndx, int iNNIndx, int iNN) {
        // input: Tree, index, d, coords
        // output: nnDist, nnIndx

        if(!Tree) return;

        double disttemp= dist2(coords[2*index],coords[2*index+1],
                               coords[2*Tree->index],coords[2*Tree->index+1]);

        if(index!=Tree->index && disttemp<nnDist[iNNIndx+iNN-1]) {
            nnDist[iNNIndx+iNN-1]=disttemp;
            nnIndx[iNNIndx+iNN-1]=Tree->index;
            rsort_with_index(&nnDist[iNNIndx], &nnIndx[iNNIndx], iNN);
        }

        Node* temp1=Tree->left;
        Node* temp2=Tree->right;

        if(coords[2*index+d]>coords[2*Tree->index+d])
            std::swap(temp1,temp2);
        get_nn(temp1,index,(d+1)%2,coords,nnDist,nnIndx,iNNIndx,iNN);
        if(fabs(coords[2*Tree->index+d]-coords[2*index+d])>nnDist[iNNIndx+iNN-1])
            return;
        get_nn(temp2,index,(d+1)%2,coords,nnDist,nnIndx,iNNIndx,iNN);
    }


    void mkNNIndxTree0(const int n, const int m, const double* coords,
                       int* nnIndx, double* nnDist, int* nnIndxLU) {
        int i, iNNIndx, iNN;
        double d;
        int nIndx = ((1+m)*m)/2+(n-m-1)*m;
        // Results seem to depend on BUCKETSIZE, which seems weird...
        int BUCKETSIZE = 10;

        std::fill(&nnDist[0], &nnDist[0]+nIndx, std::numeric_limits<double>::infinity());

        Node* Tree=nullptr;
        int time_through=-1;

        for(i=0; i<n; i++) {
            getNNIndx(i, m, iNNIndx, iNN);
            nnIndxLU[i] = iNNIndx;
            nnIndxLU[n+i] = iNN;
            if(time_through==-1)
                time_through=i;

            if(i!=0) {
                for(int j=time_through; j<i; j++) {
                    getNNIndx(i, m, iNNIndx, iNN);
                	d = dist2(coords[2*i], coords[2*i+1], coords[2*j], coords[2*j+1]);
                    if(d < nnDist[iNNIndx+iNN-1]){
            	        nnDist[iNNIndx+iNN-1] = d;
            	        nnIndx[iNNIndx+iNN-1] = j;
                    	rsort_with_index(&nnDist[iNNIndx], &nnIndx[iNNIndx], iNN);
          	        }
                }
                if(i%BUCKETSIZE==0) {
                    #ifdef _OPENMP
                    #pragma omp parallel for private(iNNIndx, iNN)
                    #endif
                	for(int j=time_through;j<time_through+BUCKETSIZE;j++){
                    	getNNIndx(j, m, iNNIndx, iNN);
                	    get_nn(Tree, j, 0, coords, nnDist, nnIndx, iNNIndx, iNN);
            	    }

                	for(int j=time_through;j<time_through+BUCKETSIZE;j++)
                	    Tree=miniInsert(Tree, coords, j, 0);

                	time_through=-1;
                }
                if(i==n-1) {
                    #ifdef _OPENMP
                    #pragma omp parallel for private(iNNIndx, iNN)
                    #endif
                	for(int j=time_through;j<n;j++){
                	    getNNIndx(j, m, iNNIndx, iNN);
                	    get_nn(Tree, j, 0, coords, nnDist, nnIndx, iNNIndx, iNN);
                	}
                }
            } else {  // i==0
                Tree=miniInsert(Tree, coords, i, 0);
                time_through=-1;
            }
        }
        delete Tree;
    }
}
