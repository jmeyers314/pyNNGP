#ifndef NNGP_noiseModel_h
#define NNGP_noiseModel_h

#include "SeqNNGP.h"

namespace pyNNGP {
    class NoiseModel {
    public:
        NoiseModel() : _isXSet(false), _XtW(nullptr, 0, 0) {}
        virtual void update(SeqNNGP&) = 0;
        virtual double invTauSq(int i) const = 0;
        virtual void setX(const Eigen::Ref<const MatrixXd>& Xt) = 0;
        virtual MatrixXd getXtW(void) const = 0;
        virtual MatrixXd getXtWX(void) const = 0;
    protected:
        bool _isXSet;
        Eigen::Map<const MatrixXd> _XtW;
        MatrixXd _XtWX;
    };

    class IGNoiseModel : public NoiseModel {
    public:
        IGNoiseModel(double tauSq, double IGa, double IGb) :
            NoiseModel(),
            _tauSq(tauSq), _invTauSq(1./_tauSq), _IGa(IGa), _IGb(IGb) {}

        virtual void update(SeqNNGP& seq) override {
            VectorXd tmp_n = seq.y - seq.w - seq.Xt.transpose()*seq.beta;
            std::gamma_distribution<> gamma{_IGa+seq.n/2., _IGb+0.5*tmp_n.squaredNorm()};
            _invTauSq = gamma(seq.gen);
            _tauSq = 1./_invTauSq;
        }

        virtual double invTauSq(int i) const override { return _invTauSq; }

        virtual void setX(const Eigen::Ref<const MatrixXd>& Xt) override {
            new (&_XtW) Eigen::Map<const MatrixXd>{Xt.data(), Xt.rows(), Xt.cols()};
            _XtWX = Xt*Xt.transpose();
            _isXSet = true;
        }

        // Would like to return an expression template.  How?
        virtual MatrixXd getXtW(void) const override {
            assert(_isXSet);
            return _XtW*_invTauSq;
        }

        virtual MatrixXd getXtWX(void) const override {
            assert(_isXSet);
            return _XtWX*_invTauSq;
        }

    private:
        double _tauSq;
        double _invTauSq;
        double _IGa;
        double _IGb;
    };

    class ConstHomogeneousNoiseModel : public NoiseModel {
    public:
        ConstHomogeneousNoiseModel(double tauSq) :
            NoiseModel(),
            _tauSq(tauSq), _invTauSq(1./_tauSq) {}

        virtual void update(SeqNNGP& seq) override {} // noop

        virtual double invTauSq(int i) const override { return _invTauSq; }

        virtual void setX(const Eigen::Ref<const MatrixXd>& Xt) override {
            MatrixXd XtW{Xt*_invTauSq};
            new (&_XtW) Eigen::Map<const MatrixXd>{XtW.data(), XtW.rows(), XtW.cols()};
            _XtWX = Xt*Xt.transpose()*_invTauSq;
            _isXSet = true;
        }

        virtual MatrixXd getXtW(void) const override {
            assert(_isXSet);
            return _XtW;
        }

        virtual MatrixXd getXtWX(void) const override {
            assert(_isXSet);
            return _XtWX;
        }

    private:
        double _tauSq;
        double _invTauSq;
    };

    class ConstHeterogeneousNoiseModel : public NoiseModel {
    public:
        ConstHeterogeneousNoiseModel(const double* tauSqPtr, int n) :
            NoiseModel(),
            _tauSq(tauSqPtr, n) {}

        virtual void update(SeqNNGP& seq) override {} // noop

        virtual double invTauSq(int i) const override { return 1./_tauSq(i); }

        virtual void setX(const Eigen::Ref<const MatrixXd>& Xt) override {
            MatrixXd XtW{Xt*_tauSq.asDiagonal().inverse()};
            new (&_XtW) Eigen::Map<const MatrixXd>{XtW.data(), XtW.rows(), XtW.cols()};
            _XtWX = Xt*_tauSq.asDiagonal().inverse()*Xt.transpose();
            _isXSet = true;
        }

        virtual MatrixXd getXtW(void) const override {
            assert(_isXSet);
            return _XtW;
        }

        virtual MatrixXd getXtWX(void) const override {
            assert(_isXSet);
            return _XtWX;
        }

    private:
        const Eigen::Map<const VectorXd> _tauSq;
    };
}

#endif
