#include <cmath>

class CovModel {
public:
    virtual double cov(double) const = 0;
};

class ExponentialCovModel : public CovModel {
public:
    ExponentialCovModel(double sigmaSq, double phi) : _sigmaSq(sigmaSq), _phi(phi) {}
    double cov(double x) const override {
        return _sigmaSq*std::exp(-x*_phi);
    }
private:
    const double _sigmaSq;
    const double _phi;
};

class SphericalCovModel : public CovModel {
public:
    SphericalCovModel(double sigmaSq, double phi) : _sigmaSq(sigmaSq), _phi(phi), _phiInv(1./_phi) {}
    double cov(double x) const override {
        if (x>0.0 && x < _phiInv) {
            return _sigmaSq*(1.0 - 1.5*_phi*x + 0.5*std::pow(_phi*x, 3));
        } else if (x >= _phiInv) {
            return 0.0;
        } else {
            return _sigmaSq;
        }
    }
private:
    const double _sigmaSq;
    const double _phi, _phiInv;
};

class SqExpCovModel : public CovModel {
public:
    SqExpCovModel(double sigmaSq, double phi) : _sigmaSq(sigmaSq), _phi(phi) {}
    double cov(double x) const override {
        return _sigmaSq*std::exp(-1.0*std::pow(_phi*x, 2));
    }
private:
    const double _sigmaSq;
    const double _phi;
};
