#include "covModel.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

namespace pyNNGP {
    void pyExportCovModel(py::module& m) {
        py::class_<CovModel>(m, "CovModel");

        py::class_<ExponentialCovModel, CovModel>(m, "Exponential")
            .def(py::init<double,double,double,double,double,double,double>())
            .def("cov", &ExponentialCovModel::cov);


        py::class_<SphericalCovModel, CovModel>(m, "Spherical")
            .def(py::init<double,double,double,double,double,double,double>())
            .def("cov", &SphericalCovModel::cov);


        py::class_<SqExpCovModel, CovModel>(m, "SqExp")
            .def(py::init<double,double,double,double,double,double,double>())
            .def("cov", &SqExpCovModel::cov);
    }
}
