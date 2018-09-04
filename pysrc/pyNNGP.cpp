#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace pyNNGP {
    void pyExportSeqNNGP(py::module& m);
    void pyExportCovModel(py::module& m);

    PYBIND11_MODULE(_pyNNGP, m) {
        pyExportSeqNNGP(m);
        pyExportCovModel(m);
    }
}
