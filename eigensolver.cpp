#include <nanobind/nanobind.h>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <madness/mra/funcdefaults.h>
#include <madness/mra/function_factory.h>
#include <madness/mra/function_interface.h>
#include <madness/mra/legendre.h>
#include <madness/mra/mra.h>
#include <madness/mra/funcplot.h>
#include <madness/mra/operator.h>
#include <madness/mra/vmra.h>
#include <madness/mra/derivative.h>
#include <madness/tensor/gentensor.h>
#include <madness/world/vector.h>
#include <madness/world/world.h>
#include <madness/world/worldmpi.h>
#include <madness/tensor/tensor.h>
#include <utility>
#include <vector>

#include "eigensolver.hpp"

namespace nb = nanobind;
using namespace madness;
using namespace nb;

using Eigensolver1D = Eigensolver<double, 1>;


NB_MODULE(eigensolver, m) {

    nb::class_<Eigensolver1D>(m, "Eigensolver1D")
        .def(nb::init<double, long, double>())

        .def("solve", &Eigensolver1D::solve))

        .def("energy", &Eigensolver1D::energy);
}