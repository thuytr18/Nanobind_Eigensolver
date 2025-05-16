#include <nanobind/nanobind.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <madness/mra/mra.h>
#include <madness/mra/function_interface.h>
#include <madness/mra/vmra.h>
#include <madness/tensor/tensor.h>
#include <madness/tensor/tensor_lapack.h>
#include <madness/world/vector.h>
#include <madness/world/world.h>
#include <ostream>
#include <vector>

#include "potential.h"

namespace nb = nanobind;

template <typename T, std::size_t NDIM>
madness::Function<T, NDIM> create_gaussian(
    GaussianPotentialGenerator<T, NDIM>& generator,
    double a,
    const std::vector<T>& mu_vec,
    const std::vector<std::vector<T>>& sigma_mat
) {
    if (mu_vec.size() != NDIM || sigma_mat.size() != NDIM || sigma_mat[0].size() != NDIM) {
        throw std::invalid_argument("Dimension mismatch in mu or sigma");
    }

    madness::Vector<T, NDIM> mu;
    for (std::size_t i = 0; i < NDIM; ++i) {
        mu[i] = mu_vec[i];
    }

    madness::Tensor<T> sigma(NDIM, NDIM);
    for (std::size_t i = 0; i < NDIM; ++i) {
        for (std::size_t j = 0; j < NDIM; ++j) {
            sigma(i, j) = sigma_mat[i][j];
        }
    }

    return generator.create_gaussianpotential(a, mu, sigma);
}

template <typename T, std::size_t NDIM>
void create_potentials(nb::module_& m) {
    using Vec = madness::Vector<T, NDIM>;
    using TensorType = madness::Tensor<T>;
    using FunctionType = madness::Function<T, NDIM>;

    nb::class_<ExponentialPotentialGenerator<T, NDIM>>(m, "ExponentialPotentialGenerator")
        .def(nb::init<madness::World&>())
        .def("create", &ExponentialPotentialGenerator<T, NDIM>::create_exponentialpotential);

    nb::class_<GaussianPotentialGenerator<T, NDIM>>(m, "GaussianPotentialGenerator")
        .def(nb::init<madness::World&>())
        .def("create", &create_gaussian<T, NDIM>,
             nb::arg("a"), nb::arg("mu"), nb::arg("sigma"));
}

NB_MODULE(potentials, m) {
    m.doc() = "Bindings for exponential and gaussian potentials using nanobind";

    nb::class_<madness::World>(m, "World")
        .def(nb::init<>());

    create_potentials<double, 1>(m);

}

