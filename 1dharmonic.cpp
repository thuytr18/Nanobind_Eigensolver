#include <nanobind/nanobind.h>
#include <madness/mra/mra.h>
#include <madness/mra/funcplot.h>
#include <madness/mra/nonlinsol.h>

#include "1dharmonic.hpp"

namespace nb = nanobind;
using namespace madness;
using namespace nb;

const double delta = 7;
//const std::string path_to_plots="/Users/truonthu/Documents/MRA/plots";

// The initial guess wave function
double guess(const coord_1d& r) {
    return exp(-(r[0]*r[0])/1.0);
}

// The shifted potential
double potential(const coord_1d& r) {
    return 0.5*(r[0]*r[0]) - delta;
}


OneDHarmonic::OneDHarmonic(double L, long k, double thresh, double DELTA)
{
    int arg = 0;
    char **a = new char*[0]();
    initialize(arg, a);
    world = new World(SafeMPI::COMM_WORLD);

    startup(*world, arg, a);

    FunctionDefaults<3>::set_k(k);
    FunctionDefaults<3>::set_thresh(thresh);
    FunctionDefaults<3>::set_refine(true);
    FunctionDefaults<3>::set_initial_level(5);
    FunctionDefaults<3>::set_truncate_mode(1);
    FunctionDefaults<3>::set_cubic_cell(-L, L);
}

OneDHarmonic::~OneDHarmonic()
{

}

double OneDHarmonic::compute_energy(World& world, const real_function_1d& phi, const real_function_1d& V) {
    double potential_energy = inner(phi,V*phi); // <phi|Vphi> = <phi|V|phi>
    double kinetic_energy = 0.0;

    real_derivative_1d D = Derivative<double,1>(world, 0);
    real_function_1d dphi = D(phi);
    kinetic_energy += 0.5*inner(dphi,dphi);  // (1/2) <dphi/dx | dphi/dx>

    return kinetic_energy + potential_energy;
}

void OneDHarmonic::solve_eigenproblem(World& world, int max_iter, double thresh, double DELTA) {
    if (world.rank() == 0) printf("starting at time %.1f\n", wall_time());
    FunctionDefaults<1>::set_k(6);
    FunctionDefaults<1>::set_thresh(thresh);
    FunctionDefaults<1>::set_cubic_cell(-L,L);
    NonlinearSolverND<1> solver;

    real_function_1d phi = real_factory_1d(world).f(guess);
    real_function_1d V = real_factory_1d(world).f(potential);

    phi.scale(1.0/phi.norm2());  // phi *= 1.0/norm

    double E = compute_energy(world,phi,V);

    for (int iter=0; iter<100; iter++) {
      real_function_1d Vphi = V*phi;
      Vphi.truncate();
      real_convolution_1d op = BSHOperator<1>(world, sqrt(-2*E), 0.01, thresh);

      real_function_1d r = phi + 2.0 * op(Vphi); // the residual
      double err = r.norm2();

      phi = solver.update(phi, r);
      //phi = phi-r;

      double norm = phi.norm2();
      phi.scale(1.0/norm);  // phi *= 1.0/norm
      E = compute_energy(world,phi,V);

      if (world.rank() == 0)
          print("iteration", iter, "energy", E, "norm", norm, "error",err);

      if (err < 5e-4) break;
    }

    print("Final energy without shift", E+DELTA);

    if (world.rank() == 0) printf("finished at time %.1f\n", wall_time());
}



NB_MODULE(madness_ext, m) {
    nb::class_<OneDHarmonic>(m, "OneDHarmonic")
        .def(nb::init<const double &, const int &, const double &, const double &>())
        .def("compute_energy", &OneDHarmonic::compute_energy)
        .def("solve_eigenproblem", &OneDHarmonic::solve_eigenproblem);
}
