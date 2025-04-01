#include <iostream>
#include <madness/mra/mra.h>
#include <madness/mra/vmra.h>
#include <madness/mra/operator.h>
#include <madness/chem/oep.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <algorithm>

using namespace madness;

class OneDHarmonic{
public:
    OneDHarmonic(double L, long k, double thresh, double DELTA);
    ~OneDHarmonic();

    double compute_energy(World& world, const real_function_1d& phi, const real_function_1d& V);
    void solve_eigenproblem(World& world, int max_iter, double thresh, double DELTA);

private:
    World* world;
    double L;
    double k;
    double thresh;
    double DELTA;
};