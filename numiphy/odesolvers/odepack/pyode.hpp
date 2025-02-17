#ifndef PYODE_HPP
#define PYODE_HPP



#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <omp.h>
#include "odes.hpp"

/*

-------------------------- DECLARATIONS -----------------------------------------

*/


namespace py = pybind11;


#pragma GCC visibility push(hidden)
template<class T>
struct PyOdeResult{
    vec::HeapArray<T> x_src;
    vec::HeapArray<T> f_src;
    py::array_t<T> x;
    py::array_t<T> f;
    bool diverges;
    bool is_stiff;
    long double runtime;
};
#pragma GCC visibility pop

#pragma GCC visibility push(hidden)
template<class T>
struct PyOdeArgs{

    py::tuple ics;
    T x;
    T dx;
    T err;
    T cutoff_step;
    py::str method;
    int max_frames;
    py::tuple pyargs;
    py::object getcond;
    py::object breakcond;

};
#pragma GCC visibility pop

#pragma GCC visibility push(hidden)
template<class Tx, class Tf>
class PyOde : public ODE<Tx, Tf>{

    public:
        PyOde(ode<Tx, Tf> df): ODE<Tx, Tf>(df) {}

        const PyOdeResult<Tx> pysolve(const py::tuple& ics, const Tx& x, const Tx& dx, const Tx& err, const Tx& cutoff_step, const py::str& method, const int& max_frames, const py::tuple& pyargs, const py::object& getcond, const py::object& breakcond) const;
        
        py::list pysolve_all(const py::list&, int threads) const;

        PyOde<Tx, Tf> copy() const;

        static py::list py_dsolve_all(const py::list& data, int threads);

};

#pragma GCC visibility pop

#pragma GCC visibility push(hidden)
template<class Tx, class Tf>
struct PyOdeSet{

    PyOde<Tx, Tf> ode;
    PyOdeArgs<Tx> params;

};
#pragma GCC visibility pop

double toCPP_Array(const py::float_& a);

template<class ArrayType>
ArrayType toCPP_Array(const py::array& A);

template<class T, size_t N, template<class, size_t> class ArrayType>
py::array_t<T> to_numpy(const ArrayType<T, N>&, const std::vector<size_t>&);

template<class T, size_t N1, size_t N2, template<class, size_t> class Tfall, template<class, size_t> class Tf>
vec::HeapArray<T> flatten(const Tfall<Tf<T, N1>, N2>&);

template<class Tx, class Tf>
OdeArgs<Tx, Tf> to_OdeArgs(const PyOdeArgs<Tx>& pyparams);

/*
------------------------------------------------------------------------------------
-------------------------- IMPLEMENTATIONS -----------------------------------------
------------------------------------------------------------------------------------
*/


double toCPP_Array(const py::float_& a){
    return a.cast<double>();
}


template<class ArrayType>
ArrayType toCPP_Array(const py::array& A){
    size_t n = A.size();
    ArrayType res;
    const double* data = static_cast<const double*>(A.data());
    res.try_alloc(n);
    res.fill();

    for (size_t i=0; i<n; i++){
        res[i] = toCPP_Array(data[i]);
    }
    return res;
}


template<class T, size_t N, template<class, size_t> class ArrayType>
py::array_t<T> to_numpy(const ArrayType<T, N>& data, const std::vector<size_t>& shape){
    return py::array_t<T>(shape, data.data());
}

template<class T, size_t N1, size_t N2, template<class, size_t> class Tfall, template<class, size_t> class Tf>
vec::HeapArray<T> flatten(const Tfall<Tf<T, N1>, N2>& f){
    size_t nt = f.size();
    size_t nd = f[0].size();
    vec::HeapArray<T> res(nt*nd, true);

    for (size_t i=0; i<nt; i++){
        for (size_t j=0; j<nd; j++){
            res[i*nd + j] = f[i][j];
        }
    }
    return res;
}

template<class Tx, class Tf>
OdeArgs<Tx, Tf> to_OdeArgs(const PyOdeArgs<Tx>& pyparams){
    ICS<Tx, Tf> ics = {pyparams.ics[0].template cast<Tx>(), toCPP_Array<Tf>(pyparams.ics[1])};
    Tx x = pyparams.x;
    Tx dx = pyparams.dx;
    Tx err = pyparams.err;
    Tx cutoff_step = pyparams.cutoff_step;
    std::string method = pyparams.method.template cast<std::string>();
    int max_frames = pyparams.max_frames;
    std::vector<Tx> args;
    cond<Tx, Tf> getcond = nullptr;
    cond<Tx, Tf> breakcond = nullptr;

    if (!pyparams.pyargs.empty()){
        args = toCPP_Array<vec::HeapArray<Tx>>(pyparams.pyargs).to_vector();
    }

    if (!pyparams.getcond.is(py::none())) {
        getcond = [pyparams](const Tx& x, const Tf& f) -> bool {
            bool res = pyparams.getcond(x, to_numpy(f, {f.size()})).equal(py::bool_(true));
            return res;
        };
    }
    
    if (!pyparams.breakcond.is(py::none())) {
        breakcond = [pyparams](const Tx& x, const Tx& f) -> bool {
            bool res = pyparams.breakcond(x, to_numpy(f, {f.size()})).equal(py::bool_(true));
            return res;
        };
    }

    OdeArgs<Tx, Tf> res = {ics, x, dx, err, cutoff_step, method, max_frames, args, getcond, breakcond};
    return res;
    
}


template<class Tx, class Tf>
const PyOdeResult<Tx> PyOde<Tx, Tf>::pysolve(const py::tuple& ics, const Tx& x, const Tx& dx, const Tx& err, const Tx& cutoff_step, const py::str& method, const int& max_frames, const py::tuple& pyargs, const py::object& getcond, const py::object& breakcond) const {

    const PyOdeArgs<Tx> pyparams = {ics, x, dx, err, cutoff_step, method, max_frames, pyargs, getcond, breakcond};
    OdeArgs<Tx, Tf> ode_args = to_OdeArgs<Tx, Tf>(pyparams);

    OdeResult<Tx, Tf> res = ODE<Tx, Tf>::solve(ode_args);
    vec::HeapArray<Tx> f_flat = flatten(res.f);
    size_t nd = res.f[0].size();
    size_t nt = res.f.size();

    PyOdeResult<Tx> odres{res.x, f_flat, to_numpy(res.x, {nt}), to_numpy(f_flat, {nt, nd}), res.diverges, res.is_stiff, res.runtime};

    return odres;
}


template<class Tx, class Tf>
py::list PyOde<Tx, Tf>::pysolve_all(const py::list& pyparams, int threads) const{

    py::list data;
    size_t n = pyparams.size();
    for (size_t i = 0; i < n; i++){
        data.append(py::make_tuple(*this, pyparams[i]));
    }
    
    py::list res = py_dsolve_all(data, threads);

    return res;

}

template<class Tx, class Tf>
py::list PyOde<Tx, Tf>::py_dsolve_all(const py::list& data, int threads){
    
    size_t n = data.size();
    size_t nd, nt;
    std::vector<OdeSet<Tx, Tf>> odeset(n);
    std::vector<OdeResult<Tx, Tf>> ode_res;
    py::list res;

    //cast py_params to cpp params
    py::tuple tup;
    py::dict kw;
    py::list pyparams;
    for (size_t i=0; i<n; i++){
        tup = data[i].cast<py::tuple>();
        kw = tup[1].cast<py::dict>();
        
        PyOdeArgs<Tx> pystruct = {kw["ics"], kw["t"].cast<Tx>(), kw["dt"].cast<Tx>(), kw["err"].cast<Tx>(), kw["cutoff_step"].cast<Tx>(), kw["method"], kw["max_frames"].cast<int>(), kw["args"], py::none(), py::none()};
        odeset[i] = { tup[0].cast<PyOde<Tx, Tf>>(), to_OdeArgs<Tx, Tf>(pystruct)};
        if (kw.size() > 8){
            throw std::runtime_error("When solving an ode in parallel, no more than 7 arguments can be passed in the ode, since the rest of them would be cast into python functions.GIL prevents the program to call python function in parallel");
        }
    }

    //retrieve array of results from base class method
    ode_res = dsolve_all(odeset, threads);
    //convert results to python type
    for (size_t i=0; i<n; i++){
        nd = ode_res[i].f[0].size();
        nt = ode_res[i].f.size();
        vec::HeapArray<Tx> f_flat = flatten(ode_res[i].f);
        res.append(PyOdeResult<Tx>({ode_res[i].x, f_flat, to_numpy(ode_res[i].x, {nt}), to_numpy(f_flat, {nt, nd}), ode_res[i].diverges, ode_res[i].is_stiff, ode_res[i].runtime}));
    }
    
    return res;
}


template<class Tx, class Tf>
PyOde<Tx, Tf> PyOde<Tx, Tf>::copy() const{
    PyOde<Tx, Tf> res(this->_ode);
    return res;
}


template <typename Tx, typename Tf>
void define_ode_module(py::module& m, ode<Tx, Tf> func_ptr) {
    py::class_<PyOde<Tx, Tf>>(m, "LowLevelODE", py::module_local())
        .def("solve", &PyOde<Tx, Tf>::pysolve,
            py::arg("ics"),
            py::arg("t"),
            py::arg("dt"),
            py::kw_only(),
            py::arg("err") = 0.,
            py::arg("cutoff_step") = 0.,
            py::arg("method") = py::str("RK4"),
            py::arg("max_frames") = -1,
            py::arg("args") = py::tuple(),
            py::arg("getcond") = py::none(),
            py::arg("breakcond") = py::none())
        .def("solve_all", &PyOde<Tx, Tf>::pysolve_all,
            py::arg("params"),
            py::arg("threads") = -1)
        .def("copy", &PyOde<Tx, Tf>::copy)
        .def_static("dsolve_all", &PyOde<Tx, Tf>::py_dsolve_all,
            py::arg("data"),
            py::arg("threads") = -1)
        .def("__deepcopy__", [](const PyOde<Tx, Tf> &self, py::dict) {
            return self.copy();  // Calls copy constructor and returns a new object
        });


    py::class_<PyOdeResult<Tx>>(m, "OdeResult", py::module_local())
        .def_readonly("var", &PyOdeResult<Tx>::x)
        .def_readonly("func", &PyOdeResult<Tx>::f)
        .def_readonly("diverges", &PyOdeResult<Tx>::diverges)
        .def_readonly("is_stiff", &PyOdeResult<Tx>::is_stiff)
        .def_readonly("runtime", &PyOdeResult<Tx>::runtime);

    m.def("ode", [func_ptr]() {
        return PyOde<Tx, Tf>(func_ptr);
    });

}



#endif