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


double toCPP_Array(const py::float_& a);

template<class ArrayType>
ArrayType toCPP_Array(const py::array& A);

template<class T, size_t N, template<class, size_t> class ArrayType>
py::array_t<T> to_numpy(const ArrayType<T, N>&, const std::vector<size_t>&);


#pragma GCC visibility push(hidden)
template<class T>
struct PyOdeResult{
    vec::HeapArray<T> x_src;
    vec::HeapArray<T> f_src;
    py::array_t<T> x;
    py::array_t<T> f;
    bool diverges;
    long double runtime;
};
#pragma GCC visibility pop



#pragma GCC visibility push(hidden)
template<class Tx, class Tf>
class PyOde : public ODE<Tx, Tf>{

    public:
        PyOde(ode<Tx, Tf> df): ODE<Tx, Tf>(df) {}

        const PyOdeResult<Tx> pysolve(const py::tuple& ics, const Tx& x, const Tx& dx, const Tx& err, py::str method = py::str("method"), const int max_frames=-1, py::tuple pyargs = py::tuple(),  py::object getcond = py::none(),  py::object breakcond = py::none(), const bool display=false) const;
        
        py::list pysolve_all(py::list& ics, const Tx& x, const Tx& dx, const Tx& err, py::str method = py::str("RK4"), const int& max_frames = -1, py::tuple pyargs = py::tuple(), int threads=-1) const;

        PyOde<Tx, Tf> copy() const;

};

#pragma GCC visibility pop


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


template<class Tx, class Tf>
const PyOdeResult<Tx> PyOde<Tx, Tf>::pysolve(const py::tuple& py_ics, const Tx& x, const Tx& dx, const Tx& err, py::str method, const int max_frames, py::tuple pyargs,  py::object getcond, py::object breakcond, const bool display) const {

    cond<Tx, Tf> wrapped_getcond = nullptr;
    cond<Tx, Tf> wrapped_breakcond = nullptr;
    vec::HeapArray<Tx> args;
    Tx x0;
    ICS<Tx, Tf> ics;
    size_t nd, nt;

    if (!getcond.is(py::none())) {
        wrapped_getcond = [getcond](const Tx& x1, const Tx& x2, const Tf& f1, const Tf& f2) -> bool {
            bool res = getcond(x1, x2, to_numpy(f1, {f1.size()}), to_numpy(f2, {f2.size()})).equal(py::bool_(true));
            return res;
        };
    }
    
    if (!breakcond.is(py::none())) {
        wrapped_breakcond = [breakcond](const Tx& x1, const Tx& x2, const Tf& f1, const Tf& f2) -> bool {
            return breakcond(x1, x2, to_numpy(f1, {f1.size()}), to_numpy(f2, {f2.size()})).equal(py::bool_(true));
        };
    }

    if (!pyargs.empty()){
        args = toCPP_Array<vec::HeapArray<Tx>>(pyargs);
    }
    x0 = py_ics[0].cast<Tx>();
    Tf f0 = toCPP_Array<Tf>(py_ics[1]);
    ics = {x0, f0};

    OdeResult<Tx, Tf> res = ODE<Tx, Tf>::solve(ics, x, dx, err, method.cast<std::string>().c_str(), max_frames, &args, wrapped_getcond, wrapped_breakcond, display);

    nd = res.f[0].size();
    nt = res.f.size();
    vec::HeapArray<Tx> f_flat(nd*nt, true);
    for (size_t i=0; i<nt; i++){
        for (size_t j=0; j<nd; j++){
            f_flat[i*nd+j] = res.f[i][j];
        }
    }

    PyOdeResult<Tx> odres{res.x, f_flat, to_numpy(res.x, {nt}), to_numpy(f_flat, {nt, nd}), res.diverges, res.runtime};

    return odres;
}


template<class Tx, class Tf>
py::list PyOde<Tx, Tf>::pysolve_all(py::list& py_ics, const Tx& x, const Tx& dx, const Tx& err, py::str method, const int& max_frames, py::tuple pyargs, int threads) const{
    size_t n = py_ics.size();
    std::vector<ICS<Tx, Tf>> ics;
    vec::HeapArray<Tx> args;
    py::list res;

    if (threads == -1){
        threads = omp_get_max_threads();
    }

    for (size_t i=0; i<n; i++){
        res.append(py::none());
    }

    #pragma omp parallel for num_threads(threads)
    for (size_t i=0; i<n; i++){
        res[i] = pysolve(py_ics[i], x, dx, err, method, max_frames, pyargs);
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
            py::arg("method") = py::str("RK4"),
            py::arg("max_frames") = -1,
            py::arg("args") = py::tuple(),
            py::arg("getcond") = py::none(),
            py::arg("breakcond") = py::none(),
            py::arg("display") = false)
        .def("solve_all", &PyOde<Tx, Tf>::pysolve_all,
            py::arg("ics"),
            py::arg("t"),
            py::arg("dt"),
            py::kw_only(),
            py::arg("err") = 0.,
            py::arg("method") = py::str("RK4"),
            py::arg("max_frames") = -1,
            py::arg("args") = py::tuple(),
            py::arg("threads") = -1)
        .def("copy", &PyOde<Tx, Tf>::copy);


    py::class_<PyOdeResult<Tx>>(m, "OdeResult", py::module_local())
        .def_readwrite("var", &PyOdeResult<Tx>::x)
        .def_readwrite("func", &PyOdeResult<Tx>::f)
        .def_readwrite("runtime", &PyOdeResult<Tx>::runtime)
        .def_readwrite("diverges", &PyOdeResult<Tx>::diverges);

    m.def("ode", [func_ptr]() {
        return PyOde<Tx, Tf>(func_ptr);
    });

}



#endif