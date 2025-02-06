#ifndef PYODE_HPP
#define PYODE_HPP


#include "odes.hpp"



#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>

/*

-------------------------- DECLARATIONS -----------------------------------------

*/


namespace py = pybind11;


double toCPP_Array(const py::float_& a);

template<class ArrayType>
ArrayType toCPP_Array(const py::list& A);

template<class ArrayType>
ArrayType toCPP_Array(const py::tuple& A);


template<class T>
py::array_t<T> to_numpy(const arr::Array<T>&, const std::vector<size_t>&);

template<class T, size_t N>
py::array_t<T> to_numpy(const arr::StaticArray<T, N>&, const std::vector<size_t>&);

// template<class T>
// arr::Array<T> flatten(const arr::Array<arr::Array<T>>& A);

// template<class T, size_t N>
// arr::Array<T> flatten(const arr::Array<arr::StaticArray<T, N>>& A);

#pragma GCC visibility push(hidden)
template<class T>
struct PyOdeResult{
    arr::Array<T> x_src;
    arr::Array<T> f_src;
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

        void set_ics(const Tx& x0, const py::list& f0);

        const PyOdeResult<Tx> pysolve(const Tx& x, const Tx& dx, const double& err, py::str method = py::str("method"), const int max_frames=-1, py::tuple pyargs = py::tuple(),  py::object getcond = py::none(),  py::object breakcond = py::none(), const bool display=false) const;
        
        py::list IntAll(py::list& ics, const Tx& x, const Tx& dx, const double& err, py::str method = py::str("RK4"), const int& max_frames = -1, py::tuple pyargs = py::tuple(), int threads=-1) const;

        PyOde<Tx, Tf> copy() const;

        PyOde<Tx, Tf> clone() const;

};

#pragma GCC visibility pop


template<class Tx, class Tf>
py::list py_IntegrateAll(const PyOde<Tx, Tf>& pyode, py::list& ics, const Tx& x, const Tx& dx, const double& err, py::str method = py::str("RK4"), const int& max_frames = -1, py::tuple pyargs = py::tuple(), int threads=-1);


double toCPP_Array(const py::float_& a){
    return a.cast<double>();
}


template<class ArrayType>
ArrayType toCPP_Array(const py::list& A){

    size_t n = A.size();
    ArrayType res;

    res.try_alloc(n);
    res.fill();

    for (size_t i=0; i<n; i++){
        res[i] = toCPP_Array(A[i]);
    }
    return res;
}

template<class ArrayType>
ArrayType toCPP_Array(const py::tuple& A){
    return toCPP_Array<ArrayType>(A.cast<py::list>());
}


template<class T>
py::array_t<T> to_numpy(const arr::Array<T>& data, const std::vector<size_t>& shape){
    return py::array_t<T>(shape, data.data());
}

template<class T, size_t N>
py::array_t<T> to_numpy(const arr::StaticArray<T, N>& data, const std::vector<size_t>& shape){
    return py::array_t<T>(shape, data.data());
}


/*
------------------------------------------------------------------------------------
-------------------------- IMPLEMENTATIONS -----------------------------------------
------------------------------------------------------------------------------------
*/






template<class Tx, class Tf>
void PyOde<Tx, Tf>::set_ics(const Tx& x0, const py::list& f0){
    Tf f = toCPP_Array<Tf>(f0);
    ODE<Tx, Tf>::set_ics(x0, f);
}


template<class Tx, class Tf>
const PyOdeResult<Tx> PyOde<Tx, Tf>::pysolve(const Tx& x, const Tx& dx, const double& err, py::str method, const int max_frames, py::tuple pyargs,  py::object getcond, py::object breakcond, const bool display) const {
    
    cond<Tx, Tf> wrapped_getcond = nullptr;
    cond<Tx, Tf> wrapped_breakcond = nullptr;
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
    
    arr::Array<Tx> args;
    
    if (!pyargs.empty()){
        args = toCPP_Array<arr::Array<Tx>>(pyargs);
    }

    OdeResult<Tx, Tf> res = ODE<Tx, Tf>::solve(x, dx, err, method.cast<std::string>().c_str(), max_frames, &args, wrapped_getcond, wrapped_breakcond, display);

    size_t nd = res.f[0].size();
    size_t nt = res.f.size();
    arr::Array<Tx> f_flat(nd*nt, true);
    for (size_t i=0; i<nt; i++){
        for (size_t j=0; j<nd; j++){
            f_flat[i*nd+j] = res.f[i][j];
        }
    }

    PyOdeResult<Tx> odres{res.x, f_flat, to_numpy(res.x, {nt}), to_numpy(f_flat, {nt, nd}), res.diverges, res.runtime};

    return odres;
}


template<class Tx, class Tf>
py::list PyOde<Tx, Tf>::IntAll(py::list& ics, const Tx& x, const Tx& dx, const double& err, py::str method, const int& max_frames, py::tuple pyargs, int threads) const{
    return py_IntegrateAll<Tx, Tf>(*this, ics, x, dx, err, method, max_frames, pyargs, threads);
}


template<class Tx, class Tf>
PyOde<Tx, Tf> PyOde<Tx, Tf>::copy() const{
    PyOde<Tx, Tf> res(this->_ode);
    return res;
}

template<class Tx, class Tf>
PyOde<Tx, Tf> PyOde<Tx, Tf>::clone() const{
    PyOde<Tx, Tf> res(this->_ode);

    py::list f0;
    for (size_t i=0; i<this->ics.f0.size(); i++){
        f0.append(this->ics.f0[i]);
    }
    res.set_ics(this->ics.x0, f0);
    return res;
}


template<class Tx, class Tf>
py::list py_IntegrateAll(const PyOde<Tx, Tf>& pyode, py::list& ics, const Tx& x, const Tx& dx, const double& err, py::str method, const int& max_frames, py::tuple pyargs, int threads){
    size_t n = ics.size();

    ODE<Tx, Tf> ode(pyode.odefunc());
    arr::Array<ODE<Tx, Tf>> ode_arr(n, true);
    arr::Array<OdeResult<Tx, Tf>> res(n, true);
    py::list pyres;
    arr::Array<PyOdeResult<Tx>> odres(n, true);
    arr::Array<Tx> args;
    
    if (threads == -1){
        threads = omp_get_max_threads();
    }

    if (!pyargs.empty()){
        args = toCPP_Array<arr::Array<Tx>>(pyargs);
    }
    
    for (size_t i=0; i<n; i++){
        ode_arr[i] = ode;
        py::tuple item = ics[i].cast<py::tuple>();
        ode_arr[i].set_ics(item[0].cast<double>(), toCPP_Array<Tf>(item[1].cast<py::list>()));
    }
    
    std::string meth = method.cast<std::string>();
    #pragma omp parallel for num_threads(threads)
    for (size_t i=0; i<n; i++){
        res[i] = ode_arr[i].solve(x, dx, err, meth.c_str(), max_frames, &args, nullptr, nullptr, false);
    }

    arr::Array<arr::Array<Tx>> f_flat(res.size(), true);
    size_t nd;
    size_t nt;
    for (size_t i=0; i<n; i++){

        nd = res[i].f[0].size();
        nt = res[i].f.size();
        f_flat[i].allocate(nd*nt);
        f_flat[i].fill();
        
        for (size_t j=0; j<nt; j++){
            for (size_t k=0; k<nd; k++){
                f_flat[i][j*nd+k] = res[i].f[j][k];
            }
        }

        pyres.append(PyOdeResult<Tx>({res[i].x, f_flat[i], to_numpy(res[i].x, {nt}), to_numpy(f_flat[i], {nt, nd}), res[i].diverges, res[i].runtime}));
    }
    return pyres;
}

template <typename Tx, typename Tf>
void define_ode_module(py::module& m, ode<Tx, Tf> func_ptr) {
    py::class_<PyOde<Tx, Tf>>(m, "LowLevelODE", py::module_local())
        .def("solve", &PyOde<Tx, Tf>::pysolve,
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
        .def("set_ics", &PyOde<Tx,Tf>::set_ics,
            py::arg("t0"),
            py::arg("f0"))
        .def("IntegrateAll", &PyOde<Tx, Tf>::IntAll,
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