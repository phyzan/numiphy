#ifndef ODES_HPP
#define ODES_HPP

#include <chrono>
#include <functional>
#include <vector>
#include "arrays.hpp"


template<class Tx, class Tf>
using ode = Tf(*)(const Tx&, const Tf&, const std::vector<Tx>&);

template<class Tx, class Tf>
using stepfunc = Tf(*)(ode<Tx, Tf>, const Tx&, const Tf&, const Tx&, const std::vector<Tx>&);

template<class Tx, class Tf>
using cond = std::function<bool(const Tx&, const Tx&, const Tf&, const Tf&)>;

template<class Any, class prec>
prec bisect_right(const Any& obj, const prec& a, const prec& b, const prec& tol = 1e-15);

template<class Tx, class Tf>
Tf _euler(ode<Tx, Tf> ode, const Tx& x, const Tf& f, const Tx& dx, const std::vector<Tx>& args);

template<class Tx, class Tf>
Tf _RK2(ode<Tx, Tf> ode, const Tx& x, const Tf& f, const Tx& dx, const std::vector<Tx>& args);

template<class Tx, class Tf>
Tf _RK4(ode<Tx, Tf> ode, const Tx& x, const Tf& f, const Tx& dx, const std::vector<Tx>& args);

template<class Tx, class Tf>
Tf _RK7(ode<Tx, Tf> ode, const Tx& x, const Tf& f, const Tx& dx, const std::vector<Tx>& args);

template<class Tx, class Tf>
Tf _cromer(ode<Tx, Tf> ode, const Tx& x, const Tf& f, const Tx& dx, const std::vector<Tx>& args);

template<class Tx, class Tf>
Tf _verlet(ode<Tx, Tf> ode, const Tx& x, const Tf& f, const Tx& dx, const std::vector<Tx>& args);

template<class Tx, class Tf>
struct OdeResult{

    vec::HeapArray<Tx> x;
    vec::HeapArray<Tf> f;
    long double runtime;
    bool diverges;
};


template<class Tx, class Tf>
struct StepGetter{

    ode<Tx, Tf> df;
    stepfunc<Tx, Tf> update;
    Tx x_initial;
    Tf f_initial;
    cond<Tx, Tf> condition;
    const std::vector<Tx>& args;

    int objfun(const Tx& x) const;

    Tx get(const Tx& dx) const;
};

template<class Tx, class Tf>
struct ICS{
    Tx x0;
    Tf f0;
};


template<class Tx, class Tf>
struct OdeArgs{

    ICS<Tx, Tf> ics;
    Tx x;
    Tx dx;
    Tx err = 1e-8;
    std::string method = "RK4";
    int max_frames = -1;
    std::vector<Tx> args;
    cond<Tx, Tf> getcond = nullptr;
    cond<Tx, Tf> breakcond = nullptr;
};

template<class Tx, class Tf>
class ODE{

    static const std::unordered_map<std::string, stepfunc<Tx, Tf>> method_map;
    static const std::unordered_map<std::string, int> lte_map;

    public:

        //constructor
        ODE(ode<Tx, Tf> ode = nullptr);

        const ode<Tx, Tf> odefunc() const;

        const OdeResult<Tx, Tf> solve(const OdeArgs<Tx, Tf>&) const;

        const std::vector<OdeResult<Tx, Tf>> solve_all(const std::vector<OdeArgs<Tx, Tf>>&, int threads=-1) const;

    protected:
        ode<Tx, Tf> _ode;

};

template<class Tx, class Tf>
struct OdeSet{

    ODE<Tx, Tf> ode;
    OdeArgs<Tx, Tf> params;

};

template<class Tx, class Tf>
std::vector<OdeResult<Tx, Tf>> dsolve_all(const std::vector<OdeSet<Tx, Tf>>& datam, int threads);


/*
-----------------------------------------------------------------------
-----------------------------DEFINITIONS-------------------------------
-----------------------------------------------------------------------
*/







template<class Any, class prec>
prec bisect_right(const Any& obj, const prec& a, const prec& b, const prec& tol){
    prec err = 2*tol;
    prec _a = a;
    prec _b = b;
    prec c;
    prec fm;

    if (obj.objfun(a) * obj.objfun(b) > 0){
        throw std::runtime_error("Bisection failed: There might be no root in the given interval");
    }
    while (err > tol){
        c = (_a+_b)/2;
        if (c == _a || c == _b){
            
            break;
        }
        fm = obj.objfun(c);
        if (obj.objfun(_a) * fm  > 0){
            _a = c;
        }
        else{
            _b = c;
        }
        err = abs(_b-_a);
    }
    return _b;
}


// DEFINE STEP FUNCTIONS
template<class Tx, class Tf>
Tf _euler(ode<Tx, Tf> ode, const Tx& x, const Tf& f, const Tx& dx, const std::vector<Tx>& args){
    return f + dx*ode(x, f, args);
}


template<class Tx, class Tf>
Tf _RK2(ode<Tx, Tf> ode, const Tx& x, const Tf& f, const Tx& dx, const std::vector<Tx>& args){
    Tf k1 = ode(x, f, args);
    Tf k2 = ode(x+dx/2, f+dx*k1/2, args);
    return f + dx*k2;
}


template<class Tx, class Tf>
Tf _RK4(ode<Tx, Tf> ode, const Tx& x, const Tf& f, const Tx& dx, const std::vector<Tx>& args){
    Tf k1 = ode(x, f, args);
    Tf k2 = ode(x+dx/2, f+dx*k1/2, args);
    Tf k3 = ode(x+dx/2, f+dx*k2/2, args);
    Tf k4 = ode(x+dx, f+dx*k3, args);
    return f + dx/6*(k1+2*k2+2*k3+k4);
}


template<class Tx, class Tf>
Tf _RK7(ode<Tx, Tf> ode, const Tx& x, const Tf& f, const Tx& dx, const std::vector<Tx>& args) {
    Tf k1 = dx * ode(x, f, args);
    Tf k2 = dx * ode(x + dx / 12, f + k1 / 12, args);
    Tf k3 = dx * ode(x + dx / 12, f + (11 * k2 - 10 * k1) / 12, args);
    Tf k4 = dx * ode(x + dx / 6, f + k3 / 6, args);
    Tf k5 = dx * ode(x + dx / 3, f + (157 * k1 - 318 * k2 + 4 * k3 + 160 * k4) / 9, args);
    Tf k6 = dx * ode(x + dx / 2, f + (-322 * k1 + 199 * k2 + 108 * k3 - 131 * k5) / 30, args);
    Tf k7 = dx * ode(x + 8 * dx / 12, f + 3158 * k1 / 45 - 638 * k2 / 6 - 23 * k3 / 2 + 157 * k4 / 3 + 157 * k6 / 45, args);
    Tf k8 = dx * ode(x + 10 * dx / 12, f - 53 * k1 / 14 + 38 * k2 / 7 - 3 * k3 / 14 - 65 * k5 / 72 + 29 * k7 / 90, args);
    Tf k9 = dx * ode(x + dx, f + 56 * k1 / 25 + 283 * k2 / 14 - 119 * k3 / 6 - 26 * k4 / 7 - 13 * k5 / 15 + 149 * k6 / 32 - 25 * k7 / 9 + 27 * k8 / 25, args);
    return f + (41 * k1 + 216 * k4 + 27 * k5 + 272 * k6 + 27 * k7 + 216 * k8 + 41 * k9) / 840;
}


template<class Tx, class Tf>
Tf _cromer(ode<Tx, Tf> ode, const Tx& x, const Tf& f, const Tx& dx, const std::vector<Tx>& args){

    Tf fnew = f.empty_copy();
    size_t nd = f.size()/2;
    Tf fdot = ode(x, f, args);

    for (int i=nd; i<2*nd; i++){
        fnew[i] = f[i] + dx*fdot[i];
    }

    for (int i=0; i<nd; i++){
        fnew[i] = f[i] + dx*fnew[i+nd];
    }

    return fnew;
}


template<class Tx, class Tf>
Tf _verlet(ode<Tx, Tf> ode, const Tx& x, const Tf& f, const Tx& dx, const std::vector<Tx>& args){
    
    Tf fnew = f.empty_copy();
    size_t nd = f.size()/2;
    Tf fnow = ode(x, f, args);

    for (int i=0; i<nd; i++){
        fnew[i] = f[i] + dx*f[i+nd] + fnow[i+nd]*dx*dx/2;
    }

    Tf fnext = ode(x+dx, fnew, args);

    for (int i=nd; i<2*nd; i++){
        fnew[i] = f[i] + (fnow[i]+fnext[i])*dx/2;
    }

    return fnew;
}



template<class Tx, class Tf>
int StepGetter<Tx, Tf>::objfun(const Tx& x) const{
    Tf f = update(df, x_initial, f_initial, x-x_initial, args);
    return (condition(x_initial, x, f_initial, f) > 0) ? 1: -1;
}

template<class Tx, class Tf>
Tx StepGetter<Tx, Tf>::get(const Tx& dx) const{
    return bisect_right(*this, x_initial, x_initial+dx) - x_initial;
}

template<class Tx, class Tf>
const std::unordered_map<std::string, stepfunc<Tx, Tf>> 
ODE<Tx, Tf>::method_map = {
    {"euler", _euler},
    {"RK2", _RK2},
    {"RK4", _RK4},
    {"RK7", _RK7}
};
template<class Tx, class Tf>
const std::unordered_map<std::string, int> 
ODE<Tx, Tf>::lte_map = {
    {"euler", 2},
    {"RK2", 3},
    {"RK4", 5},
    {"RK7", 8}
};

template<class Tx, class Tf>
ODE<Tx, Tf>::ODE(ode<Tx, Tf> ode): _ode(ode) {}

template<class Tx, class Tf>
const ode<Tx, Tf> ODE<Tx, Tf>::odefunc() const {
    return _ode;
}

template<class Tx, class Tf>
const std::vector<OdeResult<Tx, Tf>> ODE<Tx, Tf>::solve_all(const std::vector<OdeArgs<Tx, Tf>>& args, int threads) const{

    //define all sets of ode-args
    std::vector<OdeSet<Tx, Tf>> data(args.size());
    for (size_t i=0; i<args.size(); i++){
        data[i] = {*this, args[i]};
    }

    std::vector<OdeResult<Tx, Tf>> res = dsolve_all(data, threads);

    return res;
}

template<class Tx, class Tf>
const OdeResult<Tx, Tf> ODE<Tx, Tf>::solve(const OdeArgs<Tx, Tf>& params) const {
    //extract data from args first
    const ICS<Tx, Tf>& ics = params.ics;
    const Tx& x = params.x;
    Tx dx = params.dx;
    const Tx& err = params.err;
    const int& max_frames = params.max_frames;
    const std::vector<Tx>& args = params.args;
    const cond<Tx, Tf>& getcond = params.getcond;
    const cond<Tx, Tf>& breakcond = params.breakcond;
    const int lte = lte_map.find(params.method)->second;
    auto it = method_map.find(params.method);
    if (it == method_map.end()) {
        throw std::runtime_error("Unknown ode method");
    }
    stepfunc<Tx, Tf> update = it->second;
    
    const int _dir = vec::sign(dx);
    const Tx _pow = Tx(1)/lte;
    const Tx POINT_NINE = Tx(9)/10;
    const Tx& _x0 = ics.x0;
    const Tf& _f0 = ics.f0;

    bool ready = false;
    bool capture = false;
    bool diverges = false;
    int k = 1;

    Tx xi = _x0;
    Tf fi = _f0;

    Tx rel_err;
    Tf f_single;
    Tf f_half;
    Tf f_double;
    Tf f_tmp;

    vec::HeapArray<Tx> x_arr;
    vec::HeapArray<Tf> f_arr;

    if (dx*_x0 > x*dx){
        throw std::runtime_error("wrong direction of ode integration");
    }

    if (max_frames > 0){
        x_arr.allocate(max_frames);
        f_arr.allocate(max_frames);
    }

    x_arr.append(xi);
    f_arr.append(fi);
    auto t1 = std::chrono::high_resolution_clock::now();
    while (!ready){
        
        //determine step size
        rel_err = 2*err;
        while (rel_err > err){
            f_single = update(_ode, xi, fi, dx, args);
            f_half = update(_ode, xi, fi, dx/2, args);
            f_double = update(_ode, xi+dx/2, f_half, dx/2, args);

            try{
                rel_err = vec::safemax(vec::abs((f_single - f_double)/f_double));
            }
            catch (std::overflow_error& e){
                break;
            }

            if (rel_err != 0.){
                dx = POINT_NINE*dx*pow(err/rel_err, _pow); //TODO for e.g. mpfr, the mpfr::pow must be invoked
            }
            else{
                break;
            }
        }
        if ((xi+dx)*_dir > x*_dir){
            dx = x-xi;
            ready=true;
        }//step size determined
        
        f_tmp = update(_ode, xi, fi, dx, args);
        if (has_nan_inf(f_tmp)){
            diverges = true;
            break;
        }
        
        if ( (breakcond != nullptr) && breakcond(xi, xi+dx, fi, f_tmp)){
            ready = true;
            StepGetter<Tx, Tf> sg{_ode, update, xi, fi, breakcond, args};
            dx = sg.get(dx);
            fi = update(_ode, xi, fi, dx, args);
        }
        else if ( (getcond != nullptr) && getcond(xi, xi+dx, fi, f_tmp)){
            capture = true;
            StepGetter<Tx, Tf> sg{_ode, update, xi, fi, getcond, args};
            dx = sg.get(dx);
            fi = update(_ode, xi, fi, dx, args);
        }
        else{
            fi = f_tmp;
        }

        xi += dx;
        if (getcond != nullptr){
            if (capture){
                f_arr.append(fi);
                x_arr.append(xi);
                capture = false;
                k += 1;
            }
        }
        else if ( (max_frames == -1) || (std::abs(xi-_x0)*(max_frames-1) >= k*std::abs(x-_x0)) ){
            f_arr.append(fi);
            x_arr.append(xi);
            k += 1;
        }

        if (k == max_frames){
            ready = true;
        }


    }
    auto t2 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> time = t2-t1;
    OdeResult<Tx, Tf> res{x_arr, f_arr, time.count(), diverges};

    return res;
}

template<class Tx, class Tf>
std::vector<OdeResult<Tx, Tf>> dsolve_all(const std::vector<OdeSet<Tx, Tf>>& data, int threads){
    const size_t n = data.size();
    std::vector<OdeResult<Tx, Tf>> res(n);

    threads = (threads == -1) ? omp_get_max_threads() : threads;
    #pragma omp parallel for schedule(dynamic) num_threads(threads)
    for (size_t i=0; i<n; i++){
        res[i] = data[i].ode.solve(data[i].params);
    }

    return res;
}


#endif
