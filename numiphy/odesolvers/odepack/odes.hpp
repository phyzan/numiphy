#ifndef ODES_HPP
#define ODES_HPP

#include <chrono>
#include <functional>
#include <vector>
#include "arrays.hpp"


template<class Tx, class Tf>
using ode = Tf(*)(const Tx&, const Tf&, const Tx*);

template<class Tx, class Tf>
using stepfunc = Tf(*)(ode<Tx, Tf>, const Tx&, const Tf&, const Tx&, const Tx*);

template<class Tx, class Tf>
using cond = std::function<bool(const Tx&, const Tx&, const Tf&, const Tf&)>;

template<class Any, class prec>
prec bisect_right(const Any& obj, const prec& a, const prec& b, const prec& tol = 1e-15);

template<class Tx, class Tf>
Tf _euler(ode<Tx, Tf> ode, const Tx& x, const Tf& f, const Tx& dx, const Tx* args);

template<class Tx, class Tf>
Tf _RK2(ode<Tx, Tf> ode, const Tx& x, const Tf& f, const Tx& dx, const Tx* args);

template<class Tx, class Tf>
Tf _RK4(ode<Tx, Tf> ode, const Tx& x, const Tf& f, const Tx& dx, const Tx* args);

template<class Tx, class Tf>
Tf _RK7(ode<Tx, Tf> ode, const Tx& x, const Tf& f, const Tx& dx, const Tx* args);

template<class Tx, class Tf>
Tf _cromer(ode<Tx, Tf> ode, const Tx& x, const Tf& f, const Tx& dx, const Tx* args);

template<class Tx, class Tf>
Tf _verlet(ode<Tx, Tf> ode, const Tx& x, const Tf& f, const Tx& dx, const Tx* args);

template<class Tx, class Tf>
struct OdeResult{
    arr::Array<Tx> x;
    arr::Array<Tf> f;
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
    const Tx* args;

    int objfun(const Tx& x) const;

    Tx get(const Tx& dx) const;
};

template<class Tx, class Tf>
struct ICS{
    Tx x0;
    Tf f0;
};


template<class Tx, class Tf>
class ODE{

    static const std::unordered_map<std::string, stepfunc<Tx, Tf>> method_map;
    static const std::unordered_map<std::string, int> lte_map;

    public:

        //constructor
        ODE() : _ode(nullptr) {};
        ODE(ode<Tx, Tf> ode);
        ODE(ode<Tx, Tf> ode, const Tx& x0, const Tf& f0);

        //destructor
        ~ODE();

        void set_ics(const Tx& x0, const Tf& f0);
        //dsolve methods

        const ICS<Tx, Tf>* get_ics() const;
        const ode<Tx, Tf> odefunc() const;

        const OdeResult<Tx, Tf> solve(const Tx& x, const Tx& dx, const double& err, const char* method = "RK4", const int max_frames = -1, const arr::Array<Tx>* args=nullptr, cond<Tx, Tf> getcond = nullptr, cond<Tx, Tf> breakcond = nullptr, const bool display = false) const;

    protected:
        ode<Tx, Tf> _ode;
        ICS<Tx, Tf>* ics = nullptr;

        //core algorithm
        const OdeResult<Tx, Tf> _dsolve(const Tx& x, Tx dx, stepfunc<Tx, Tf> update, const int lte, const double err, const int max_frames, const Tx* args, cond<Tx, Tf> getcond, cond<Tx, Tf> breakcond, const bool display) const;

};

// template<class Tx, size_t N>
// class HamiltonianSystem : public ODE<Tx, arr::StaticArray<Tx, 2*N>>{
//     using Tf = arr::StaticArray<Tx, 2*N>
//     public:
//         HamiltonianSystem() : ODE<Tx, Tf>() {};

//         HamiltonianSystem(vdotfunc<Tx, N> vdot) : ODE<Tx, Tf>(_to_odefunc(vdot)), _vdot(vdot) {};
        
//         OdeResult<Tx, Tf> cromer(const Tx& t, const Tx& dt, const int max_frames = -1, const bool display = false, cond<Tx, Tf> getcond = nullptr, cond<Tx, Tf> breakcond = nullptr) const;

//         OdeResult<Tx, Tf> verlet(const Tx& t, const Tx& dt, const int max_frames = -1, const bool display = false, cond<Tx, Tf> getcond = nullptr, cond<Tx, Tf> breakcond = nullptr) const;

//     private:
//         vdotfunc<Tx, N> _vdot;
// };





//DEFINITIONS

// template<class Tx, size_t nd>
// ode<Tx, arr::StaticArray<Tx, 2*nd>> _to_odefunc(vdotfunc<arr::StaticArray<Tx, nd>> vdot){
//     return [vdot](const Tx& t, const arr::StaticArray<Tx, 2*nd>& q){
//         arr::StaticArray<Tx, 2*nd> res;
//         arr::StaticArray<Tx, nd> x;
//         arr::StaticArray<Tx, nd> v;

//         for (size_t i=0; i<nd; i++){
//             x[i] = q[i];
//         }
//         arr::StaticArray<Tx, nd> a = vdot(x);

//         for (size_t i=0; i<nd; i++){
//             res[i] = q[i+nd];
//             res[i+nd] = a[i];
//         }
//         return res;
//     };
// }

// template<class Tx>
// ode<Tx, arr::Array<Tx>> _to_odefunc(vdotfunc<arr::Array<Tx>> vdot){
//     return [vdot](const Tx& t, const arr::Array<Tx>& q){
//         size_t nd = q.size()/2;
//         arr::Array<Tx> res(2*nd, true);
//         arr::Array<Tx> x(nd, true);
//         arr::Array<Tx> v(nd, true);

//         for (size_t i=0; i<nd; i++){
//             x[i] = q[i];
//         }
//         arr::Array<Tx> a = vdot(x);

//         for (size_t i=0; i<nd; i++){
//             res[i] = q[i+nd];
//             res[i+nd] = a[i];
//         }
//         return res;
//     };
// }


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
Tf _euler(ode<Tx, Tf> ode, const Tx& x, const Tf& f, const Tx& dx, const Tx* args){
    return f + dx*ode(x, f, args);
}


template<class Tx, class Tf>
Tf _RK2(ode<Tx, Tf> ode, const Tx& x, const Tf& f, const Tx& dx, const Tx* args){
    Tf k1 = ode(x, f, args);
    Tf k2 = ode(x+dx/2., f+dx*k1/2., args);
    return f + dx*k2;
}


template<class Tx, class Tf>
Tf _RK4(ode<Tx, Tf> ode, const Tx& x, const Tf& f, const Tx& dx, const Tx* args){
    Tf k1 = ode(x, f, args);
    Tf k2 = ode(x+dx/2., f+dx*k1/2., args);
    Tf k3 = ode(x+dx/2., f+dx*k2/2., args);
    Tf k4 = ode(x+dx, f+dx*k3, args);
    return f + dx/6.*(k1+2.*k2+2.*k3+k4);
}


template<class Tx, class Tf>
Tf _RK7(ode<Tx, Tf> ode, const Tx& x, const Tf& f, const Tx& dx, const Tx* args) {
    Tf k1 = dx * ode(x, f, args);
    Tf k2 = dx * ode(x + dx / 12., f + k1 / 12., args);
    Tf k3 = dx * ode(x + dx / 12., f + (11. * k2 - 10. * k1) / 12., args);
    Tf k4 = dx * ode(x + dx / 6., f + k3 / 6., args);
    Tf k5 = dx * ode(x + dx / 3., f + (157. * k1 - 318. * k2 + 4. * k3 + 160. * k4) / 9., args);
    Tf k6 = dx * ode(x + dx / 2., f + (-322. * k1 + 199. * k2 + 108. * k3 - 131. * k5) / 30., args);
    Tf k7 = dx * ode(x + 8. * dx / 12., f + 3158. * k1 / 45. - 638. * k2 / 6. - 23. * k3 / 2. + 157. * k4 / 3. + 157. * k6 / 45., args);
    Tf k8 = dx * ode(x + 10. * dx / 12., f - 53. * k1 / 14. + 38. * k2 / 7. - 3. * k3 / 14. - 65. * k5 / 72. + 29. * k7 / 90., args);
    Tf k9 = dx * ode(x + dx, f + 56. * k1 / 25. + 283. * k2 / 14. - 119. * k3 / 6. - 26. * k4 / 7. - 13. * k5 / 15. + 149. * k6 / 32. - 25. * k7 / 9. + 27. * k8 / 25., args);
    return f + (41. * k1 + 216. * k4 + 27. * k5 + 272. * k6 + 27. * k7 + 216. * k8 + 41. * k9) / 840.;
}


template<class Tx, class Tf>
Tf _cromer(ode<Tx, Tf> ode, const Tx& x, const Tf& f, const Tx& dx, const Tx* args){

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
Tf _verlet(ode<Tx, Tf> ode, const Tx& x, const Tf& f, const Tx& dx, const Tx* args){
    
    Tf fnew = f.empty_copy();
    size_t nd = f.size()/2;
    Tf fnow = ode(x, f, args);

    for (int i=0; i<nd; i++){
        fnew[i] = f[i] + dx*f[i+nd] + 0.5*fnow[i+nd]*dx*dx;
    }

    Tf fnext = ode(x+dx, fnew, args);

    for (int i=nd; i<2*nd; i++){
        fnew[i] = f[i] + (fnow[i]+fnext[i])*dx/2.;
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
ODE<Tx, Tf>::ODE(ode<Tx, Tf> ode, const Tx& x0, const Tf& f0): _ode(ode), ics(new ICS<Tx, Tf>{x0, f0}){}

template<class Tx, class Tf>
ODE<Tx, Tf>::~ODE(){
    delete ics;
    ics = nullptr;
}

template<class Tx, class Tf>
void ODE<Tx, Tf>::set_ics(const Tx& x0, const Tf& f0){
    delete ics;
    ics = new ICS<Tx, Tf>{x0, f0};
}

template<class Tx, class Tf>
const ICS<Tx, Tf>* ODE<Tx, Tf>::get_ics() const {
    return ics;
}

template<class Tx, class Tf>
const ode<Tx, Tf> ODE<Tx, Tf>::odefunc() const {
    return _ode;
}

template<class Tx, class Tf>
const OdeResult<Tx, Tf> ODE<Tx, Tf>::solve(const Tx& x, const Tx& dx, const double& err, const char* method, const int max_frames, const arr::Array<Tx>* args, cond<Tx, Tf> getcond, cond<Tx, Tf> breakcond, const bool display) const{
    auto it = method_map.find(method);
    if (it != method_map.end()) {
        return _dsolve(x, dx, it->second, lte_map.find(method)->second, err, max_frames, args->data(), getcond, breakcond, display);
    } else {
        throw std::runtime_error("Unknown ode method");
    }
}

template<class Tx, class Tf>
const OdeResult<Tx, Tf> ODE<Tx, Tf>::_dsolve(const Tx& x, Tx dx, stepfunc<Tx, Tf> update, const int lte, const double err, const int max_frames, const Tx* args, cond<Tx, Tf> getcond, cond<Tx, Tf> breakcond, const bool display) const {
    if (ics == nullptr){
        throw std::runtime_error("No initial conditions have been set");
    }
    Tx& _x0 = ics->x0;
    Tf& _f0 = ics->f0;
    if (dx*_x0 > x*dx){
        throw std::runtime_error("wrong direction of ode integration");
    }

    bool ready = false;
    bool capture = false;

    Tx xi = _x0;
    Tf fi = _f0;

    arr::Array<Tx> x_arr;
    arr::Array<Tf> f_arr;
    x_arr.append(xi);
    f_arr.append(fi);

    double _pow = 1./lte;
    int k = 1;
    int _dir = arr::sign(dx);
    Tx rel_err;
    Tf f_single;
    Tf f_half;
    Tf f_double;
    Tf f_tmp;
    bool diverges = false;

    auto t1 = std::chrono::high_resolution_clock::now();
    
    while (!ready){
        
        //determine step size
        rel_err = 2.*err;
        
        while (rel_err > err){
            f_single = update(_ode, xi, fi, dx, args);
            f_half = update(_ode, xi, fi, dx/2., args);
            f_double = update(_ode, xi+dx/2., f_half, dx/2., args);

            try{
                rel_err = arr::safemax(arr::abs((f_single - f_double)/f_double));
            }
            catch (std::overflow_error& e){
                break;
            }
            if (rel_err != 0.){
                dx = 0.9*dx*pow(err/rel_err, _pow);
            }
            else{
                break;
            }
        }
        

        if ((xi+dx)*_dir > x*_dir){
            dx = x-xi;
            ready=true;
        }
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
arr::Array<OdeResult<Tx, Tf>> IntegrateAll(const ODE<Tx, Tf>& ode, const arr::Array<ICS<Tx, Tf>>& ics, const bool& parallel, const Tx& x, const Tx& dx, const double& err, const char* method = "RK4", const int max_frames = -1, const arr::Array<Tx>* args=nullptr, cond<Tx, Tf> getcond = nullptr, cond<Tx, Tf> breakcond = nullptr){

    size_t n = ics.size();
    arr::Array<ODE<Tx, Tf>> ode_arr(n, true);
    arr::Array<OdeResult<Tx, Tf>> res(n, true);

    if (parallel){
        #pragma omp parallel for schedule(dynamic)
        for (size_t i=0; i<n; i++){
            ode_arr[i] = ode;
            ode_arr[i].set_ics(ics[i]->x0, ics[i]->f0);
            res[i] = ode_arr[i].solve(x, dx, err, method, max_frames, args, getcond, breakcond, false);
        }
    }
    else{
        for (size_t i=0; i<n; i++){
            ode_arr[i] = ode;
            ode_arr[i].set_ics(ics[i]->x0, ics[i]->f0);
            res[i] = ode_arr[i].solve(x, dx, err, method, max_frames, args, getcond, breakcond, false);
        }
    }

    return res;
}


#endif
