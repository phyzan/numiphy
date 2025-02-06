// #ifndef ORBITS_HPP
// #define ORBITS_HPP

// #include "odes.hpp"

// template<class prec, unsigned int dim=3>
// class Orbit{

//     using timestep = StaticArray<prec, 2*dim>;//a timestep is at least 2D: x,p_x
//     using forcefunc = timestep(*)(const prec&, const timestep&);

//     public:

//         static const unsigned int nd = dim;
//         static const unsigned int Nd = 2*dim;
//         const prec t0;
//         timestep q0;
        
//         Orbit();//default constructor

//         Orbit(forcefunc f, prec t0, timestep q0);

//         timestep qdot(const prec& t, timestep& q) const;//returns *qdot etc

//         void integrate(const prec& Delta_t, const prec& dt, const prec& err);

//         const Array<timestep>& trajectory() const;

//     private:

//         Array<prec> _t;
//         Array<timestep> _q;
        
//         const forcefunc _qdot_ptr=nullptr;
//         // const ODE<prec, timestep> _ode;
// };

// //DEFINE
// template<class prec, unsigned int dim>
// Orbit<prec, dim>::Orbit(){}

// template<class prec, unsigned int dim>
// Orbit<prec, dim>::Orbit(forcefunc f, prec t_0, timestep q_0): _qdot_ptr(f), t0(t_0), q0(q_0), _t({t_0}), _q({q_0}){}


// template<class prec, unsigned int dim>
// StaticArray<prec, 2*dim> Orbit<prec, dim>::qdot(const prec& t, timestep& q) const{
//     return _qdot_ptr(t, q);
// }


// template<class prec, unsigned int dim>
// const Array<StaticArray<prec, 2*dim>>& Orbit<prec, dim>::trajectory() const{
//     return _q;
// }

// template<class prec, unsigned int dim>
// void Orbit<prec, dim>::integrate(const prec& Delta_t, const prec& dt, const prec& err){
    
//     const ODE<prec, timestep> ode(_qdot_ptr, _t[-1], _q[-1]);

//     const OdeResult<prec, timestep> res = ode.RK4(_t[-1]+Delta_t, dt, err);
//     _t.pop();
//     _q.pop();

//     _t.append(res.x);
//     _q.append(res.f);

// }
// #endif