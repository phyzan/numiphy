#ifndef ARRAYS_HPP
#define ARRAYS_HPP

#include <iostream>
#include <cmath>
#include <omp.h>

namespace vec{

using std::sin, std::cos, std::tan, std::abs;

template<class T, size_t N = 0>
class HeapArray{

    public:

        //constructor
        HeapArray(const size_t& size=0, const bool& fill=false);

        HeapArray(const std::initializer_list<T> arr);

        //destructor
        inline ~HeapArray(){delete[] _arr;}

        //copy constructor
        inline HeapArray(const HeapArray<T, N>& other){ _clone_from(other);}

        //assignment operator
        HeapArray<T, N>& operator=(const HeapArray<T, N>& other);

        //index operator
        T& operator[](int index);

        const T& operator[](int index) const;

        //identity operator
        inline HeapArray<T, N>& operator+() const {return *this;}

        HeapArray<T, N> operator-() const;

        HeapArray<T, N>& operator+=(const HeapArray<T, N>& other);

        HeapArray<T, N>& operator+=(const T& other);

        HeapArray<T, N>& operator-=(const HeapArray<T, N>& other);

        HeapArray<T, N>& operator-=(const T& other);

        inline size_t size() const{ return _size;}

        inline int max_size() const{return _alloc;}

        inline bool full() const { return _alloc == _size;}

        inline bool empty() const{ return _size == 0 && _alloc > 0;}

        inline auto max() const{ return max(*this);}

        inline auto min() const{ return min(*this);}

        inline HeapArray<T, N> abs() const{ return abs(*this);}

        void allocate(size_t newSize);

        void try_alloc(size_t newSize);

        void fill(const T& value);

        void fill();

        HeapArray<T, N> empty_copy() const;

        inline void fit(){ if (!full()) allocate(_size);}

        void append(const T& element);

        void append(const HeapArray<T, N>& HeapArray);

        void pop();

        void show() const;

        const T* data() const;

        std::vector<T> to_vector() const;

        template<class R, size_t n>
        friend HeapArray<R, n> _operation(const HeapArray<R, n>&, const R&, R (*oper)(const R&, const R&));

        template<class R, size_t n>
        friend HeapArray<R, n> _operation(const R&, const HeapArray<R, n>&, R (*oper)(const R&, const R&));

        template<class R, size_t n>
        friend HeapArray<R, n> _operation(const HeapArray<R, n>&, const HeapArray<R, n>&, R (*oper)(const R&, const R&));

        template<class R, size_t n>
        friend HeapArray<R, n> _mathfunc(const HeapArray<R, n>& x, R (*f)(const R&));

    private:
        size_t _alloc;//size of allocated HeapArray
        size_t _size;//actual size of content inside HeapArray. _size <= _alloc
        T* _arr = nullptr;//allocated HeapArray

        void _clone_from(const HeapArray<T, N>& other);

        void _check_compatibility(const HeapArray<T, N>& other) const;
};



template<class T, size_t N>
class StackArray{

    public:

        //default constructor
        StackArray(){};

        //constructor
        StackArray(const std::initializer_list<T> arr);

        //index operator
        T& operator[](int index);

        const T& operator[](int index) const;

        inline StackArray<T, N>& operator+(){return *this;}

        StackArray<T, N> operator=(const StackArray<T, N>& other);

        StackArray<T, N> operator-() const;

        StackArray<T, N>& operator+=(const StackArray<T, N>& other);

        StackArray<T, N>& operator+=(const T& other);

        StackArray<T, N>& operator-=(const StackArray<T, N>& other);

        StackArray<T, N>& operator-=(const T& other);

        inline size_t size() const{ return _size;}
        
        inline auto max() const{ return max(*this);}

        inline auto min() const{ return min(*this);}

        inline StackArray<T, N> abs() const{ return abs(*this);}

        inline void fit(){}

        inline void try_alloc(size_t newSize) const{};

        void fill(const T& value);

        void fill() {}

        StackArray<T, N> empty_copy() const;

        void show() const;

        const T* data() const;

        std::vector<T> to_vector() const;

        template<class R, size_t n>
        friend StackArray<R, n> _operation(const StackArray<R, n>&, const R&, R (*oper)(const R&, const R&));

        template<class R, size_t n>
        friend StackArray<R, n> _operation(const R&, const StackArray<R, n>&, R (*oper)(const R&, const R&));

        template<class R, size_t n>
        friend StackArray<R, n> _operation(const StackArray<R, n>&, const StackArray<R, n>&, R (*oper)(const R&, const R&));

        template<class R, size_t n>
        friend StackArray<R, n> _mathfunc(const StackArray<R, n>& x, R (*f)(const R&));

    private:

        const size_t _size = N;
        T _arr[N];
};


void _check_index(const int& index, const int& size);

int sign(double x);

// CORE OPERATIONS
template<class T>
inline T _add(const T& a, const T& b) {return a + b;}

template<class T>
inline T _sub(const T& a, const T& b) {return a - b;}

template<class T>
inline T _mul(const T& a, const T& b) {return a * b;}

template<class T>
inline T _div(const T& a, const T& b) {return a / b;}

template<class T>
inline T _pow(const T& a, const T& b) {return std::pow(a, b);}


template<class R, size_t n>
HeapArray<R, n> _operation(const HeapArray<R, n>& A, const R& b, R (*oper)(const R&, const R&));


template<class R, size_t n>
HeapArray<R, n> _operation(const R& b, const HeapArray<R, n>& A, R (*oper)(const R&, const R&));


template<class R, size_t n>
HeapArray<R, n> _operation(const HeapArray<R, n>& A, const HeapArray<R, n>& B, R (*oper)(const R&, const R&));



template<class R, size_t n>
StackArray<R, n> _operation(const StackArray<R, n>&, const R&, R (*oper)(const R&, const R&));

template<class R, size_t n>
StackArray<R, n> _operation(const R&, const StackArray<R, n>&, R (*oper)(const R&, const R&));

template<class R, size_t n>
StackArray<R, n> _operation(const StackArray<R, n>&, const StackArray<R, n>&, R (*oper)(const R&, const R&));



//--------------ADDITION------------------------

template<class T, size_t N, template<class, size_t> class D>
D<T, N> add(const D<T, N>& a, const T& b){return _operation(a, b, _add);}

template<class T, size_t N, template<class, size_t> class D>
D<T, N> add(const T& a, const D<T, N>& b){return _operation(a, b, _add);}

template<class T, size_t N, template<class, size_t> class D>
D<T, N> add(const D<T, N>& a, const D<T, N>& b){return _operation(a, b, _add);}

//int case
template<class T, size_t N, template<class, size_t> class D>
D<T, N> add(const D<T, N>& a, const int& b){return _operation(a, T(b), _add);}

template<class T, size_t N, template<class, size_t> class D>
D<T, N> add(const int& a, const D<T, N>& b){return _operation(T(a), b, _add);}

//--------------SUBTRACTION------------------------

template<class T, size_t N, template<class, size_t> class D>
D<T, N> sub(const D<T, N>& a, const T& b){return _operation(a, b, _sub);}

template<class T, size_t N, template<class, size_t> class D>
D<T, N> sub(const T& a, const D<T, N>& b){return _operation(a, b, _sub);}

template<class T, size_t N, template<class, size_t> class D>
D<T, N> sub(const D<T, N>& a, const D<T, N>& b){return _operation(a, b, _sub);}

//int case
template<class T, size_t N, template<class, size_t> class D>
D<T, N> sub(const D<T, N>& a, const int& b){return _operation(a, T(b), _sub);}

template<class T, size_t N, template<class, size_t> class D>
D<T, N> sub(const int& a, const D<T, N>& b){return _operation(T(a), b, _sub);}

//--------------MULTIPLICATION------------------------

template<class T, size_t N, template<class, size_t> class D>
D<T, N> mul(const D<T, N>& a, const T& b){return _operation(a, b, _mul);}

template<class T, size_t N, template<class, size_t> class D>
D<T, N> mul(const T& a, const D<T, N>& b){return _operation(a, b, _mul);}

template<class T, size_t N, template<class, size_t> class D>
D<T, N> mul(const D<T, N>& a, const D<T, N>& b){return _operation(a, b, _mul);}

//int case
template<class T, size_t N, template<class, size_t> class D>
D<T, N> mul(const D<T, N>& a, const int& b){return _operation(a, T(b), _mul);}

template<class T, size_t N, template<class, size_t> class D>
D<T, N> mul(const int& a, const D<T, N>& b){return _operation(T(a), b, _mul);}

//--------------DIVISION------------------------

template<class T, size_t N, template<class, size_t> class D>
D<T, N> div(const D<T, N>& a, const T& b){return _operation(a, b, _div);}

template<class T, size_t N, template<class, size_t> class D>
D<T, N> div(const T& a, const D<T, N>& b){return _operation(a, b, _div);}

template<class T, size_t N, template<class, size_t> class D>
D<T, N> div(const D<T, N>& a, const D<T, N>& b){return _operation(a, b, _div);}

//int case
template<class T, size_t N, template<class, size_t> class D>
D<T, N> div(const D<T, N>& a, const int& b){return _operation(a, T(b), _div);}

template<class T, size_t N, template<class, size_t> class D>
D<T, N> div(const int& a, const D<T, N>& b){return _operation(T(a), b, _div);}

//--------------POWER------------------------

template<class T, size_t N, template<class, size_t> class D>
D<T, N> pow(const D<T, N>& a, const T& b){return _operation(a, b, _pow);}

template<class T, size_t N, template<class, size_t> class D>
D<T, N> pow(const T& a, const D<T, N>& b){return _operation(a, b, _pow);}

template<class T, size_t N, template<class, size_t> class D>
D<T, N> pow(const D<T, N>& a, const D<T, N>& b){return _operation(a, b, _pow);}

//int case
template<class T, size_t N, template<class, size_t> class D>
D<T, N> pow(const D<T, N>& a, const int& b){return _operation(a, T(b), _pow);}

template<class T, size_t N, template<class, size_t> class D>
D<T, N> pow(const int& a, const D<T, N>& b){return _operation(T(a), b, _pow);}


/*
--------------------------------------------------------------------------------
-------------------------OPERATOR OVERLOADING-----------------------------------
--------------------------------------------------------------------------------
*/

//--------------ADDITION------------------------

template<class T, size_t N, template<class, size_t> class D>
D<T, N> operator+(const D<T, N>& a, const auto& b){return add(a, b);}

template<class T, size_t N, template<class, size_t> class D>
D<T, N> operator+(const auto& a, const D<T, N>& b){return add(a, b);}

template<class T, size_t N, template<class, size_t> class D>
D<T, N> operator+(const D<T, N>& a, const D<T, N>& b){return add(a, b);}


//--------------SUBTRACTION------------------------

template<class T, size_t N, template<class, size_t> class D>
D<T, N> operator-(const D<T, N>& a, const auto& b){return sub(a, b);}

template<class T, size_t N, template<class, size_t> class D>
D<T, N> operator-(const auto& a, const D<T, N>& b){return sub(a, b);}

template<class T, size_t N, template<class, size_t> class D>
D<T, N> operator-(const D<T, N>& a, const D<T, N>& b){return sub(a, b);}

//--------------MULTIPLICATION------------------------

template<class T, size_t N, template<class, size_t> class D>
D<T, N> operator*(const D<T, N>& a, const auto& b){return mul(a, b);}

template<class T, size_t N, template<class, size_t> class D>
D<T, N> operator*(const auto& a, const D<T, N>& b){return mul(a, b);}

template<class T, size_t N, template<class, size_t> class D>
D<T, N> operator*(const D<T, N>& a, const D<T, N>& b){return mul(a, b);}


//--------------DIVISION------------------------

template<class T, size_t N, template<class, size_t> class D>
D<T, N> operator/(const D<T, N>& a, const auto& b){return div(a, b);}

template<class T, size_t N, template<class, size_t> class D>
D<T, N> operator/(const auto& a, const D<T, N>& b){return div(a, b);}

template<class T, size_t N, template<class, size_t> class D>
D<T, N> operator/(const D<T, N>& a, const D<T, N>& b){return div(a, b);}




//DECLARE MAX/MIN MATH FUNCTIONS

template<class ArrayType>
auto _max_impl(const ArrayType&);

template<class ArrayType>
auto _safemax_impl(const ArrayType&);

template<class ArrayType>
auto _min_impl(const ArrayType&);

template<class ArrayType>
auto _safemin_impl(const ArrayType&);

template<class T>
inline T max(const T& x){return x;}

template<class T, size_t N>
auto max(const HeapArray<T, N>& A){ return _max_impl(A); }

template<class T, size_t n>
auto max(const StackArray<T, n>& A){return _max_impl(A);}

template<class T>
inline T safemax(const T& x){return x;}

template<class T, size_t N>
auto safemax(const HeapArray<T, N>& A){return _safemax_impl(A);}

template<class T, size_t n>
auto safemax(const StackArray<T, n>& A){return _safemax_impl(A);}


template<class T>
inline T min(const T& x){return x;}

template<class T, size_t N>
auto min(const HeapArray<T, N>& A){ return _min_impl(A); }

template<class T, size_t n>
auto min(const StackArray<T, n>& A){return _min_impl(A);}

template<class T>
inline T safemin(const T& x){return x;}

template<class T, size_t N>
auto safemin(const HeapArray<T, N>& A){return _safemin_impl(A);}

template<class T, size_t n>
auto safemin(const StackArray<T, n>& A){return _safemin_impl(A);}

template<class T>
bool has_nan_inf(const T&);

template<class T, size_t N>
bool has_nan_inf(const HeapArray<T, N>&);

template<class T, size_t n>
bool has_nan_inf(const StackArray<T, n>&);

bool _isValid(auto value);

template<class T, size_t N>
const auto& nested_element(const HeapArray<T, N>& A);

template<class T, size_t N>
const auto& nested_element(const StackArray<T, N>& A);

template<class T>
const auto& nested_element(const T& A);




//DECLARE GENERAL MATH FUNCTIONS
template<class R, size_t n>
inline HeapArray<R, n> _mathfunc(const HeapArray<R, n>& x, R (*f)(const R&));

template<class R, size_t n>
StackArray<R, n> _mathfunc(const StackArray<R, n>& x, R (*f)(const R&));

template<class T>
inline T abs(const T& x);

template<class T>
inline T sin(const T& x);

template<class T>
inline T cos(const T& x);

template<class T>
inline T tan(const T& x);


template<class T, size_t N>
inline HeapArray<T, N> abs(const HeapArray<T, N>& x);

template<class T, size_t N>
inline HeapArray<T, N> sin(const HeapArray<T, N>& x);

template<class T, size_t N>
inline HeapArray<T, N> cos(const HeapArray<T, N>& x);

template<class T, size_t N>
inline HeapArray<T, N> tan(const HeapArray<T, N>& x);


template<class T, size_t n>
inline StackArray<T, n> abs(const StackArray<T, n>& x);

template<class T, size_t n>
inline StackArray<T, n> sin(const StackArray<T, n>& x);

template<class T, size_t n>
inline StackArray<T, n> cos(const StackArray<T, n>& x);

template<class T, size_t n>
inline StackArray<T, n> tan(const StackArray<T, n>& x);




//HeapArray DEFINITION

template<class T, size_t N>
HeapArray<T, N>::HeapArray(const std::initializer_list<T> arr){
    delete[] _arr;
    _arr = new T[arr.size()];
    std::copy(arr.begin(), arr.end(), _arr);
    _size = arr.size();
    _alloc = _size;
}

template<class T, size_t N>
HeapArray<T, N>::HeapArray(const size_t& size, const bool& fill): _alloc(size), _size(fill*size){
    if (_alloc == 0){ _arr = nullptr;}
    else{delete[] _arr; _arr = new T[_alloc];}
}

template<class T, size_t N>
HeapArray<T, N>& HeapArray<T, N>::operator=(const HeapArray<T, N>& other){
    if (&other != this) _clone_from(other);
    return *this;
}

template<class T, size_t N>
T& HeapArray<T, N>::operator[](int index){
    _check_index(index, _size);
    return _arr[(index >= 0) ? index : _size+index];
}

template<class T, size_t N>
const T& HeapArray<T, N>::operator[](int index) const{
    _check_index(index, _size);
    return _arr[(index >= 0) ? index : _size+index];
}

template<class T, size_t N>
HeapArray<T, N> HeapArray<T, N>::operator-() const {
    HeapArray<T, N> res(_alloc); res._size = _size;
    for (size_t i=0; i<_size; i++){
        res._arr[i] = -_arr[i];
    }
    return res;
}

template<class T, size_t N>
HeapArray<T, N>& HeapArray<T, N>::operator+=(const HeapArray<T, N>& other){
    _check_compatibility(other);
    for (size_t i=0; i<_size; i++){
        _arr[i] += other._arr[i];
    }
    return *this;
}

template<class T, size_t N>
HeapArray<T, N>& HeapArray<T, N>::operator+=(const T& other){
    for (size_t i=0; i<_size; i++){
        _arr[i] += other;
    }
    return *this;
}


template<class T, size_t N>
HeapArray<T, N>& HeapArray<T, N>::operator-=(const HeapArray<T, N>& other){
    _check_compatibility(other);
    for (size_t i=0; i<_size; i++){
        _arr[i] -= other._arr[i];
    }
    return *this;
}

template<class T, size_t N>
HeapArray<T, N>& HeapArray<T, N>::operator-=(const T& other){
    for (size_t i=0; i<_size; i++){
        _arr[i] -= other;
    }
    return *this;
}

template<class T, size_t N>
void HeapArray<T, N>::allocate(size_t newSize){
    if (newSize < _size || newSize == 0) return;
    T* newArr = new T[newSize];
    
    for (size_t i=0; i<_size; i++){
        newArr[i] = _arr[i];
    }

    delete[] _arr;
    _alloc = newSize;
    _arr = newArr;
    
}

template<class T, size_t N>
void HeapArray<T, N>::try_alloc(size_t newSize){
    allocate(newSize);
}

template<class T, size_t N>
void HeapArray<T, N>::fill(const T& value){
    for (size_t i=0; i<_alloc; i++){
        _arr[i] = value;
    }
    _size = _alloc;
}

template<class T, size_t N>
void HeapArray<T, N>::fill(){
    _size = _alloc;
}

template<class T, size_t N>
HeapArray<T, N> HeapArray<T, N>::empty_copy() const{
    HeapArray<T, N> res(_size);
    res._size = res._alloc;
    return res;
}

template<class T, size_t N>
void HeapArray<T, N>::append(const T& element){
    if (full()) allocate( (_alloc == 0) ? 1 : 2*_alloc );
    _arr[_size++] = element;
}

template<class T, size_t N>
void HeapArray<T, N>::append(const HeapArray<T, N>& HeapArray){
    if (_size + HeapArray._size > _alloc) allocate(_size + HeapArray._size);

    for (size_t i=0; i<HeapArray._size; i++){
        _arr[_size+i] = HeapArray._arr[i];
    }
    _size += HeapArray._size;
}

template<class T, size_t N>
void HeapArray<T, N>::pop(){
    if (_size > 0){
        _size--;
    }
}

template<class T, size_t N>
void HeapArray<T, N>::_clone_from(const HeapArray<T, N>& other){
    delete[] _arr;

    _alloc = other._alloc;
    _size = other._size;
    _arr = new T[_alloc];
    for (size_t i = 0; i<_size; i++){
        _arr[i] = other._arr[i];
    }
}


template<class T, size_t N>
void HeapArray<T, N>::_check_compatibility(const HeapArray<T, N>& other) const {
    if (_size != other._size){
        throw std::runtime_error("Cannot complete operation on arrays of different size");
    }
}

template<class T, size_t N>
void HeapArray<T, N>::show() const{
    std::cout << "\n[";
    for (size_t i=0; i<_size; i++){
        std::cout << " " << _arr[i];
    }
    std::cout << " ]\n";
}

template<class T, size_t N>
const T* HeapArray<T, N>::data() const{
    return _arr;
}

template<class T, size_t N>
std::vector<T> HeapArray<T, N>::to_vector() const{
    std::vector<T> res(_size);
    for (size_t i=0; i<_size; i++){
        res[i] = _arr[i];
    }
    return res;
}







template<class T, size_t N>
StackArray<T, N>::StackArray(const std::initializer_list<T> arr){
    if (arr.size() != N) {
        throw std::out_of_range("Initializer list has a fixed size");
    }
    std::copy(arr.begin(), arr.end(), _arr);
}

template<class T, size_t N>
StackArray<T, N> StackArray<T, N>::operator=(const StackArray<T, N>& other){
    if (&other != this){
        for (size_t i=0; i<size(); i++){
            _arr[i] = other._arr[i];
        }
    }
    return *this;
}

template<class T, size_t N>
T& StackArray<T, N>::operator[](int index){
    _check_index(index, size());
    return _arr[(index >= 0) ? index : size()+index];
}

template<class T, size_t N>
const T& StackArray<T, N>::operator[](int index) const{
    _check_index(index, size());
    return _arr[(index >= 0) ? index : size()+index];
}

template<class T, size_t N>
StackArray<T, N> StackArray<T, N>::operator-() const {
    StackArray<T, N> res;
    for (size_t i=0; i<size(); i++){
        res._arr[i] = - _arr[i];
    }
    return res;
}


template<class T, size_t N>
StackArray<T, N>& StackArray<T, N>::operator+=(const StackArray<T, N>& other){
    for (size_t i=0; i<size(); i++){
        _arr[i] += other._arr[i];
    }
    return *this;
}

template<class T, size_t N>
StackArray<T, N>& StackArray<T, N>::operator+=(const T& other){
    for (size_t i=0; i<size(); i++){
        _arr[i] += other;
    }
    return *this;
}


template<class T, size_t N>
StackArray<T, N>& StackArray<T, N>::operator-=(const StackArray<T, N>& other){
    for (size_t i=0; i<size(); i++){
        _arr[i] -= other._arr[i];
    }
    return *this;
}

template<class T, size_t N>
StackArray<T, N>& StackArray<T, N>::operator-=(const T& other){
    for (size_t i=0; i<size(); i++){
        _arr[i] -= other;
    }
    return *this;
}


template<class T, size_t N>
void StackArray<T, N>::fill(const T& value){
    for (size_t i=0; i<N; i++){
        _arr[i] = value;
    }
}

template<class T, size_t N>
StackArray<T, N> StackArray<T, N>::empty_copy() const{
    StackArray<T, N> res;
    return res;
}

template<class T, size_t n>
void StackArray<T, n>::show() const{
    std::cout << "\n[";
    for (size_t i=0; i<size(); i++){
        std::cout << " " << _arr[i];
    }
    std::cout << " ]\n";
}

template<class T, size_t n>
const T* StackArray<T, n>::data() const{
    return _arr;
}

template<class T, size_t N>
std::vector<T> StackArray<T, N>::to_vector() const{
    std::vector<T> res(_size);
    for (size_t i=0; i<_size; i++){
        res[i] = _arr[i];
    }
    return res;
}




template<class R, size_t n>
HeapArray<R, n> _operation(const HeapArray<R, n>& A, const R& b, R (*oper)(const R&, const R&)){
    HeapArray<R, n> res(A._alloc);
    res._size = A._size;
    for (size_t i = 0; i < res._size; i++){
        res._arr[i] = oper(A._arr[i], b);
    }
    return res;
}


template<class R, size_t n>
HeapArray<R, n> _operation(const R& b, const HeapArray<R, n>& A, R (*oper)(const R&, const R&)){
    HeapArray<R, n> res(A._alloc);
    res._size = A._size;
    for (size_t i = 0; i < res._size; i++){
        res._arr[i] = oper(b, A._arr[i]);
    }
    return res;
}


template<class R, size_t n>
HeapArray<R, n> _operation(const HeapArray<R, n>& A, const HeapArray<R, n>& B, R (*oper)(const R&, const R&)){
    A._check_compatibility(B);
    HeapArray<R, n> res( (A._alloc > B._alloc) ? A._alloc : B._alloc);
    res._size = A._size;
    for (size_t i = 0; i < res._size; i++){
        res._arr[i] = oper(A._arr[i], B._arr[i]);
    }
    return res;
}

//DEFINE ELEMENT-WISE MATH FUNCTIONS

template<class R, size_t n>
HeapArray<R, n> _mathfunc(const HeapArray<R, n>& x, R (*f)(const R&)){
    HeapArray<R, n> res(x._alloc);
    res._size = x._size;
    for (size_t i=0; i<res._size; i++){
        res._arr[i] = f(x._arr[i]);
    }
    return res;
}


template<class R, size_t n>
StackArray<R, n> _operation(const StackArray<R, n>& A, const R& b, R (*oper)(const R&, const R&)){
    StackArray<R, n> res;
    for (size_t i=0; i < res.size(); i++){
        res._arr[i] = oper(A._arr[i], b);
    }
    return res;
}

template<class R, size_t n>
StackArray<R, n> _operation(const R& b, const StackArray<R, n>& A, R (*oper)(const R&, const R&)){
    StackArray<R, n> res;
    for (size_t i=0; i < res.size(); i++){
        res._arr[i] = oper(b, A._arr[i]);
    }
    return res;
}

template<class R, size_t n>
StackArray<R, n> _operation(const StackArray<R, n>& A, const StackArray<R, n>& B, R (*oper)(const R&, const R&)){
    StackArray<R, n> res;
    for (size_t i=0; i < res.size(); i++){
        res._arr[i] = oper(A._arr[i], B._arr[i]);
    }
    return res;
}

template<class R, size_t n>
StackArray<R, n> _mathfunc(const StackArray<R, n>& x, R (*f)(const R&)){
    StackArray<R, n> res;
    for (size_t i=0; i<n; i++){
        res[i] = f(x._arr[i]);
    }
    return res;
}




//DEFINE MAX, MIN FUNCTIONS

template<class ArrayType>
auto _max_impl(const ArrayType& A){
    auto res = max(A[0]);
    auto tmp = res;
    for(size_t i=0; i<A.size(); i++){
        tmp = max(A[i]);
        if (tmp > res){
            res = tmp;
        }
    }
    return res;
}

template<class ArrayType>
auto _safemax_impl(const ArrayType& A){
    unsigned int i = 0;
    auto res = nested_element(A); // only so that "auto" works
    while (i < A.size()){
        res = safemax(A[i]);
        if (_isValid(res)){
            break;
        }
        i++;
    }

    if (i == A.size()){
        throw std::overflow_error("All numbers are nan or +-inf in the given HeapArray");
    }

    auto tmp = res;
    while (i < A.size()){
        tmp = safemax(A[i]);
        if (tmp > res && _isValid(tmp)){
            res = tmp;
        }
        i++;
    }
    return res;
}


template<class ArrayType>
auto _min_impl(const ArrayType& A){
    auto res = min(A[0]);
    auto tmp = res;
    for(size_t i=0; i<A.size(); i++){
        tmp = min(A[i]);
        if (tmp < res){
            res = tmp;
        }
    }
    return res;
}

template<class ArrayType>
auto _safemin_impl(const ArrayType& A){
    unsigned i = 0;
    auto res = nested_element(A); // only so that "auto" works
    while (i < A.size()){
        res = safemin(A[i]);
        if (_isValid(res)){
            break;
        }
        i++;
    }

    if (i == A.size()){
        throw std::overflow_error("All numbers are nan or +-inf in the given HeapArray");
    }

    auto tmp = res;
    while (i < A.size()){
        tmp = safemax(A[i]);
        if (tmp < res && _isValid(tmp)){
            res = tmp;
        }
        i++;
    }
    return res;
}


template<class T>
inline T abs(const T& x) {return abs(x);}

template<class T>
inline T sin(const T& x) {return sin(x);}

template<class T>
inline T cos(const T& x) {return cos(x);}

template<class T>
inline T tan(const T& x) {return tan(x);}





template<class T, size_t N>
inline HeapArray<T, N> abs(const HeapArray<T, N>& x){ return _mathfunc(x, abs);}

template<class T, size_t N>
inline HeapArray<T, N> sin(const HeapArray<T, N>& x){ return _mathfunc(x, sin);}

template<class T, size_t N>
inline HeapArray<T, N> cos(const HeapArray<T, N>& x){ return _mathfunc(x, cos);}

template<class T, size_t N>
inline HeapArray<T, N> tan(const HeapArray<T, N>& x){ return _mathfunc(x, tan);}


template<class T, size_t n>
inline StackArray<T, n> abs(const StackArray<T, n>& x){ return _mathfunc(x, abs);}

template<class T, size_t n>
inline StackArray<T, n> sin(const StackArray<T, n>& x){ return _mathfunc(x, sin);}

template<class T, size_t n>
inline StackArray<T, n> cos(const StackArray<T, n>& x){ return _mathfunc(x, cos);}

template<class T, size_t n>
inline StackArray<T, n> tan(const StackArray<T, n>& x){ return _mathfunc(x, tan);}


bool _isValid(auto value) {
    return !std::isnan(value) && std::isfinite(value);
}


template<class T, size_t N>
const auto& nested_element(const HeapArray<T, N>& A){
    return nested_element(A[0]);
}

template<class T, size_t N>
const auto& nested_element(const StackArray<T, N>& A){
    return nested_element(A[0]);
}

template<class T>
const auto& nested_element(const T& A){
    return A;
}

template<class T>
bool has_nan_inf(const T& A){
    return !_isValid(A);
}

template<class T, size_t N>
bool has_nan_inf(const HeapArray<T, N>& A){
    
    for(size_t i=0; i<A.size(); i++){
        if (has_nan_inf(A[i])){
            return true;
        }
    }
    return false;
}

template<class T, size_t n>
bool has_nan_inf(const StackArray<T, n>& A){
    
    for(size_t i=0; i<A.size(); i++){
        if (has_nan_inf(A[i])){
            return true;
        }
    }
    return false;
}

void _check_index(const int& index, const int& size){
    if (index >= 0){
        if (index < size){return;}
    }
    else{
        if (index > -size-1){return;}
    }
    throw std::out_of_range("Index " + std::to_string(index) + " is out of range for an HeapArray of size " + std::to_string(size));
}


int sign(double x){
    if (x>0){
        return 1;
    }
    else if (x<0){
        return -1;
    }
    else{
        return 0;
    }
}

}

#endif