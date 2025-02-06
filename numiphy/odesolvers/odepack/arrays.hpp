#ifndef ARRAYS_HPP
#define ARRAYS_HPP

#include <iostream>
#include <cmath>
#include <omp.h>

namespace arr{

using std::sin, std::cos, std::tan, std::abs;

template<class T>
class Array{

    public:

        //constructor
        Array(const size_t& size=0, const bool& fill=false);

        Array(const std::initializer_list<T> arr);

        //destructor
        inline ~Array(){delete[] _arr;}

        //copy constructor
        inline Array(const Array<T>& other){ _clone_from(other);}

        //assignment operator
        Array<T>& operator=(const Array<T>& other);

        //index operator
        T& operator[](int index);

        const T& operator[](int index) const;

        //identity operator
        inline Array<T>& operator+() const {return *this;}

        Array<T> operator-() const;

        Array<T>& operator+=(const Array<T>& other);

        Array<T>& operator+=(const T& other);

        Array<T>& operator-=(const Array<T>& other);

        Array<T>& operator-=(const T& other);

        inline size_t size() const{ return _size;}

        inline int max_size() const{return _alloc;}

        inline bool full() const { return _alloc == _size;}

        inline bool empty() const{ return _size == 0 && _alloc > 0;}

        inline auto max() const{ return max(*this);}

        inline auto min() const{ return min(*this);}

        inline Array<T> abs() const{ return abs(*this);}

        void allocate(size_t newSize);

        void try_alloc(size_t newSize);

        void fill(const T& value);

        void fill();

        Array<T> empty_copy() const;

        inline void fit(){ if (!full()) allocate(_size);}

        void append(const T& element);

        void append(const Array<T>& array);

        void pop();

        void show() const;

        const T* data() const;

        template<class R>
        friend Array<R> _operation(const Array<R>&, const R&, R (*oper)(const R&, const R&));

        template<class R>
        friend Array<R> _operation(const R&, const Array<R>&, R (*oper)(const R&, const R&));

        template<class R>
        friend Array<R> _operation(const Array<R>&, const Array<R>&, R (*oper)(const R&, const R&));

        template<class R>
        friend Array<R> _mathfunc(const Array<R>& x, R (*f)(const R&));

    private:
        size_t _alloc;//size of allocated array
        size_t _size;//actual size of content inside array. _size <= _alloc
        T* _arr = nullptr;//allocated array

        void _clone_from(const Array<T>& other);

        void _check_compatibility(const Array<T>& other) const;
};



template<class T, size_t N>
class StaticArray{

    public:

        //default constructor
        StaticArray(){};

        //constructor
        StaticArray(const std::initializer_list<T> arr);

        //index operator
        T& operator[](int index);

        const T& operator[](int index) const;

        inline StaticArray<T, N>& operator+(){return *this;}

        StaticArray<T, N> operator=(const StaticArray<T, N>& other);

        StaticArray<T, N> operator-() const;

        StaticArray<T, N>& operator+=(const StaticArray<T, N>& other);

        StaticArray<T, N>& operator+=(const T& other);

        StaticArray<T, N>& operator-=(const StaticArray<T, N>& other);

        StaticArray<T, N>& operator-=(const T& other);

        inline size_t size() const{ return _size;}
        
        inline auto max() const{ return max(*this);}

        inline auto min() const{ return min(*this);}

        inline StaticArray<T, N> abs() const{ return abs(*this);}

        inline void fit(){}

        inline void try_alloc(size_t newSize) const{};

        void fill(const T& value);

        void fill() {}

        StaticArray<T, N> empty_copy() const;

        void show() const;

        const T* data() const;

        template<class R, size_t n>
        friend StaticArray<R, n> _operation(const StaticArray<R, n>&, const R&, R (*oper)(const R&, const R&));

        template<class R, size_t n>
        friend StaticArray<R, n> _operation(const R&, const StaticArray<R, n>&, R (*oper)(const R&, const R&));

        template<class R, size_t n>
        friend StaticArray<R, n> _operation(const StaticArray<R, n>&, const StaticArray<R, n>&, R (*oper)(const R&, const R&));

        template<class R, size_t n>
        friend StaticArray<R, n> _mathfunc(const StaticArray<R, n>& x, R (*f)(const R&));

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


template<class R>
Array<R> _operation(const Array<R>& A, const R& b, R (*oper)(const R&, const R&));


template<class R>
Array<R> _operation(const R& b, const Array<R>& A, R (*oper)(const R&, const R&));


template<class R>
Array<R> _operation(const Array<R>& A, const Array<R>& B, R (*oper)(const R&, const R&));



template<class R, size_t n>
StaticArray<R, n> _operation(const StaticArray<R, n>&, const R&, R (*oper)(const R&, const R&));

template<class R, size_t n>
StaticArray<R, n> _operation(const R&, const StaticArray<R, n>&, R (*oper)(const R&, const R&));

template<class R, size_t n>
StaticArray<R, n> _operation(const StaticArray<R, n>&, const StaticArray<R, n>&, R (*oper)(const R&, const R&));


inline auto add(const auto& a, const auto& b){return _operation(a, b, _add);}

inline auto sub(const auto& a, const auto& b){return _operation(a, b, _sub);}

inline auto mul(const auto& a, const auto& b){return _operation(a, b, _mul);}

inline auto div(const auto& a, const auto& b){return _operation(a, b, _div);}

inline auto pow(const auto& a, const auto& b){return _operation(a, b, _pow);}


inline auto operator+(const auto& arr1, const auto& arr2){ return add(arr1, arr2); }

inline auto operator-(const auto& arr1, const auto& arr2){ return sub(arr1, arr2);}

inline auto operator*(const auto& arr1, const auto& arr2){ return mul(arr1, arr2); }

inline auto operator/(const auto& arr1, const auto& arr2){ return div(arr1, arr2);}





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

template<class T>
auto max(const Array<T>& A){ return _max_impl(A); }

template<class T, size_t n>
auto max(const StaticArray<T, n>& A){return _max_impl(A);}

template<class T>
inline T safemax(const T& x){return x;}

template<class T>
auto safemax(const Array<T>& A){return _safemax_impl(A);}

template<class T, size_t n>
auto safemax(const StaticArray<T, n>& A){return _safemax_impl(A);}


template<class T>
inline T min(const T& x){return x;}

template<class T>
auto min(const Array<T>& A){ return _min_impl(A); }

template<class T, size_t n>
auto min(const StaticArray<T, n>& A){return _min_impl(A);}

template<class T>
inline T safemin(const T& x){return x;}

template<class T>
auto safemin(const Array<T>& A){return _safemin_impl(A);}

template<class T, size_t n>
auto safemin(const StaticArray<T, n>& A){return _safemin_impl(A);}

template<class T>
bool has_nan_inf(const T&);

template<class T>
bool has_nan_inf(const Array<T>&);

template<class T, size_t n>
bool has_nan_inf(const StaticArray<T, n>&);

bool _isValid(auto value);

template<class T>
const auto& nested_element(const Array<T>& A);

template<class T, size_t N>
const auto& nested_element(const StaticArray<T, N>& A);

template<class T>
const auto& nested_element(const T& A);




//DECLARE GENERAL MATH FUNCTIONS
template<class R>
inline Array<R> _mathfunc(const Array<R>& x, R (*f)(const R&));

template<class R, size_t n>
StaticArray<R, n> _mathfunc(const StaticArray<R, n>& x, R (*f)(const R&));

template<class T>
inline T abs(const T& x);

template<class T>
inline T sin(const T& x);

template<class T>
inline T cos(const T& x);

template<class T>
inline T tan(const T& x);


template<class T>
inline Array<T> abs(const Array<T>& x);

template<class T>
inline Array<T> sin(const Array<T>& x);

template<class T>
inline Array<T> cos(const Array<T>& x);

template<class T>
inline Array<T> tan(const Array<T>& x);


template<class T, size_t n>
inline StaticArray<T, n> abs(const StaticArray<T, n>& x);

template<class T, size_t n>
inline StaticArray<T, n> sin(const StaticArray<T, n>& x);

template<class T, size_t n>
inline StaticArray<T, n> cos(const StaticArray<T, n>& x);

template<class T, size_t n>
inline StaticArray<T, n> tan(const StaticArray<T, n>& x);




//ARRAY DEFINITION

template<class T>
Array<T>::Array(const std::initializer_list<T> arr){
    delete[] _arr;
    _arr = new T[arr.size()];
    std::copy(arr.begin(), arr.end(), _arr);
    _size = arr.size();
    _alloc = _size;
}

template<class T>
Array<T>::Array(const size_t& size, const bool& fill): _alloc(size), _size(fill*size){
    if (_alloc == 0){ _arr = nullptr;}
    else{delete[] _arr; _arr = new T[_alloc];}
}

template<class T>
Array<T>& Array<T>::operator=(const Array<T>& other){
    if (&other != this) _clone_from(other);
    return *this;
}

template<class T>
T& Array<T>::operator[](int index){
    _check_index(index, _size);
    return _arr[(index >= 0) ? index : _size+index];
}

template<class T>
const T& Array<T>::operator[](int index) const{
    _check_index(index, _size);
    return _arr[(index >= 0) ? index : _size+index];
}

template<class T>
Array<T> Array<T>::operator-() const {
    Array<T> res(_alloc); res._size = _size;
    for (size_t i=0; i<_size; i++){
        res._arr[i] = -_arr[i];
    }
    return res;
}

template<class T>
Array<T>& Array<T>::operator+=(const Array<T>& other){
    _check_compatibility(other);
    for (size_t i=0; i<_size; i++){
        _arr[i] += other._arr[i];
    }
    return *this;
}

template<class T>
Array<T>& Array<T>::operator+=(const T& other){
    for (size_t i=0; i<_size; i++){
        _arr[i] += other;
    }
    return *this;
}


template<class T>
Array<T>& Array<T>::operator-=(const Array<T>& other){
    _check_compatibility(other);
    for (size_t i=0; i<_size; i++){
        _arr[i] -= other._arr[i];
    }
    return *this;
}

template<class T>
Array<T>& Array<T>::operator-=(const T& other){
    for (size_t i=0; i<_size; i++){
        _arr[i] -= other;
    }
    return *this;
}

template<class T>
void Array<T>::allocate(size_t newSize){
    if (newSize < _size || newSize == 0) return;
    T* newArr = new T[newSize];
    
    for (size_t i=0; i<_size; i++){
        newArr[i] = _arr[i];
    }

    delete[] _arr;
    _alloc = newSize;
    _arr = newArr;
    
}

template<class T>
void Array<T>::try_alloc(size_t newSize){
    allocate(newSize);
}

template<class T>
void Array<T>::fill(const T& value){
    for (size_t i=0; i<_alloc; i++){
        _arr[i] = value;
    }
    _size = _alloc;
}

template<class T>
void Array<T>::fill(){
    _size = _alloc;
}

template<class T>
Array<T> Array<T>::empty_copy() const{
    Array<T> res(_size);
    res._size = res._alloc;
    return res;
}

template<class T>
void Array<T>::append(const T& element){
    if (full()) allocate( (_alloc == 0) ? 1 : 2*_alloc );
    _arr[_size++] = element;
}

template<class T>
void Array<T>::append(const Array<T>& array){
    if (_size + array._size > _alloc) allocate(_size + array._size);

    for (size_t i=0; i<array._size; i++){
        _arr[_size+i] = array._arr[i];
    }
    _size += array._size;
}

template<class T>
void Array<T>::pop(){
    if (_size > 0){
        _size--;
    }
}

template<class T>
void Array<T>::_clone_from(const Array<T>& other){
    delete[] _arr;

    _alloc = other._alloc;
    _size = other._size;
    _arr = new T[_alloc];
    for (size_t i = 0; i<_size; i++){
        _arr[i] = other._arr[i];
    }
}


template<class T>
void Array<T>::_check_compatibility(const Array<T>& other) const {
    if (_size != other._size){
        throw std::runtime_error("Cannot complete operation on arrays of different size");
    }
}

template<class T>
void Array<T>::show() const{
    std::cout << "\n[";
    for (size_t i=0; i<_size; i++){
        std::cout << " " << _arr[i];
    }
    std::cout << " ]\n";
}

template<class T>
const T* Array<T>::data() const{
    return _arr;
}









template<class T, size_t N>
StaticArray<T, N>::StaticArray(const std::initializer_list<T> arr){
    if (arr.size() != N) {
        throw std::out_of_range("Initializer list has a fixed size");
    }
    std::copy(arr.begin(), arr.end(), _arr);
}

template<class T, size_t N>
StaticArray<T, N> StaticArray<T, N>::operator=(const StaticArray<T, N>& other){
    if (&other != this){
        for (size_t i=0; i<size(); i++){
            _arr[i] = other._arr[i];
        }
    }
    return *this;
}

template<class T, size_t N>
T& StaticArray<T, N>::operator[](int index){
    _check_index(index, size());
    return _arr[(index >= 0) ? index : size()+index];
}

template<class T, size_t N>
const T& StaticArray<T, N>::operator[](int index) const{
    _check_index(index, size());
    return _arr[(index >= 0) ? index : size()+index];
}

template<class T, size_t N>
StaticArray<T, N> StaticArray<T, N>::operator-() const {
    StaticArray<T, N> res;
    for (size_t i=0; i<size(); i++){
        res._arr[i] = - _arr[i];
    }
    return res;
}


template<class T, size_t N>
StaticArray<T, N>& StaticArray<T, N>::operator+=(const StaticArray<T, N>& other){
    for (size_t i=0; i<size(); i++){
        _arr[i] += other._arr[i];
    }
    return *this;
}

template<class T, size_t N>
StaticArray<T, N>& StaticArray<T, N>::operator+=(const T& other){
    for (size_t i=0; i<size(); i++){
        _arr[i] += other;
    }
    return *this;
}


template<class T, size_t N>
StaticArray<T, N>& StaticArray<T, N>::operator-=(const StaticArray<T, N>& other){
    for (size_t i=0; i<size(); i++){
        _arr[i] -= other._arr[i];
    }
    return *this;
}

template<class T, size_t N>
StaticArray<T, N>& StaticArray<T, N>::operator-=(const T& other){
    for (size_t i=0; i<size(); i++){
        _arr[i] -= other;
    }
    return *this;
}


template<class T, size_t N>
void StaticArray<T, N>::fill(const T& value){
    for (size_t i=0; i<N; i++){
        _arr[i] = value;
    }
}

template<class T, size_t N>
StaticArray<T, N> StaticArray<T, N>::empty_copy() const{
    StaticArray<T, N> res;
    return res;
}

template<class T, size_t n>
void StaticArray<T, n>::show() const{
    std::cout << "\n[";
    for (size_t i=0; i<size(); i++){
        std::cout << " " << _arr[i];
    }
    std::cout << " ]\n";
}

template<class T, size_t n>
const T* StaticArray<T, n>::data() const{
    return _arr;
}




template<class R>
Array<R> _operation(const Array<R>& A, const R& b, R (*oper)(const R&, const R&)){
    Array<R> res(A._alloc);
    res._size = A._size;
    for (size_t i = 0; i < res._size; i++){
        res._arr[i] = oper(A._arr[i], b);
    }
    return res;
}


template<class R>
Array<R> _operation(const R& b, const Array<R>& A, R (*oper)(const R&, const R&)){
    Array<R> res(A._alloc);
    res._size = A._size;
    for (size_t i = 0; i < res._size; i++){
        res._arr[i] = oper(b, A._arr[i]);
    }
    return res;
}


template<class R>
Array<R> _operation(const Array<R>& A, const Array<R>& B, R (*oper)(const R&, const R&)){
    A._check_compatibility(B);
    Array<R> res( (A._alloc > B._alloc) ? A._alloc : B._alloc);
    res._size = A._size;
    for (size_t i = 0; i < res._size; i++){
        res._arr[i] = oper(A._arr[i], B._arr[i]);
    }
    return res;
}

//DEFINE ELEMENT-WISE MATH FUNCTIONS

template<class R>
Array<R> _mathfunc(const Array<R>& x, R (*f)(const R&)){
    Array<R> res(x._alloc);
    res._size = x._size;
    for (size_t i=0; i<res._size; i++){
        res._arr[i] = f(x._arr[i]);
    }
    return res;
}


template<class R, size_t n>
StaticArray<R, n> _operation(const StaticArray<R, n>& A, const R& b, R (*oper)(const R&, const R&)){
    StaticArray<R, n> res;
    for (size_t i=0; i < res.size(); i++){
        res._arr[i] = oper(A._arr[i], b);
    }
    return res;
}

template<class R, size_t n>
StaticArray<R, n> _operation(const R& b, const StaticArray<R, n>& A, R (*oper)(const R&, const R&)){
    StaticArray<R, n> res;
    for (size_t i=0; i < res.size(); i++){
        res._arr[i] = oper(b, A._arr[i]);
    }
    return res;
}

template<class R, size_t n>
StaticArray<R, n> _operation(const StaticArray<R, n>& A, const StaticArray<R, n>& B, R (*oper)(const R&, const R&)){
    StaticArray<R, n> res;
    for (size_t i=0; i < res.size(); i++){
        res._arr[i] = oper(A._arr[i], B._arr[i]);
    }
    return res;
}

template<class R, size_t n>
StaticArray<R, n> _mathfunc(const StaticArray<R, n>& x, R (*f)(const R&)){
    StaticArray<R, n> res;
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
        throw std::overflow_error("All numbers are nan or +-inf in the given array");
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
        throw std::overflow_error("All numbers are nan or +-inf in the given array");
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





template<class T>
inline Array<T> abs(const Array<T>& x){ return _mathfunc(x, abs);}

template<class T>
inline Array<T> sin(const Array<T>& x){ return _mathfunc(x, sin);}

template<class T>
inline Array<T> cos(const Array<T>& x){ return _mathfunc(x, cos);}

template<class T>
inline Array<T> tan(const Array<T>& x){ return _mathfunc(x, tan);}


template<class T, size_t n>
inline StaticArray<T, n> abs(const StaticArray<T, n>& x){ return _mathfunc(x, abs);}

template<class T, size_t n>
inline StaticArray<T, n> sin(const StaticArray<T, n>& x){ return _mathfunc(x, sin);}

template<class T, size_t n>
inline StaticArray<T, n> cos(const StaticArray<T, n>& x){ return _mathfunc(x, cos);}

template<class T, size_t n>
inline StaticArray<T, n> tan(const StaticArray<T, n>& x){ return _mathfunc(x, tan);}


bool _isValid(auto value) {
    return !std::isnan(value) && std::isfinite(value);
}


template<class T>
const auto& nested_element(const Array<T>& A){
    return nested_element(A[0]);
}

template<class T, size_t N>
const auto& nested_element(const StaticArray<T, N>& A){
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

template<class T>
bool has_nan_inf(const Array<T>& A){
    
    for(size_t i=0; i<A.size(); i++){
        if (has_nan_inf(A[i])){
            return true;
        }
    }
    return false;
}

template<class T, size_t n>
bool has_nan_inf(const StaticArray<T, n>& A){
    
    for(size_t i=0; i<A.size(); i++){
        if (has_nan_inf(A[i])){
            return true;
        }
    }
    return false;
}
}

#endif