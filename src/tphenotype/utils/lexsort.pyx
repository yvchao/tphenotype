# distutils: language = c++
from libcpp.vector cimport vector

ctypedef fused complex_t:
    float complex
    double complex

ctypedef fused float_t:
    float
    double

cdef extern from * nogil:
    """
    #include <vector>
    #include <numeric>      // std::iota
    #include <algorithm> 
    #include <complex>
    #include <cmath>
    
    template <typename T>
    bool cmp(const std::complex<T> &a, const std::complex<T> &b, double threshold=1.0) { 
        std::complex<T> diff = a - b;
        if (std::abs(diff.real()) <= threshold){
            return diff.imag() <= 0.0;
        }
        else{
            return diff.real() <= 0.0;
        }
    }
    
    template <class RandomIt>
    std::vector<int> sort_indexes(const RandomIt &first, int length, double threshold=1.0) {
      std::vector<int> idx(length);
      std::iota(idx.begin(), idx.end(), 0);

      std::stable_sort(idx.begin(), idx.end(),
           [&first, threshold](int i1, int i2) {
           return cmp(*(first+i1), *(first+i2), threshold);
           });

      return idx;
    }
    """
    vector[int] sort_indexes[Iter](const Iter &first, int length, double threshold)  except +


# needs to be compiled with c++        
def sort_complex(complex_t [::1] a, float_t threshold=1.0):
    # a must be c continuous (enforced with [::1])
    idx = sort_indexes(&a[0], a.shape[0], threshold)
    return idx
