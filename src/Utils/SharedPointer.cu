#include <cstring>

#include <SharedPointer.cuh>

template <typename T>
static SharedPointer<T> SharedPointer<T>::makeShared(T&& obj)
{
#ifdef __CUDACC__
    T* pointer = nullptr;
    cudaMallocManaged(static_cast<void**>(&dPointer), sizeof(T));
    memcpy(static_cast<void*>(dPointer), static_cast<void*>(&obj), sizeof(T));
#else
    T* pointer = new T{obj};
#endif
    return SharedPointer<T>(pointer);
}

template <typename T>
SharedPointer<T>::SharedPointer(T* ptr) : Pointer{ptr} {}

template <typename T>
SharedPointer<T>::SharedPointer(const SharedPointer<T>& other) : ptr{other.ptr}, refcnt{other.refcnt}
{ 
    incRef();
}

template <typename T>
SharedPointer<T>& SharedPointer<T>::operator=(const SharedPointer<T>& other)
{
    ptr = other.ptr; 
    refcnt = other.refcnt; 
    incRef();
    return *this;
}