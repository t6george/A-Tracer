#include <Pointer.cuh>
#include <Util.h>

template <typename T>
Pointer<T>::Pointer(T* ptr) : refcnt{ptr ? 
    #ifdef __CUDACC__
    cudaMallocManaged(static_cast<void**>(&refcnt), sizeof(int))
    #else
    new int{1}
    #endif
    : nullptr},
    ptr{ptr} 
{
#ifdef __CUDACC__
    cudaDeviceSynchronize();
    *refcnt = 1;
#endif
}

template <typename T>
Pointer<T>::~Pointer()
{
    decRef();
    if (*refcnt == 0)
    {
        destroy();
    }
}

template <typename T>
Pointer<T>::destroy()
{
#ifdef __CUDACC__
    cudaDeviceSynchronize();
    cudaFree(ptr);
    cudaFree(refcnt);
#else
    delete ptr;
    delete refcnt;
#endif
    ptr = nullptr;
    refcnt = nullptr;
}

void incRef() noexcept
{
    refcnt && ++(*refcnt);
}

void decRef() noexcept
{
    refcnt && --(*refcnt);
}

template <typename T>
Pointer<T>::swap(SharedPointer<T>& other)
{ 
    utils::swap(ptr, other.ptr);
    utils::swap(refcnt, other.refcnt);
}

template <typename T>
T* Pointer<T>::get() const
{
    return ptr;
}

template <typename T>
T& Pointer<T>::operator *() const
{
    return *ptr;
}

template <typename T>
T* Pointer<T>::operator ->() const
{
    return ptr;
}

template <typename T>
T& Pointer<T>::operator ==(const SharedPointer<T>& other) const
{
    return ptr == other.ptr;
}

template <typename T>
T* Pointer<T>::operator !=(const SharedPointer<T>& other) const
{
    return ptr != other.ptr;
}


template <typename T>
bool Pointer<T>::operator bool() const
{
    return ptr != nullptr;
}

template <typename T>
Pointer<T>::SharedPointer(Pointer<T>&& other) : Pointer{nullptr}
{
    swap(other);
}

template <typename T>
Pointer<T>& Pointer<T>::operator=(Pointer<T>&& other)
{
    if (this != &other)
    {
        destroy();
    }

    swap(other);

    return *this;
}