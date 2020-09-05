#include <Pointer.cuh>

template <typename T>
Pointer<T>::Pointer(T* ptr) : refcnt{ptr ? new int{1} : nullptr}, ptr{ptr} {}

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
T* Pointer<T>::get() const
{
    return ptr;
}

template <typename T>
T* Pointer<T>::operator->() const
{
    return ptr;
}

template <typename T>
T& Pointer<T>::operator*() const
{
    return *ptr;
}