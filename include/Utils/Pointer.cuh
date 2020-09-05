#pragma once

#include <Util.h>

template <typename T>
class Pointer
{
    int* refcnt;
    T* ptr;
protected:
    Pointer(T* ptr) noexcept : refcnt{ptr ? 
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

    void destroy() noexcept
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

    void swap(Pointer<T>& other)
    {
        utils::swap(ptr, other.ptr);
        utils::swap(refcnt, other.refcnt);
    }
    
public:
    virtual ~Pointer() noexcept
    {
        decRef();
        if (*refcnt == 0)
        {
            destroy();
        }
    }

    T* get() const noexcept
    {
        return ptr;
    }

    T& operator*() const noexcept
    {
        return *ptr;
    }

    T* operator->() const noexcept
    {
        return ptr;
    }

    bool operator==(const SharedPointer<T>& other) const noexcept
    {
        return ptr == other.ptr;
    }

    bool operator!=(const SharedPointer<T>& other) const noexcept
    {
        return ptr != other.ptr;
    }

    explicit bool operator bool() const noexcept
    {
        return ptr != nullptr;
    }

    Pointer(Pointer<T>&& other) noexcept : Pointer{nullptr}
    {
        swap(other);
    }

    Pointer<T>& operator=(Pointer<T>&& other) noexcept
    {
        if (this != &other)
        {
            destroy();
        }

        swap(other);

        return *this;
    }
};