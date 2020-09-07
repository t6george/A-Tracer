#pragma once

#include <Macro.cuh>
#include <Util.cuh>

template <typename T>
class Pointer
{
protected:
    int* refcnt;
    T* ptr;

    DEV HOST explicit Pointer(T* ptr = nullptr) noexcept : refcnt{ptr ? 
#if GPU == 1
    nullptr
#else
    new int{1}
#endif
    : nullptr},
    ptr{ptr} 
    {
#if GPU == 1
        if (ptr)
        {
            T* pointer = nullptr;

            cudaMallocManaged((void**)&pointer, sizeof(T));
            memcpy((void*)pointer, static_cast<void*>(ptr), sizeof(T));
            delete ptr;
            ptr = pointer;

	    cudaMallocManaged((void**)&refcnt, sizeof(int));
        
            cudaDeviceSynchronize();
            *refcnt = 1;
        }
#endif
    }

    DEV HOST void destroy() noexcept
    {
#if GPU == 1
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

    DEV HOST void incRef() noexcept
    {
        refcnt && ++(*refcnt);
    }

    DEV HOST void decRef() noexcept
    {
        refcnt && --(*refcnt);
    }

    HOST void swap(Pointer<T>& other)
    {
        utils::swap(ptr, other.ptr);
        utils::swap(refcnt, other.refcnt);
    }
    
public:
    DEV HOST virtual ~Pointer() noexcept
    {
        decRef();
        if (refcnt && *refcnt == 0)
        {
            destroy();
        }
    }

    DEV HOST T* get() const noexcept
    {
        return ptr;
    }

    DEV HOST T& operator*() const noexcept
    {
        return *ptr;
    }

    DEV HOST T* operator->() const noexcept
    {
        return ptr;
    }

    DEV HOST bool operator==(const Pointer<T>& other) const noexcept
    {
        return ptr == other.ptr;
    }

    DEV HOST bool operator!=(const Pointer<T>& other) const noexcept
    {
        return ptr != other.ptr;
    }

    DEV HOST explicit operator bool() const noexcept
    {
        return ptr != nullptr;
    }

    HOST Pointer(Pointer<T>&& other) noexcept : Pointer{nullptr}
    {
        swap(other);
    }

    HOST Pointer<T>& operator=(Pointer<T>&& other) noexcept
    {
        if (this != &other)
        {
            destroy();
        }

        swap(other);

        return *this;
    }
};
