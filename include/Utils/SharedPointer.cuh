#pragma once

#include <Pointer.cuh>

template <typename T>
class SharedPointer : public Pointer<T>
{
public:
    static SharedPointer<T> makeShared(T&& obj) noexcept
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

    explicit SharedPointer(T* ptr) : Pointer{ptr} {}
    ~SharedPointer() noexcept = default;

    SharedPointer(const SharedPointer<T>& other) noexcept : ptr{other.ptr}, refcnt{other.refcnt}
    {
        incRef();
    }

    SharedPointer<T>& operator=(const SharedPointer<T>& other) noexcept
    {
        ptr = other.ptr; 
        refcnt = other.refcnt; 
        incRef();
        return *this;
    }
};