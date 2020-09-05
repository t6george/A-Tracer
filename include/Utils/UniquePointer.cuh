#pragma once

#include <Pointer.cuh>

template <typename T>
class UniquePointer : public Pointer<T>
{
public:
    static UniquePointer<T> makeUnique(T&& obj)
    {
#ifdef __CUDACC__
        T* pointer = nullptr;
        cudaMallocManaged(static_cast<void**>(&pointer), sizeof(T));
        memcpy(static_cast<void*>(pointer), static_cast<void*>(&obj), sizeof(T));
#else
        T* pointer = new T{obj};
#endif
        return UniquePointer<T>(pointer);
    }

    explicit UniquePointer(T* ptr = nullptr) : Pointer<T>::Pointer{ptr} {}
    ~UniquePointer() noexcept = default;

    UniquePointer(const UniquePointer<T>& other) = delete;
    UniquePointer<T>& operator=(const UniquePointer<T>& other) = delete;
};
