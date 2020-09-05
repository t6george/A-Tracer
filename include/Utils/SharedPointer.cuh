#pragma once

#include <Pointer.cuh>

template <typename T>
class SharedPointer : public Pointer<T>
{
public:
    static SharedPointer<T> makeShared(T&& obj);

    explicit SharedPointer(T* ptr);
    ~SharedPointer() noexcept = default;

    SharedPointer(SharedPointer<T>& other);
    SharedPointer<T>& operator=(SharedPointer<T>& other);
    
    SharedPointer(SharedPointer<T>&& other);
    SharedPointer<T>& operator=(SharedPointer<T>&& other);
};