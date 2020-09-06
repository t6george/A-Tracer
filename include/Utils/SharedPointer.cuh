#pragma once

#include <Pointer.cuh>

template <typename T>
class SharedPointer : public Pointer<T>
{
public:
    HOST explicit SharedPointer(T* ptr = nullptr) : Pointer<T>{ptr} {}
    HOST ~SharedPointer() noexcept = default;

    HOST SharedPointer(const SharedPointer<T>& other) noexcept
    {
        Pointer<T>::ptr = other.ptr;
        Pointer<T>::refcnt = other.refcnt;
        Pointer<T>::incRef();
    }

    DEV HOST SharedPointer<T>& operator=(const SharedPointer<T>& other) noexcept
    {
        Pointer<T>::ptr = other.ptr; 
        Pointer<T>::refcnt = other.refcnt; 
        Pointer<T>::incRef();
        return *this;
    }
};
