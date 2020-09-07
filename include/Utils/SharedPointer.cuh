#pragma once

#include <Pointer.cuh>

template <typename T>
class SharedPointer : public Pointer<T>
{
public:
    using PtrType = T;

    DEV HOST explicit SharedPointer(T* ptr = nullptr) : Pointer<T>{ptr} {}
    DEV HOST ~SharedPointer() noexcept = default;

    HOST int* getRef() const noexcept
    {
        return Pointer<T>::refcnt;
    }
	    
    HOST SharedPointer(T* ptr, int* ref) noexcept
    {
        Pointer<T>::ptr = ptr;
        Pointer<T>::refcnt = ref;
        Pointer<T>::incRef();
    }

    HOST SharedPointer(const SharedPointer<T>& other) noexcept
    {
        Pointer<T>::ptr = other.ptr;
        Pointer<T>::refcnt = other.refcnt;
        Pointer<T>::incRef();
    }

    DEV HOST SharedPointer<T>& operator=(const SharedPointer<T>& other) noexcept
    {
	Pointer<T>::decRef();
        if (Pointer<T>::refcnt && *Pointer<T>::refcnt == 0)
        {
            Pointer<T>::destroy();
        }

        Pointer<T>::ptr = other.ptr; 
        Pointer<T>::refcnt = other.refcnt; 
        Pointer<T>::incRef();
        return *this;
    }
};

