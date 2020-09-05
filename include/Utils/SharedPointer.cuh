#pragma once

#include <Pointer.cuh>

#include <Objects.cuh>
#include <Transformations.cuh>
#include <Light.cuh>
#include <Materials.cuh>
#include <Pdfs.cuh>
#include <SceneGeneration.cuh>
#include <Textures.cuh>
#include <Vec3.cuh>

template <typename T>
class SharedPointer : public Pointer<T>
{
public:
    HOST static SharedPointer<T> makeShared(T&& obj) noexcept
    {
#ifdef __CUDACC__
        T* pointer = nullptr;
        cudaMallocManaged((void**)&pointer, sizeof(T));
        memcpy((void*)pointer, static_cast<void*>(&obj), sizeof(T));
#else
        T* pointer = new T{obj};
#endif
        return SharedPointer<T>(pointer);
    }

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
