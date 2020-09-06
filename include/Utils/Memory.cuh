#pragma once

#include <Macro.cuh>
#include <UniquePointer.cuh>
#include <SharedPointer.cuh>

/*#include <Objects.cuh>
#include <Transformations.cuh>
#include <Light.cuh>
#include <Materials.cuh>
#include <Pdfs.cuh>
#include <SceneGeneration.cuh>
#include <Textures.cuh>
#include <Vec3.cuh>*/

namespace mem
{
    template<typename T, typename ... Args>
    HOST SharedPointer<T> MakeShared(Args&& ... args)
    {
        T obj(Args&& ... args);
    #ifdef __CUDACC__
        T* pointer = nullptr;
        cudaMallocManaged((void**)&pointer, sizeof(T));
    #else
        T* pointer = new T;
    #endif
        memcpy((void*)pointer, static_cast<void*>(&obj), sizeof(T));
        return SharedPointer<T>(pointer);
    }

    template<typename T, typename ... Args>
    HOST UniquePointer<T> MakeUnique(Args&& ... args)
    {
        T obj(Args&& ... args);
    #ifdef __CUDACC__
        T* pointer = nullptr;
        cudaMallocManaged((void**)&pointer, sizeof(T));
    #else
        T* pointer = new T;
    #endif
        memcpy((void*)pointer, static_cast<void*>(&obj), sizeof(T));
        return UniquePointer<T>(pointer);
    }
 
    template<typename T, typename U>
    DEV HOST inline SharedPointer<T> static_pointer_cast(const SharedPointer<U>& sp) noexcept
    {
        return SharedPointer<T>(static_cast<typename SharedPointer<T>::PtrType*>(sp.get()), sp.getRef());
    }

    template<typename T, typename U>
    DEV HOST inline SharedPointer<T> const_pointer_cast(const SharedPointer<U>& sp) noexcept
    {
        return SharedPointer<T>(const_cast<typename SharedPointer<T>::PtrType*>(sp.get()), sp.getRef());
    }

    template<typename T, typename U>
    DEV HOST inline SharedPointer<T> dynamic_pointer_cast(const SharedPointer<U>& sp) noexcept
    {
        if (auto* _p = dynamic_cast<typename SharedPointer<T>::PtrType*>(sp.get()))
            return SharedPointer<T>(_p, sp.getRef());
        
	return SharedPointer<T>();
    }

    template<typename T, typename U>
    DEV HOST inline SharedPointer<T> reinterpret_pointer_cast(const SharedPointer<U>& sp) noexcept
    {
        return SharedPointer<T>(reinterpret_cast<typename SharedPointer<T>::PtrType*>(sp.get()), sp.getRef());
    }
} // namespace mem
