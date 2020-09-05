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
class UniquePointer : public Pointer<T>
{
public:
    HOST static UniquePointer<T> makeUnique(T&& obj)
    {
#ifdef __CUDACC__
        T* pointer = nullptr;
        cudaMallocManaged((void**)&pointer, sizeof(T));
        memcpy((void*)pointer, static_cast<void*>(&obj), sizeof(T));
#else
        T* pointer = new T{obj};
#endif
        return UniquePointer<T>(pointer);
    }

    HOST explicit UniquePointer(T* ptr = nullptr) : Pointer<T>::Pointer{ptr} {}
    HOST ~UniquePointer() noexcept = default;

    UniquePointer(const UniquePointer<T>& other) = delete;
    UniquePointer<T>& operator=(const UniquePointer<T>& other) = delete;
};
