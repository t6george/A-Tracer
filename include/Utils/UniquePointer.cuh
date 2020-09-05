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
    template<template T, template ... Args>
    HOST static UniquePointer<T> makeUnique(Args&& ... args)
    {
        T obj{std::forward<Args>(args)...);
#ifdef __CUDACC__
        T* pointer = nullptr;
        cudaMallocManaged((void**)&pointer, sizeof(T));
#else
        T* pointer = new T;
#endif
        memcpy((void*)pointer, static_cast<void*>(&obj), sizeof(T));
        return UniquePointer<T>(pointer);
    }

    HOST explicit UniquePointer(T* ptr = nullptr) : Pointer<T>::Pointer{ptr} {}
    HOST ~UniquePointer() noexcept = default;

    UniquePointer(const UniquePointer<T>& other) = delete;
    UniquePointer<T>& operator=(const UniquePointer<T>& other) = delete;
};
