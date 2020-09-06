#pragma once

#include <Pointer.cuh>

template <typename T>
class UniquePointer : public Pointer<T>
{
public:
    HOST explicit UniquePointer(T* ptr = nullptr) : Pointer<T>::Pointer{ptr} {}
    HOST ~UniquePointer() noexcept = default;

    UniquePointer(const UniquePointer<T>& other) = delete;
    UniquePointer<T>& operator=(const UniquePointer<T>& other) = delete;
};
