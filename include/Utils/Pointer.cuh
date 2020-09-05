#pragma once

template <typename T>
class Pointer
{
    int* refcnt;
    T* ptr;
protected:
    Pointer(T* ptr = nullptr) noexcept;
    void destroy() noexcept;
    void incRef() noexcept;
    void decRef() noexcept;
    void swap(Pointer<T>& other);
    
public:
    virtual ~Pointer() noexcept;

    T* get() const noexcept;

    T& operator*() const noexcept;
    T* operator->() const noexcept;
    bool operator==(const SharedPointer<T>& other) const noexcept;
    bool operator!=(const SharedPointer<T>& other) const noexcept;
    explicit bool operator bool() const noexcept;

    Pointer(Pointer<T>&& other) noexcept;
    Pointer<T>& operator=(Pointer<T>&& other) noexcept;
};