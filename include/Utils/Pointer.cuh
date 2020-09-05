#pragma once

template <typename T>
class Pointer
{
    int* refcnt;
    T* ptr;
protected:
    Pointer(T* ptr = nullptr);
    void destroy() noexcept;
    void incRef() noexcept;
    void decRef() noexcept;
    
public:
    virtual ~Pointer() noexcept;

    T* get() const;
    T& operator*() const;
    T* operator->() const;
};