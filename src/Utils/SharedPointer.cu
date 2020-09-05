#include <SharedPointer.cuh>
#include <Util.h>

template <typename T>
static SharedPointer<T> SharedPointer<T>::makeShared(T&& obj)
{
#ifdef __CUDACC__
    T* pointer = nullptr;
    cudaMallocManaged((void**) &dPointer, sizeof(T));
#else
    T* pointer = new T{obj};
#endif
    return SharedPointer<T>(pointer);
}

template <typename T>
SharedPointer<T>::SharedPointer(T* ptr) : Pointer{ptr} {}

template <typename T>
SharedPointer<T>::SharedPointer(SharedPointer<T>& other) : ptr{other.ptr}, refcnt{other.refcnt}
{ 
    incRef();
}

template <typename T>
SharedPointer<T>& SharedPointer<T>::operator=(SharedPointer<T>& other)
{
    ptr = other.ptr; 
    refcnt = other.refcnt; 
    incRef();
    return *this;
}

template <typename T>
SharedPointer<T>::SharedPointer(SharedPointer<T>&& other) : Pointer{nullptr}
{ 
    utils::swap(ptr, other.ptr);
    utils::swap(refcnt, other.refcnt);
}

template <typename T>
SharedPointer<T>& SharedPointer<T>::operator=(SharedPointer<T>&& other)
{
    if (this != &other)
    {
        destroy();
    }

    utils::swap(ptr, other.ptr);
    utils::swap(refcnt, other.refcnt);

    return *this;
}