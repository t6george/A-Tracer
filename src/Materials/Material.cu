#include <Material.cuh>
#include <Vec3.cuh>
#include <WeightedPdf.cuh>
#include <Texture.cuh>

DEV HOST Material::Material(const SharedPointer<Texture> albedo) : albedo{albedo} {}

DEV bool Material::scatterRay(const Ray &ray, Hittable::HitRecord &record) const { return false; }

DEV Vec3 Material::emitCol(const Ray &ray, Hittable::HitRecord &record, 
    const Vec3 &point) const { return Vec3 {}; }
