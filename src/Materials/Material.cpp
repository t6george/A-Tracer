#include <Material.cuh>
#include <Vec3.cuh>
#include <WeightedPdf.cuh>

Material::Material(const std::shared_ptr<Texture> albedo) : albedo{albedo} {}

bool Material::scatterRay(const Ray &ray, Hittable::HitRecord &record) const { return false; }

Vec3 Material::emitCol(const Ray &ray, Hittable::HitRecord &record, 
    const Vec3 &point) const { return Vec3 {}; }