#include <Material.hpp>
#include <Vec3.hpp>
#include <WeightedPdf.hpp>

Material::Material(const std::shared_ptr<Texture> albedo) : albedo{albedo} {}

bool Material::scatterRay(const Ray &ray, Hittable::HitRecord &record) const { return false; }

Vec3 Material::emitCol(const Ray &ray, Hittable::HitRecord &record, 
    const Vec3 &point) const { return Vec3 {}; }