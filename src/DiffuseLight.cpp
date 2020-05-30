#include <DiffuseLight.hpp>
#include <Texture.hpp>

DiffuseLight::DiffuseLight(const std::shared_ptr<Texture> emitter) : Material::Material{emitter} {}

bool DiffuseLight::scatterRay(const Ray &ray, Hittable::HitRecord &record) const
{
    record.emitted = emitCol(record.u, record.v, record.point);
    return false;
}

Vec3 DiffuseLight::emitCol(double u, double v, const Vec3 &point) const { return albedo->getValue(u, v, point); }