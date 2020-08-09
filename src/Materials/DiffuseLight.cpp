#include <DiffuseLight.hpp>
#include <Texture.hpp>
#include <WeightedPdf.hpp>

DiffuseLight::DiffuseLight(const std::shared_ptr<Texture> emitter) : Material::Material{emitter} {}

bool DiffuseLight::scatterRay(const Ray &ray, Hittable::HitRecord &record,
    WeightedPdf& pdf) const
{
    record.emitted = emitCol(ray, record, record.point);
    return false;
}

Vec3 DiffuseLight::emitCol(const Ray& ray, Hittable::HitRecord& record, const Vec3 &point) const 
{
    Vec3 col{};
    if (record.isInFront)
        col = albedo->getValue(record.u, record.v, point); 

    return col;
}