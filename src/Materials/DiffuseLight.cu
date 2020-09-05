#include <DiffuseLight.cuh>
#include <Texture.cuh>
#include <WeightedPdf.cuh>

DEV HOST DiffuseLight::DiffuseLight(const SharedPointer<Texture> emitter) : Material::Material{emitter} {}

DEV bool DiffuseLight::scatterRay(const Ray &ray, Hittable::HitRecord &record) const
{
    record.emitted = emitCol(ray, record, record.point);
    return false;
}

DEV Vec3 DiffuseLight::emitCol(const Ray& ray, Hittable::HitRecord& record, const Vec3 &point) const 
{
    Vec3 col{};
    if (record.isInFront)
        col = albedo->getValue(record.u, record.v, point); 

    return col;
}