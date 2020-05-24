#include <math.h>
#include <cassert>

#include <Sphere.hpp>
#include <Ray.hpp>
#include <Material.hpp>

Sphere::Sphere(const Vec3 &center0, const double R, const std::shared_ptr<Material> material,
               const double t0, const double t1)
    : Hittable::Hittable{material, t0, t1}, center0{center0}, center1{center0 + Vec3{1., 0., 0.}},
      center{center0}, R{R}, boundingBox{center0 - Vec3{R, R, R}, center0 + Vec3{R, R, R}}
{
    assert(center0 != center1);
}

Sphere::Sphere(const Vec3 &center0, const Vec3 &center1, const double R,
               const std::shared_ptr<Material> material, const double t0, const double t1)
    : Hittable::Hittable{material, t0, t1}, center0{center0}, center1{center1},
      center{center0}, R{R}, boundingBox{AABB::combineAABBs(AABB{center0 - Vec3{R, R, R}, center0 + Vec3{R, R, R}},
                                                            AABB{center1 - Vec3{R, R, R}, center1 + Vec3{R, R, R}})}
{
    assert(center0 != center1);
}

bool Sphere::getCollisionData(const Ray &ray, HitRecord &record, double tMin, double tMax)
{
    translate(ray.time());
    Vec3 los = ray.origin() - center;

    double a = ray.direction().o(ray.direction());
    double half_b = ray.direction().o(los);
    double c = los.o(los) - R * R;
    double disc = half_b * half_b - a * c;
    double disc_root, t;

    if (disc > 0.)
    {
        disc_root = sqrt(disc);
        t = (-half_b - disc_root) / a;
        if (t > tMin && t < tMax)
        {
            record.t = t;
            record.point = ray.eval(t);
            record.normal = (record.point - center) / R;
            record.setLightPosition(ray);
            return material->scatterRay(ray, record);
        }
        t = (-half_b + disc_root) / a;
        if (t > tMin && t < tMax)
        {
            record.t = t;
            record.point = ray.eval(t);
            record.normal = (record.point - center) / R;
            record.setLightPosition(ray);
            return material->scatterRay(ray, record);
        }
    }
    return false;
}

const Vec3 &Sphere::getCenter() const
{
    return center;
}

void Sphere::translate(const double time)
{
    center = center0 + (center1 - center0) * (time - time0) / (time1 - time0);
}

bool Sphere::getBoundingBox(double time0, double time1, AABB &box) const
{
    box = boundingBox;
    return true;
}