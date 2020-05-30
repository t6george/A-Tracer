#include <Sphere.hpp>
#include <Ray.hpp>
#include <Material.hpp>
#include <Utils.hpp>

void Sphere::getSphereUV(const Vec3 &p, double &u, double &v)
{
    double phi = atan2(p.z(), p.x());
    double theta = asin(p.y());
    u = 1. - (phi + utils::pi) / (2. * utils::pi);
    v = (theta + utils::pi / 2.) / utils::pi;
}

Sphere::Sphere(const Vec3 &center0, const double R, const std::shared_ptr<Material> material,
               const double t0, const double t1)
    : Shape::Shape{material, t0, t1, AABB{center0 - Vec3{R, R, R}, center0 + Vec3{R, R, R}}},
      center0{center0}, center1{center0}, center{center0}, R{R} {}

Sphere::Sphere(const Vec3 &center0, const Vec3 &center1, const double R,
               const std::shared_ptr<Material> material, const double t0, const double t1)
    : Shape::Shape{material, t0, t1,
                   AABB::combineAABBs(
                       AABB{center0 - Vec3{R, R, R},
                            center0 + Vec3{R, R, R}},
                       AABB{center1 - Vec3{R, R, R},
                            center1 + Vec3{R, R, R}})},
      center0{center0}, center1{center1},
      center{center0}, R{R} {}

Hittable::HitType Sphere::getCollisionData(const Ray &ray, HitRecord &record, double tMin, double tMax)
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
            Sphere::getSphereUV(record.normal, record.u, record.v);
            record.setLightPosition(ray);
            return material->scatterRay(ray, record) ? Hittable::HitType::HIT_SCATTER : Hittable::HitType::HIT_NO_SCATTER;
        }
        t = (-half_b + disc_root) / a;
        if (t > tMin && t < tMax)
        {
            record.t = t;
            record.point = ray.eval(t);
            record.normal = (record.point - center) / R;
            Sphere::getSphereUV(record.normal, record.u, record.v);
            record.setLightPosition(ray);
            return material->scatterRay(ray, record) ? Hittable::HitType::HIT_SCATTER : Hittable::HitType::HIT_NO_SCATTER;
        }
    }
    return Hittable::HitType::NO_HIT;
}

const Vec3 &Sphere::getCenter() const
{
    return center;
}

void Sphere::translate(const double time)
{
    center = center0 + (center1 - center0) * (time - time0) / (time1 - time0);
}