#include <XYRect.hpp>

XYRect::XYRect(const std::shared_ptr<Material> material,
               const double x0, const double x1, const double y0,
               const double y1, const double z, const double t0, const double t1)
    : Shape::Shape{material, t0, t1, AABB{Vec3{x0, y0, z - .0001}, Vec3{x1, y1, z + .0001}}},
      x0{x0}, x1{x1}, y0{y0}, y1{y1}, z{z} {}

Hittable::HitType XYRect::getCollisionData(const Ray &ray, HitRecord &record, double tMin, double tMax)
{
    double t = (z - ray.origin().z()) / ray.direction().z();
    Hittable::HitType hit = Hittable::HitType::NO_HIT;

    if (t <= tMax && t >= tMin)
    {
        double x = ray.origin().x() + ray.direction().x() * t;
        double y = ray.origin().y() + ray.direction().y() * t;
        if (x >= x0 && x <= x1 && y >= y0 && y <= y1)
        {
            record.t = t;
            record.u = (x - x0) / (x1 - x0);
            record.v = (y - y0) / (y1 - y0);
            record.normal = Vec3{0., 0., 1.};
            record.setLightPosition(ray);
            record.point = Vec3{x, y, z};
        }
    }

    return hit;
}

void XYRect::translate(const double time) {}