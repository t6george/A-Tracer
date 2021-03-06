#include <Sphere.cuh>
#include <Ray.cuh>
#include <Material.cuh>
#include <Util.cuh>
#include <OrthonormalBasis.cuh>
#include <WeightedPdf.cuh>

DEV void Sphere::getSphereUV(const Vec3 &p, double &u, double &v)
{
    double phi = atan2(p.z(), p.x());
    double theta = asin(p.y());
    u = 1. - (phi + utils::pi) / (2. * utils::pi);
    v = (theta + utils::pi / 2.) / utils::pi;
}

DEV HOST Sphere::Sphere(const Vec3 &center0, const double R, const SharedPointer<Material> material,
               const double t0, const double t1)
    : Shape::Shape{material, AABB{center0 - Vec3{R, R, R}, center0 + Vec3{R, R, R}}},
      center0{center0}, center1{center0}, R{R}, time0{t0}, time1{t1} {}

DEV HOST Sphere::Sphere(const Vec3 &center0, const Vec3 &center1, const double R,
               const SharedPointer<Material> material, const double t0, const double t1)
    : Shape::Shape{material,
                   AABB::combineAABBs(
                       AABB{center0 - Vec3{R, R, R},
                            center0 + Vec3{R, R, R}},
                       AABB{center1 - Vec3{R, R, R},
                            center1 + Vec3{R, R, R}})},
      center0{center0}, center1{center1},
      R{R}, time0{t0}, time1{t1} {}

DEV Hittable::HitType Sphere::getCollisionData(const Ray &ray, HitRecord &record,
                             double tMin, double tMax, bool flip) const
{
    Vec3 center = blur(ray.getTime());
    Vec3 los = ray.getOrigin() - center;

    double a = ray.getDirection().o(ray.getDirection());
    double half_b = ray.getDirection().o(los);
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

            if (flip)
            {
                record.isInFront ^= true;
            }

            return material->scatterRay(ray, record) ? Hittable::HitType::HIT_SCATTER
                                                     : Hittable::HitType::HIT_NO_SCATTER;
        }
        t = (-half_b + disc_root) / a;
        if (t > tMin && t < tMax)
        {
            record.t = t;
            record.point = ray.eval(t);
            record.normal = (record.point - center) / R;
            Sphere::getSphereUV(record.normal, record.u, record.v);
            record.setLightPosition(ray);

            if (flip)
            {
                record.isInFront ^= true;
            }

            return material->scatterRay(ray, record) ? Hittable::HitType::HIT_SCATTER
                                                     : Hittable::HitType::HIT_NO_SCATTER;
        }
    }
    return Hittable::HitType::NO_HIT;
}

DEV Vec3 Sphere::blur(const double time) const
{
    return center0 + (center1 - center0) * (time - time0) / (time1 - time0);
}

DEV Vec3 Sphere::genRandomVector(const Vec3& origin) const
{
    Vec3 dist = center0 - origin;
    double disSq = dist.sqLen();
    OrthonormalBasis basis{dist};
    return basis.getVec(Vec3::randomVecToSphere(R, disSq));
}

DEV double Sphere::eval(const Vec3& origin, const Vec3& v, bool flip) const
{
    Hittable::HitRecord record;
    double prob = 0.;
    WeightedPdf pdf = WeightedPdf{nullptr, nullptr, 0.};

    if(static_cast<bool>(getCollisionData(Ray(origin, v), record, .001, utils::infinity, flip)))
    {
        double solidAngle = 2. * utils::pi * (1. - sqrt(1. - R * R / (center0 - origin).sqLen()));
        prob = 1. / solidAngle;
    }

    return prob;
}
