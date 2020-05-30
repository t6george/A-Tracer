#include <Material.hpp>
#include <Vec3.hpp>

Material::Material(const std::shared_ptr<Texture> albedo) : albedo{albedo} {}

Vec3 Material::emitCol(double u, double v, const Vec3 &point) const { return Vec3{}; }