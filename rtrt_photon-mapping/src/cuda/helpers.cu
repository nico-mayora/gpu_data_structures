#define EPS 1e-3f
#define INFTY 1e10f

inline __device__
float norm_squared(owl::vec3f v) {
    return v.x * v.x + v.y * v.y + v.z * v.z;
}

inline __device__
owl::vec3f random_in_unit_sphere(Random rand) {
    owl::vec3f v;
    do {
        v = {rand()*2-1, rand()*2-1, rand()*2-1};
    } while (norm_squared(v) > 1.f);

    return v;
}