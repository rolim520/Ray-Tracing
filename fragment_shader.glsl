#version 330 core
out vec4 FragColor;

// Uniforms passed from Python
uniform vec2 u_resolution;
uniform vec3 u_camera_pos;
uniform vec3 u_light_pos;

// --- Data Structures ---
// A ray is an origin and a direction
struct Ray {
    vec3 origin;
    vec3 direction;
};

// Information about an intersection
struct HitInfo {
    bool hit;
    float t;          // Distance along the ray to the hit point
    vec3 position;
    vec3 normal;
    vec3 color;
    float reflectivity;
    float transparency;
    float refractive_index;
};

// A sphere is a center, radius, color and reflectivity
struct Sphere {
    vec3 center;
    float radius;
    vec3 color;
    float reflectivity;
    float transparency;
    float refractive_index;
};

// --- New Data Structure for the Iterative Stack ---
struct RayState {
    Ray ray;
    vec3 throughput; // How much color/energy is carried by this ray
    int depth;
    float current_refractive_index; // Refractive index of the medium the ray is currently in
};

// --- Scene Definition ---
const int NUM_SPHERES = 4;
Sphere scene[NUM_SPHERES];
const vec3 BACKGROUND_COLOR = vec3(0.1, 0.1, 0.1); // Corresponds to "Return the background color"

// Forward declaration for the recursive function
vec3 RayTrace(Ray ray, int depth);

// --- Ray-Sphere Intersection ---
// Solves the quadratic equation for ray-sphere intersection.
// Returns a HitInfo struct.
HitInfo intersect_sphere(Ray ray, Sphere s) {
    HitInfo info;
    info.hit = false;
    vec3 oc = ray.origin - s.center;
    float a = dot(ray.direction, ray.direction);
    float b = 2.0 * dot(oc, ray.direction);
    float c = dot(oc, oc) - s.radius * s.radius;
    float discriminant = b*b - 4.0*a*c;

    if (discriminant < 0.0) {
        return info; // No hit
    }

    float t = (-b - sqrt(discriminant)) / (2.0 * a);
    if (t < 0.0) { // If the intersection is behind the ray, try the other root
        t = (-b + sqrt(discriminant)) / (2.0 * a);
    }
    
    if (t > 0.001) { // Epsilon to avoid self-intersection
        info.hit = true;
        info.t = t;
        info.position = ray.origin + t * ray.direction;
        info.normal = normalize(info.position - s.center);
        info.color = s.color;
        info.reflectivity = s.reflectivity;
        info.transparency = s.transparency;
        info.refractive_index = s.refractive_index;
    }
    return info;
}


// --- Main Trace Function ---
// This corresponds to "Check the ray... let z be the first intersection point"
HitInfo trace(Ray ray) {
    HitInfo closest_hit;
    closest_hit.hit = false;
    closest_hit.t = 1e30; // A very large number (infinity)

    for (int i = 0; i < NUM_SPHERES; ++i) {
        HitInfo current_hit = intersect_sphere(ray, scene[i]);
        if (current_hit.hit && current_hit.t < closest_hit.t) {
            closest_hit = current_hit;
        }
    }
    return closest_hit;
}


// --- Shadow Calculation ---
// Corresponds to the "shadow feeler" part of the algorithm
// Checks if a point is in shadow by casting a ray to the light
float calculate_shadow(vec3 point, vec3 light_pos) {
    Ray shadow_ray;
    shadow_ray.origin = point;
    shadow_ray.direction = normalize(light_pos - point);
    
    HitInfo shadow_hit = trace(shadow_ray);

    // If the shadow ray hits something before the light, the point is in shadow
    float light_dist = length(light_pos - point);
    if (shadow_hit.hit && shadow_hit.t < light_dist) {
        return 0.0; // In shadow
    }
    return 1.0; // Not in shadow
}


// --- Phong Lighting Calculation ---
// Corresponds to "Set color = I_local"
vec3 phong_lighting(HitInfo info, vec3 light_pos, vec3 camera_pos) {
    vec3 ambient = 0.1 * info.color;

    // Calculate shadow factor
    float shadow_factor = calculate_shadow(info.position, light_pos);
    if (shadow_factor < 1.0) {
        return ambient; // If in shadow, only return ambient light
    }

    // Diffuse
    vec3 light_dir = normalize(light_pos - info.position);
    float diff = max(dot(info.normal, light_dir), 0.0);
    vec3 diffuse = diff * info.color;

    // Specular
    vec3 view_dir = normalize(camera_pos - info.position);
    vec3 reflect_dir = reflect(-light_dir, info.normal);
    float spec = pow(max(dot(view_dir, reflect_dir), 0.0), 32.0);
    vec3 specular = 0.7 * spec * vec3(1.0, 1.0, 1.0); // White highlights

    return ambient + diffuse + specular;
}

/**
 * Calculates the reflection coefficient using Schlick's approximation.
 * This determines the ratio of reflected light to refracted light for a dielectric material.
 * @param cos_theta The cosine of the angle between the incoming light vector and the normal.
 * @param n1 The refractive index of the medium the ray is in.
 * @param n2 The refractive index of the medium the ray is entering.
 * @return The amount of light that is reflected (a value between 0.0 and 1.0).
 */
float calculate_fresnel(float cos_theta, float n1, float n2) {
    float r0 = (n1 - n2) / (n1 + n2);
    r0 = r0 * r0;
    // Handle total internal reflection
    float n = n1 / n2;
    float sin_t2 = n * n * (1.0 - cos_theta * cos_theta);
    if (sin_t2 > 1.0) {
        return 1.0; // Total internal reflection
    }
    return r0 + (1.0 - r0) * pow(1.0 - cos_theta, 5.0);
}

/**
 * An iterative version of the RayTrace function that uses a manual stack
 * to handle both reflection and refraction without recursion.
 *
 * @param initial_ray The first ray cast from the camera.
 * @param max_depth The maximum number of times a ray is allowed to bounce.
 * @return The final calculated color for the ray path.
 */
vec3 RayTraceIterative(Ray initial_ray, int max_depth) {
    vec3 final_color = vec3(0.0);

    const int STACK_SIZE = 16;
    RayState stack[STACK_SIZE];
    int stack_ptr = 0;

    // A small value to offset new rays to prevent self-intersection.
    const float RAY_EPSILON = 0.001;

    // --- Push the initial ray onto the stack ---
    stack[stack_ptr].ray = initial_ray;
    stack[stack_ptr].throughput = vec3(1.0, 1.0, 1.0);
    stack[stack_ptr].depth = 0;
    stack[stack_ptr].current_refractive_index = 1.0;
    stack_ptr++;

    // --- Process rays until the stack is empty ---
    while (stack_ptr > 0) {
        stack_ptr--;
        RayState current_state = stack[stack_ptr];

        HitInfo hit = trace(current_state.ray);

        if (!hit.hit) {
            final_color += BACKGROUND_COLOR * current_state.throughput;
            continue;
        }

        vec3 outward_normal;
        float n1, n2;
        if (dot(current_state.ray.direction, hit.normal) < 0.0) {
            outward_normal = hit.normal;
            n1 = current_state.current_refractive_index;
            n2 = hit.refractive_index;
        } else {
            outward_normal = -hit.normal;
            n1 = hit.refractive_index;
            n2 = 1.0;
        }

        float cos_theta = abs(dot(current_state.ray.direction, outward_normal));
        float fresnel_reflectance = calculate_fresnel(cos_theta, n1, n2);

        float local_coef = 1.0 - hit.reflectivity - hit.transparency;
        if (local_coef > 0.0) {
            vec3 local_color = phong_lighting(hit, u_light_pos, u_camera_pos);
            final_color += local_color * local_coef * current_state.throughput;
        }

        if (current_state.depth >= max_depth - 1) {
            continue;
        }

        // --- Handle Refraction (Transparency) ---
        if (hit.transparency > 0.0) {
            vec3 refraction_dir = refract(current_state.ray.direction, outward_normal, n1 / n2);
            if (dot(refraction_dir, refraction_dir) > 0.0 && stack_ptr < STACK_SIZE) {
                RayState refracted_state;
                // Bias the origin along the new direction to prevent self-intersection.
                refracted_state.ray.origin = hit.position + refraction_dir * RAY_EPSILON;
                refracted_state.ray.direction = refraction_dir;

                refracted_state.throughput = current_state.throughput * (1.0 - fresnel_reflectance) * hit.transparency * hit.color;
                refracted_state.depth = current_state.depth + 1;
                refracted_state.current_refractive_index = n2;
                stack[stack_ptr] = refracted_state;
                stack_ptr++;
            }
        }

        // --- Handle Reflection ---
        float total_reflectivity = hit.reflectivity + (1.0 - hit.reflectivity) * fresnel_reflectance;
        if (total_reflectivity > 0.0) {
            if (stack_ptr < STACK_SIZE) {
                RayState reflected_state;
                
                reflected_state.ray.origin = hit.position + outward_normal * RAY_EPSILON;
                reflected_state.ray.direction = reflect(current_state.ray.direction, outward_normal);;
                
                reflected_state.throughput = current_state.throughput * total_reflectivity;
                reflected_state.depth = current_state.depth + 1;
                reflected_state.current_refractive_index = current_state.current_refractive_index;
                stack[stack_ptr] = reflected_state;
                stack_ptr++;
            }
        }
    }
    return final_color;
}


void main() {
    // --- Setup the Scene ---
    // We define the scene objects here
    // A large glass sphere is placed in front of the camera.
    scene[0] = Sphere(vec3(0.0, 0.0, -0.6), 1.0, vec3(1.0, 1.0, 1.0), 0.15, 0.9, 1.5);  // Large glass sphere

    // Three RGB spheres are placed further back in a triangle formation.
    // Top of the triangle
    scene[1] = Sphere(vec3(0.0, 1.0, -5.0), 0.5, vec3(1.0, 0.2, 0.2), 0.2, 0.0, 1.5);  // Red sphere
    // Bottom-left of the triangle
    scene[2] = Sphere(vec3(-1.0, -0.5, -5.0), 0.5, vec3(0.2, 1.0, 0.2), 0.2, 0.0, 1.5); // Green sphere
    // Bottom-right of the triangle
    scene[3] = Sphere(vec3(1.0, -0.5, -5.0), 0.5, vec3(0.2, 0.2, 1.0), 0.2, 0.0, 1.5);  // Blue sphere

    // --- Primary Ray Generation ---
    // This is the "For each pixel p" part from RayTraceMain()
    vec2 uv = (gl_FragCoord.xy * 2.0 - u_resolution.xy) / u_resolution.y;

    // The distance from the camera to the virtual viewing plane.
    // A smaller value creates a wider field of view (like a wide-angle lens).
    const float focal_length = 1.0;

    // Maximum depth of the ray tracing
    const int max_depth = 10;

    Ray primary_ray;
    primary_ray.origin = u_camera_pos;
    primary_ray.direction = normalize(vec3(uv, -focal_length)); // Simple camera pointing down -Z

    // --- Trace and Color ---
    // This calls the main RayTrace logic
    vec3 final_color = RayTraceIterative(primary_ray, max_depth);

    FragColor = vec4(final_color, 1.0);
}
