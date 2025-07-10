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
};

// A sphere is a center, radius, color and reflectivity
struct Sphere {
    vec3 center;
    float radius;
    vec3 color;
    float reflectivity;
};

// --- Scene Definition ---
const int NUM_SPHERES = 3;
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
    vec3 ambient = 0.15 * info.color;

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

// --- Main Ray Tracing Function ---
vec3 RayTrace(Ray ray, int depth) {

    // Nonrecursive Computations
    // Check if the ray intersects any of the objects in the scene
    HitInfo hit = trace(ray);

    if (!hit.hit) {
        // If there is no object intersection, we return the background color
        return BACKGROUND_COLOR;
    }
    // If we hit an object, calculate its color using the Phong model
    vec3 local_color = phong_lighting(hit, u_light_pos, u_camera_pos);

    // Recursive Computations
    if (depth <= 0 || hit.reflectivity <= 0.0) {
        return local_color; // Reached maximum depth or not reflective
    }

    // Calculate reflected ray
    Ray reflected_ray;
    reflected_ray.origin = hit.position;
    reflected_ray.direction = reflect(ray.direction, hit.normal);

    // Recursively trace the reflected ray
    vec3 reflected_color = RayTrace(reflected_ray, depth - 1);

    // Combine local and reflected color
    return mix(local_color, reflected_color, hit.reflectivity);
}

/**
 * An iterative version of the RayTrace function.
 *
 * @param initial_ray The first ray cast from the camera.
 * @param max_depth The maximum number of times a ray is allowed to bounce.
 * @return The final calculated color for the ray path.
 */
vec3 RayTraceIterative(Ray initial_ray, int max_depth) {

    vec3 final_color = vec3(0.0, 0.0, 0.0); // The accumulated color, starts at black.
    float throughput = 1.0;                  // The accumulated reflectivity, starts at 1.0.
    Ray current_ray = initial_ray;           // The ray for the current iteration.

    for (int depth = 0; depth < max_depth; ++depth) {
        // Trace the current ray to find the closest intersection.
        HitInfo hit = trace(current_ray);

        // --- Termination Case 1: Ray misses all objects. ---
        // Add the background color, attenuated by the path's throughput, and stop.
        if (!hit.hit) {
            final_color += BACKGROUND_COLOR * throughput;
            break; // Exit the loop.
        }

        // We have an intersection, so calculate its direct illumination (Phong).
        vec3 local_color = phong_lighting(hit, u_light_pos, u_camera_pos);

        // --- Termination Case 2: Final bounce. ---
        // This happens if the surface is not reflective or we've reached the depth limit.
        // Add the full local color (attenuated by throughput) and stop.
        if (hit.reflectivity <= 0.0 || depth == max_depth - 1) {
            final_color += local_color * throughput;
            break; // Exit the loop.
        }

        // --- Continuation Case: The ray bounces. ---
        // Add the local color's contribution, weighted by (1 - reflectivity).
        final_color += local_color * (1.0 - hit.reflectivity) * throughput;

        // Update throughput for the next bounce.
        throughput *= hit.reflectivity;

        // Update the ray to be the new reflected ray for the next iteration.
        current_ray.origin = hit.position;
        current_ray.direction = reflect(current_ray.direction, hit.normal);
    }

    return final_color;
}


void main() {
    // --- Setup the Scene ---
    // We define the scene objects here
    scene[0] = Sphere(vec3(0.0, 0.0, -2.0), 0.5, vec3(1.0, 0.2, 0.2), 0.5);  // Red sphere
    scene[1] = Sphere(vec3(-1.2, 0.0, -2.0), 0.5, vec3(0.2, 1.0, 0.2), 0.5); // Green sphere
    scene[2] = Sphere(vec3(1.2, 0.0, -2.0), 0.5, vec3(0.2, 0.2, 1.0), 0.5);  // Blue sphere

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
