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

// --- Ray-Plane Intersection ---
// Calculates intersection with the horizontal ground plane and applies a checkerboard pattern.
HitInfo intersect_plane(Ray ray) {
    HitInfo info;
    info.hit = false;
    
    // Define the ground plane at y = -1.0
    const float ground_level = -1.0; 
    const vec3 plane_normal = vec3(0.0, 1.0, 0.0);

    // Calculate the distance 't' to the intersection point.
    // The ray is parallel to the plane if the denominator is near zero.
    float denominator = dot(ray.direction, plane_normal);
    if (abs(denominator) > 0.0001) {
        float t = (ground_level - ray.origin.y) / denominator;
        
        // Ensure the intersection is in front of the ray.
        if (t > 0.001) {
            info.hit = true;
            info.t = t;
            info.position = ray.origin + t * ray.direction;
            info.normal = plane_normal;
            
            // Set the material properties for the ground
            info.transparency = 0.0;
            info.refractive_index = 1.0;
            info.reflectivity = 0.05; // A slightly reflective ground looks nice

            // --- Apply the Checkerboard Pattern ---
            vec3 color_white = vec3(0.9, 0.9, 0.9);
            vec3 color_black = vec3(0.1, 0.1, 0.1);
            
            // Use floor and mod on the x and z coordinates to create tiles.
            // If floor(x) + floor(z) is even, the color is white, otherwise it's black.
            if (mod(floor(info.position.x) + floor(info.position.z), 2.0) == 0.0) {
                info.color = color_white;
            } else {
                info.color = color_black;
            }
        }
    }
    
    return info;
}


// --- Main Trace Function (UPDATED) ---
// This now checks for intersections with spheres AND the ground plane.
HitInfo trace(Ray ray) {
    HitInfo closest_hit;
    closest_hit.hit = false;
    closest_hit.t = 1e30; // A very large number (infinity)

    // Check intersections with all spheres in the scene
    for (int i = 0; i < NUM_SPHERES; ++i) {
        HitInfo current_hit = intersect_sphere(ray, scene[i]);
        if (current_hit.hit && current_hit.t < closest_hit.t) {
            closest_hit = current_hit;
        }
    }

    // --- NEW: Check for intersection with the ground plane ---
    HitInfo plane_hit = intersect_plane(ray);
    if (plane_hit.hit && plane_hit.t < closest_hit.t) {
        // If the plane was hit and is closer than any sphere, it's our new closest hit.
        closest_hit = plane_hit;
    }

    return closest_hit;
}


// --- Light Attenuation Calculation (CORRECTED SEQUENTIAL FILTERING) ---
// This version correctly blends the light filter based on transparency.
vec3 calculate_light_attenuation(vec3 point, vec3 light_pos) {
    // Start with a light filter of pure white (100% light passthrough).
    vec3 light_filter = vec3(1.0); 

    float light_dist = length(light_pos - point);
    
    Ray shadow_ray;
    shadow_ray.origin = point + normalize(light_pos - point) * 0.001; 
    shadow_ray.direction = normalize(light_pos - point);

    float distance_traveled = 0.0;

    for(int i = 0; i < 5; i++) { 
        HitInfo hit = trace(shadow_ray);

        if (hit.hit && (hit.t + distance_traveled < light_dist)) {
            
            // --- THE FIX IS HERE ---
            // We calculate the filter for THIS object. It's a mix between pure black
            // (no light) and the object's color, based on its transparency.
            vec3 object_filter = mix(vec3(0.0), hit.color, hit.transparency);

            // We then multiply our running light_filter by this object's filter.
            light_filter *= object_filter;
            
            // If the light has been completely blocked, we can stop early.
            if (dot(light_filter, light_filter) == 0.0) {
                return vec3(0.0);
            }
            
            // Continue tracing from the new position.
            distance_traveled += hit.t;
            shadow_ray.origin = hit.position + shadow_ray.direction * 0.001;

        } else {
            // No more objects in the path to the light.
            break;
        }
    }

    return light_filter;
}


// --- Phong Lighting Calculation (UPDATED) ---
// Now uses the vec3 attenuation color for sophisticated, filtered shadows.
vec3 phong_lighting(HitInfo info, vec3 light_pos, vec3 camera_pos) {
    vec3 ambient = 0.05 * info.color;

    // Calculate the light's attenuation color using our new, advanced function.
    vec3 attenuation = calculate_light_attenuation(info.position, light_pos);

    // If attenuation is black, we're in a full opaque shadow.
    // We only contribute the ambient light term.
    if (dot(attenuation, attenuation) == 0.0) {
        return ambient;
    }

    // Diffuse
    vec3 light_dir = normalize(light_pos - info.position);
    float diff = max(dot(info.normal, light_dir), 0.0);
    vec3 diffuse = diff * info.color;

    // Specular
    vec3 view_dir = normalize(camera_pos - info.position);
    vec3 reflect_dir = reflect(-light_dir, info.normal);
    float spec = pow(max(dot(view_dir, reflect_dir), 0.0), 32.0);
    vec3 specular = 0.5 * spec * vec3(1.0, 1.0, 1.0); // White highlights

    // Final color is the ambient term plus the diffuse and specular terms,
    // which are filtered by the shadow's attenuation color.
    return ambient + (diffuse + specular) * attenuation;
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

    const float RAY_EPSILON = 0.001;

    stack[stack_ptr].ray = initial_ray;
    stack[stack_ptr].throughput = vec3(1.0, 1.0, 1.0);
    stack[stack_ptr].depth = 0;
    stack[stack_ptr].current_refractive_index = 1.0;
    stack_ptr++;

    while (stack_ptr > 0) {
        stack_ptr--;
        RayState current_state = stack[stack_ptr];

        HitInfo hit = trace(current_state.ray);

        // --- THIS IS THE MODIFIED BLOCK ---
        if (!hit.hit) {
            // If the ray misses all objects, calculate a procedural sky color
            // instead of using a flat background color.
            
            // Define the colors for the sky gradient.
            vec3 sky_color_zenith = vec3(0.5, 0.7, 1.0); // A nice light blue
            vec3 sky_color_horizon = vec3(0.8, 0.9, 1.0); // A brighter, whiter horizon

            // Use the ray's y-direction to blend between the two colors.
            // A 't' of 0.0 is horizontal, 1.0 is straight up.
            float t = 0.5 + 0.5 * current_state.ray.direction.y;
            vec3 sky_color = mix(sky_color_horizon, sky_color_zenith, t);

            final_color += sky_color * current_state.throughput;
            continue;
        }

        // --- THE REST OF THE FUNCTION IS UNCHANGED ---
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

        if (hit.transparency > 0.0) {
            vec3 refraction_dir = refract(current_state.ray.direction, outward_normal, n1 / n2);
            if (dot(refraction_dir, refraction_dir) > 0.0 && stack_ptr < STACK_SIZE) {
                RayState refracted_state;
                refracted_state.ray.origin = hit.position + refraction_dir * RAY_EPSILON;
                refracted_state.ray.direction = refraction_dir;
                refracted_state.throughput = current_state.throughput * (1.0 - fresnel_reflectance) * hit.transparency * hit.color;
                refracted_state.depth = current_state.depth + 1;
                refracted_state.current_refractive_index = n2;
                stack[stack_ptr] = refracted_state;
                stack_ptr++;
            }
        }

        float total_reflectivity = hit.reflectivity + (1.0 - hit.reflectivity) * fresnel_reflectance;
        if (total_reflectivity > 0.0) {
            if (stack_ptr < STACK_SIZE) {
                RayState reflected_state;
                reflected_state.ray.origin = hit.position + outward_normal * RAY_EPSILON;
                reflected_state.ray.direction = reflect(current_state.ray.direction, outward_normal);
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
    // This part is unchanged.
    scene[0] = Sphere(vec3(0.0, 0.0, -0.6), 1.0, vec3(1.0, 1.0, 1.0), 0.1, 0.9, 1.5);
    scene[1] = Sphere(vec3(-0.5, -0.5, -3.0), 0.5, vec3(0.2, 1.0, 0.2), 0.05, 0.0, 1.5);
    scene[2] = Sphere(vec3(0.5, -0.5, -3.0), 0.5, vec3(0.2, 0.2, 1.0), 0.05, 0.0, 1.5);
    scene[3] = Sphere(vec3(0.0, 0.366, -3.0), 0.5, vec3(1.0, 0.2, 0.2), 0.05, 0.0, 1.5);

    // --- Primary Ray Generation ---
    vec2 uv = (gl_FragCoord.xy * 2.0 - u_resolution.xy) / u_resolution.y;

    const float focal_length = 1.5;
    const int max_depth = 10;

    Ray primary_ray;
    
    // --- Manual Camera Setup ---
    // This approach gives you direct control over the camera's position and orientation.
    
    // 1. Set the camera's position in the world.
    primary_ray.origin = u_camera_pos;

    // 2. Define the direction the camera is looking. This is the 'forward' vector.
    //    It must be a normalized direction vector. To look at the origin from the camera's
    //    position, you would use: normalize(vec3(0.0) - primary_ray.origin)
    // --- To move without rotating, we can define a fixed forward direction ---
    // This camera will always look straight ahead along the world's negative Z-axis.
    // Now, changing the z-component of primary_ray.origin will move the camera forward/backward.
    vec3 camera_fwd = vec3(-1.0, -0.3, 0.0);

    // 3. Define a temporary 'up' direction. This helps orient the camera.
    //    It controls the camera's roll. For no roll, (0, 1, 0) is standard.
    vec3 temp_up = vec3(0.0, 1.0, 0.0);

    // 4. Calculate the camera's 'right' and true 'up' vectors to form an orthonormal basis.
    //    This ensures the vectors are perpendicular, creating a stable view.
    vec3 camera_right = normalize(cross(camera_fwd, temp_up));
    vec3 camera_up = normalize(cross(camera_right, camera_fwd)); // Recalculate 'up' to ensure it's orthogonal

    primary_ray.direction = normalize(
        uv.x * camera_right + 
        uv.y * camera_up + 
        focal_length * camera_fwd
    );
    
    // --- Trace and Color ---
    vec3 final_color = RayTraceIterative(primary_ray, max_depth);

    FragColor = vec4(final_color, 1.0);
}