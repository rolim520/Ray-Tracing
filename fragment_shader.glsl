#version 330 core
out vec4 FragColor;

// Uniforms passed from Python
uniform vec2 u_resolution;
uniform mat4 u_camera_to_world; // REPLACES u_camera_pos
uniform vec3 u_light_pos;
uniform float u_focal_length;

// --- Data Structures ---
// Type constants for scene objects
const int SHAPE_SPHERE = 1;
const int SHAPE_TORUS = 2;

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

// A unified scene object that can be a sphere or a torus
struct SceneObject {
    int type;

    // Geometric properties
    vec3 center;
    float radius;       // For sphere
    vec3 normal;        // For torus
    float major_radius; // For torus
    float minor_radius; // For torus

    // Material properties
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
const int NUM_OBJECTS = 5;
uniform SceneObject scene[NUM_OBJECTS];

// --- Ray-Sphere Intersection ---
// Solves the quadratic equation for ray-sphere intersection.
// Returns a HitInfo struct.
HitInfo intersect_sphere(Ray ray, SceneObject s) {
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

// Define a small epsilon for floating point comparisons
const float EPS = 1e-7;

/**
 * @brief Solves a quadratic equation ax^2 + bx + c = 0 for real roots.
 * @param a The coefficient of the x^2 term.
 * @param b The coefficient of the x term.
 * @param c The constant term.
 * @param roots An output vector to store the real roots.
 * @return The number of real roots found (0, 1, or 2).
 */
int solve_quadratic(float a, float b, float c, out vec2 roots) {
    if (abs(a) < EPS) { // Linear equation
        if (abs(b) < EPS) return 0; // 0 = c, no solution or infinite solutions
        roots[0] = -c / b;
        return 1;
    }
    
    float discriminant = b*b - 4.0*a*c;
    
    if (discriminant < -EPS) {
        return 0; // No real roots
    }
    
    discriminant = max(0.0, discriminant);
    float sqrt_d = sqrt(discriminant);
    float inv_2a = 0.5 / a;
    
    roots[0] = (-b + sqrt_d) * inv_2a;
    roots[1] = (-b - sqrt_d) * inv_2a;
    
    return (discriminant < EPS) ? 1 : 2;
}

/**
 * @brief Solves a cubic equation ax^3 + bx^2 + cx + d = 0 for real roots.
 * Uses Cardano's method with a trigonometric solution for the 3-real-root case.
 * @param a The coefficient of the x^3 term.
 * @param b The coefficient of the x^2 term.
 * @param c The coefficient of the x term.
 * @param d The constant term.
 * @param roots An output vector to store the real roots.
 * @return The number of real roots found (1, 2, or 3).
 */
int solve_cubic(float a, float b, float c, float d, out vec3 roots) {
    if (abs(a) < EPS) { // Not a cubic, solve as a quadratic
        vec2 quad_roots;
        int num = solve_quadratic(b, c, d, quad_roots);
        for(int i=0; i<num; ++i) roots[i] = quad_roots[i];
        return num;
    }
    
    // Normalize to a monic cubic: x^3 + a'x^2 + b'x + c' = 0
    float inv_a = 1.0 / a;
    float an = b * inv_a;
    float bn = c * inv_a;
    float cn = d * inv_a;
    
    // Depress the cubic: y^3 + py + q = 0, where x = y - an/3
    float an2 = an * an;
    float p = bn - an2 / 3.0;
    float q = cn - an * bn / 3.0 + 2.0 * an2 * an / 27.0;
    float offset = an / 3.0;
    
    // Solve the depressed cubic
    float half_q = 0.5 * q;
    float p3_27 = p*p*p / 27.0;
    float discriminant = half_q*half_q + p3_27;

    if (discriminant >= -EPS) { // 1 real root (Cardano's formula)
        discriminant = max(0.0, discriminant);
        float sqrt_d = sqrt(discriminant);
        float A = -half_q + sqrt_d;
        float B = -half_q - sqrt_d;
        float rootA = sign(A) * pow(abs(A), 1.0/3.0);
        float rootB = sign(B) * pow(abs(B), 1.0/3.0);
        
        roots[0] = rootA + rootB - offset;
        return 1;
    } else { // 3 real roots (Trigonometric solution)
        const float TWO_PI_3 = 2.09439510239; // 2*PI/3
        float rho = sqrt(-p*p*p / 27.0);
        float theta = acos(clamp(-half_q / rho, -1.0, 1.0)) / 3.0;
        float m = 2.0 * sqrt(-p/3.0);

        roots[0] = m * cos(theta) - offset;
        roots[1] = m * cos(theta + TWO_PI_3) - offset;
        roots[2] = m * cos(theta - TWO_PI_3) - offset;
        return 3;
    }
}

/**
 * @brief Solves a monic quartic equation x^4 + ax^3 + bx^2 + cx + d = 0 for real roots.
 * Uses Ferrari's method, which involves solving a resolvent cubic.
 * @param a The coefficient of the x^3 term.
 * @param b The coefficient of the x^2 term.
 * @param c The coefficient of the x term.
 * @param d The constant term.
 * @param roots An output vector to store the real roots.
 * @return The number of real roots found (0, 1, 2, 3, or 4).
 */
int solve_quartic(float a, float b, float c, float d, out vec4 roots) {
    // Depress the quartic: x = y - a/4 -> y^4 + py^2 + qy + r = 0
    float a2 = a * a;
    float p  = b - 3.0/8.0 * a2;
    float q  = c - 0.5 * a * b + 1.0/8.0 * a2 * a;
    float r  = d - 0.25 * a * c + 1.0/16.0 * a2 * b - 3.0/256.0 * a2 * a2;
    float offset = 0.25 * a;
    
    int numRoots = 0;

    // --- Case 1: Biquadratic (q ≈ 0) ---
    // The equation becomes y^4 + py^2 + r = 0, which is a quadratic in y^2.
    if (abs(q) < EPS) {
        vec2 y2_roots;
        int num_y2_roots = solve_quadratic(1.0, p, r, y2_roots);
        for (int i = 0; i < num_y2_roots; ++i) {
            float z = y2_roots[i];
            if (z >= -EPS) {
                float y = sqrt(max(0.0, z));
                roots[numRoots++] =  y - offset;
                if (y > EPS) { // Avoid adding the same root twice if y=0
                    roots[numRoots++] = -y - offset;
                }
            }
        }
        return numRoots;
    }
    
    // --- Case 2: General quartic (Ferrari’s method) ---
    // Solve the resolvent cubic: u^3 + 2pu^2 + (p^2 - 4r)u - q^2 = 0
    vec3 cubic_roots;
    solve_cubic(1.0, 2.0*p, p*p - 4.0*r, -q*q, cubic_roots);
    
    // Pick the largest real root for u. It's guaranteed to be non-negative
    // if the original quartic has real roots.
    float u = cubic_roots[0];

    // u must be non-negative to proceed.
    if (u < 0.0) return 0;
    
    float w = sqrt(u);

    // Solve the two quadratic equations for y:
    // (y^2 + w*y + (p/2 + u/2 - q/(2w))) = 0
    // (y^2 - w*y + (p/2 + u/2 + q/(2w))) = 0
    float term_A = 0.5 * p + 0.5 * u;
    float term_B = 0.5 * q / (w + EPS); // Add EPS for stability if w is near zero
    
    vec2 quad_roots1;
    int num1 = solve_quadratic(1.0, w, term_A - term_B, quad_roots1);
    for(int i=0; i<num1; ++i) roots[numRoots++] = quad_roots1[i] - offset;
    
    vec2 quad_roots2;
    int num2 = solve_quadratic(1.0, -w, term_A + term_B, quad_roots2);
    for(int i=0; i<num2; ++i) roots[numRoots++] = quad_roots2[i] - offset;
    
    return numRoots;
}



// --- Corrected Ray-Torus Intersection ---
HitInfo intersect_torus(Ray ray, SceneObject torus) {
    HitInfo info;
    info.hit = false;

    // --- Transform ray to torus's local space (Your code is correct) ---
    vec3 local_ray_origin = ray.origin - torus.center;
    vec3 w_axis = normalize(torus.normal);
    vec3 u_axis = normalize(cross(w_axis, abs(w_axis.y) > 0.99 ? vec3(1,0,0) : vec3(0,1,0) ));
    vec3 v_axis = cross(w_axis, u_axis);
    vec3 ro = vec3(dot(local_ray_origin, u_axis), dot(local_ray_origin, w_axis), dot(local_ray_origin, v_axis));
    vec3 rd = vec3(dot(ray.direction, u_axis), dot(ray.direction, w_axis), dot(ray.direction, v_axis));

    // --- [START OF CORRECTION] Solve Quartic Equation for local-space intersection ---
    float R = torus.major_radius;
    float r = torus.minor_radius;
    float R2 = R * R;
    float r2 = r * r;

    // These coefficients are for a quartic equation At^4 + Bt^3 + Ct^2 + Dt + E = 0,
    // where A is 1 because the ray direction is normalized.
    // The code passes a,b,c,d for t^4 + a*t^3 + b*t^2 + c*t + d = 0
    
    float m = dot(ro, ro);
    float n = dot(ro, rd);
    
    // This term is part of the standard derivation (see link below)
    float term = m - r2 - R2;

    float a = 4.0 * n;
    float b = 2.0 * term + 4.0 * n * n + 4.0 * R2 * rd.y * rd.y;
    float c = 4.0 * n * term + 8.0 * R2 * ro.y * rd.y;
    float d = term * term - 4.0 * R2 * (r2 - ro.y * ro.y);

    // The quartic solver expects coefficients for t^4 + a*t^3 + ...
    // The formula we used gives coefficients for a general quartic.
    // Since dot(rd,rd) is 1, the t^4 coefficient is 1, so the coefficients match.
    // However, the standard formula is (p.p + R^2 - r^2)^2 = 4R^2(p.x^2+p.z^2)
    // Let's use a known-good set of coefficients derived from that equation.
    
    // Re-deriving based on (dot(p,p) + R^2 - r^2)^2 = 4R^2(p.x^2 + p.z^2)
    float rd_dxz = rd.x*rd.x + rd.z*rd.z;
    float ro_rd_dxz = ro.x*rd.x + ro.z*rd.z;
    float ro_dxz = ro.x*ro.x + ro.z*ro.z;
    
    float k = m + R2 - r2;

    float A_coeff = 4.0 * n;
    float B_coeff = 2.0 * k + 4.0 * n * n - 4.0 * R2 * rd_dxz;
    float C_coeff = 4.0 * n * k - 8.0 * R2 * ro_rd_dxz;
    float D_coeff = k * k - 4.0 * R2 * ro_dxz;

    vec4 roots;
    int num_roots = solve_quartic(A_coeff, B_coeff, C_coeff, D_coeff, roots);
    // --- [END OF CORRECTION] ---

    float t = 1e20;
    bool found_root = false;
    for (int i = 0; i < num_roots; i++) {
        if (roots[i] > 0.001 && roots[i] < t) {
            t = roots[i];
            found_root = true;
        }
    }

    if (!found_root) return info;

    // --- If hit, calculate world-space properties (Your code is correct) ---
    info.hit = true;
    info.t = t;
    info.position = ray.origin + t * ray.direction;
    
    vec3 hit_pos_local = ro + t * rd;
    float alpha = R / sqrt(hit_pos_local.x*hit_pos_local.x + hit_pos_local.z*hit_pos_local.z);
    vec3 normal_local = normalize(vec3(hit_pos_local.x * (1.0 - alpha), hit_pos_local.y, hit_pos_local.z * (1.0 - alpha)));

    info.normal = normalize(normal_local.x * u_axis + normal_local.y * w_axis + normal_local.z * v_axis);

    info.color = torus.color;
    info.reflectivity = torus.reflectivity;
    info.transparency = torus.transparency;
    info.refractive_index = torus.refractive_index;

    return info;
}


// This version correctly checks for intersections with all scene objects.
HitInfo trace(Ray ray) {
    HitInfo closest_hit;
    closest_hit.hit = false;
    closest_hit.t = 1e30; // A very large number (infinity)

    // Check intersections with all objects in the scene
    for (int i = 0; i < NUM_OBJECTS; ++i) {
        HitInfo current_hit;
        if (scene[i].type == SHAPE_SPHERE) {
            current_hit = intersect_sphere(ray, scene[i]);
        } else if (scene[i].type == SHAPE_TORUS) {
            current_hit = intersect_torus(ray, scene[i]);
        }

        if (current_hit.hit && current_hit.t < closest_hit.t) {
            closest_hit = current_hit;
        }
    }

    // Check for intersection with the ground plane
    HitInfo plane_hit = intersect_plane(ray);
    if (plane_hit.hit && plane_hit.t < closest_hit.t) {
        closest_hit = plane_hit;
    }

    return closest_hit;
}


// --- Light Attenuation Calculation
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
vec3 RayTraceIterative(Ray initial_ray, int max_depth, vec3 camera_pos) {
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
            vec3 local_color = phong_lighting(hit, u_light_pos, camera_pos);
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
    

    // --- Primary Ray Generation (THE CHANGE) ---
    vec2 uv = (gl_FragCoord.xy * 2.0 - u_resolution.xy) / u_resolution.y;
    const int max_depth = 10;

    // Deconstruct the camera matrix to get its position and orientation
    vec3 camera_pos   = u_camera_to_world[3].xyz;
    vec3 camera_right = u_camera_to_world[0].xyz;
    vec3 camera_up    = u_camera_to_world[1].xyz;
    vec3 camera_fwd   = u_camera_to_world[2].xyz;

    Ray primary_ray;
    primary_ray.origin = camera_pos;
    primary_ray.direction = normalize(
        uv.x * camera_right + 
        uv.y * camera_up + 
        u_focal_length * camera_fwd
    );
    
    // --- Trace and Color ---
    // Pass the camera position to the ray tracer.
    vec3 final_color = RayTraceIterative(primary_ray, max_depth, camera_pos);

    FragColor = vec4(final_color, 1.0);
}