// Este arquivo é um Fragment Shader escrito em GLSL (OpenGL Shading Language).
// Sua principal responsabilidade é determinar a cor de cada pixel (fragmento) na tela.
// Ele implementa um algoritmo de Ray Tracing iterativo para renderizar uma cena 3D
// com esferas e toros, incluindo suporte para reflexão, refração, sombras e
// um plano de chão com textura de tabuleiro de xadrez.

#version 330 core
out vec4 FragColor; // Variável de saída que define a cor final do fragmento

// --- Uniforms ---
// Variáveis passadas do script Python para o shader.
uniform vec2 u_resolution;      // A resolução da tela (largura, altura)
uniform mat4 u_camera_to_world; // Matriz que transforma as coordenadas da câmera para o mundo
uniform vec3 u_light_pos;       // A posição da fonte de luz na cena
uniform float u_focal_length;   // A distância focal da câmera

// --- Estruturas de Dados ---
// Constantes para identificar os tipos de objetos na cena
const int SHAPE_SPHERE = 1; // Identificador para uma esfera
const int SHAPE_TORUS = 2;  // Identificador para um toro

// Um raio é definido por uma origem e uma direção.
struct Ray {
    vec3 origin;
    vec3 direction;
};

// Armazena informações sobre a interseção de um raio com um objeto.
struct HitInfo {
    bool hit;               // Verdadeiro se houve uma interseção
    float t;                // Distância ao longo do raio até o ponto de interseção
    vec3 position;          // Posição 3D do ponto de interseção
    vec3 normal;            // Vetor normal na superfície no ponto de interseção
    vec3 color;             // Cor do objeto no ponto de interseção
    float reflectivity;     // Coeficiente de refletividade do material (0 a 1)
    float transparency;     // Coeficiente de transparência do material (0 a 1)
    float refractive_index; // Índice de refração do material
};

// Representa um objeto unificado na cena, que pode ser uma esfera ou um toro.
struct SceneObject {
    int type; // Tipo do objeto (SHAPE_SPHERE ou SHAPE_TORUS)

    // Propriedades geométricas
    vec3 center;        // Centro da esfera ou do toro
    float radius;       // Raio da esfera
    vec3 normal;        // Vetor normal para a orientação do toro
    float major_radius; // Raio maior do toro
    float minor_radius; // Raio menor do toro

    // Propriedades do material
    vec3 color;
    float reflectivity;
    float transparency;
    float refractive_index;
};

// Estrutura de dados para a pilha iterativa do Ray Tracing.
// Evita o uso de recursão, que não é bem suportada em todos os drivers GLSL.
struct RayState {
    Ray ray;                        // O raio atual sendo traçado
    vec3 throughput;                // A cor/energia transportada por este raio
    int depth;                      // A profundidade atual do raio (quantas vezes ele ricocheteou)
    float current_refractive_index; // O índice de refração do meio em que o raio está atualmente
};

// --- Definição da Cena ---
const int NUM_OBJECTS = 5; // O número de objetos na cena
uniform SceneObject scene[NUM_OBJECTS]; // Um array contendo todos os objetos da cena

// --- Interseção Raio-Esfera ---
// Resolve a equação quadrática para a interseção de um raio com uma esfera.
// Retorna uma estrutura HitInfo com os detalhes da interseção.
HitInfo intersect_sphere(Ray ray, SceneObject s) {
    HitInfo info;
    info.hit = false;
    vec3 oc = ray.origin - s.center;
    float a = dot(ray.direction, ray.direction);
    float b = 2.0 * dot(oc, ray.direction);
    float c = dot(oc, oc) - s.radius * s.radius;
    float discriminant = b*b - 4.0*a*c;

    if (discriminant < 0.0) {
        return info; // Nenhuma interseção
    }

    float t = (-b - sqrt(discriminant)) / (2.0 * a);
    if (t < 0.0) { // Se a interseção está atrás do raio, tenta a outra raiz
        t = (-b + sqrt(discriminant)) / (2.0 * a);
    }
    
    if (t > 0.001) { // Epsilon para evitar auto-interseção
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

// --- Interseção Raio-Plano ---
// Calcula a interseção com o plano horizontal do chão e aplica um padrão de tabuleiro de xadrez.
HitInfo intersect_plane(Ray ray) {
    HitInfo info;
    info.hit = false;
    
    // Define o plano do chão em y = -1.0
    const float ground_level = -1.0; 
    const vec3 plane_normal = vec3(0.0, 1.0, 0.0);

    // Calcula a distância 't' até o ponto de interseção.
    float denominator = dot(ray.direction, plane_normal);
    if (abs(denominator) > 0.0001) { // O raio é paralelo ao plano se o denominador for próximo de zero.
        float t = (ground_level - ray.origin.y) / denominator;
        
        // Garante que a interseção esteja na frente do raio.
        if (t > 0.001) {
            info.hit = true;
            info.t = t;
            info.position = ray.origin + t * ray.direction;
            info.normal = plane_normal;
            
            // Define as propriedades do material para o chão
            info.transparency = 0.0;
            info.refractive_index = 1.0;
            info.reflectivity = 0.05; // Um chão levemente reflexivo fica bom

            // --- Aplica o Padrão de Tabuleiro de Xadrez ---
            vec3 color_white = vec3(0.9, 0.9, 0.9);
            vec3 color_black = vec3(0.1, 0.1, 0.1);
            
            // Usa floor e mod nas coordenadas x e z para criar os "azulejos".
            if (mod(floor(info.position.x) + floor(info.position.z), 2.0) == 0.0) {
                info.color = color_white;
            } else {
                info.color = color_black;
            }
        }
    }
    
    return info;
}

// Define um pequeno epsilon para comparações de ponto flutuante
const float EPS = 1e-7;

/**
 * @brief Resolve uma equação quadrática ax^2 + bx + c = 0 para raízes reais.
 * @param a O coeficiente do termo x^2.
 * @param b O coeficiente do termo x.
 * @param c O termo constante.
 * @param roots Um vetor de saída para armazenar as raízes reais.
 * @return O número de raízes reais encontradas (0, 1 ou 2).
 */
int solve_quadratic(float a, float b, float c, out vec2 roots) {
    if (abs(a) < EPS) { // Equação linear
        if (abs(b) < EPS) return 0; // 0 = c, sem solução ou infinitas soluções
        roots[0] = -c / b;
        return 1;
    }
    
    float discriminant = b*b - 4.0*a*c;
    
    if (discriminant < -EPS) {
        return 0; // Sem raízes reais
    }
    
    discriminant = max(0.0, discriminant);
    float sqrt_d = sqrt(discriminant);
    float inv_2a = 0.5 / a;
    
    roots[0] = (-b + sqrt_d) * inv_2a;
    roots[1] = (-b - sqrt_d) * inv_2a;
    
    return (discriminant < EPS) ? 1 : 2;
}

/**
 * @brief Resolve uma equação cúbica ax^3 + bx^2 + cx + d = 0 para raízes reais.
 * Usa o método de Cardano com uma solução trigonométrica para o caso de 3 raízes reais.
 * @param a O coeficiente do termo x^3.
 * @param b O coeficiente do termo x^2.
 * @param c O coeficiente do termo x.
 * @param d O termo constante.
 * @param roots Um vetor de saída para armazenar as raízes reais.
 * @return O número de raízes reais encontradas (1, 2 ou 3).
 */
int solve_cubic(float a, float b, float c, float d, out vec3 roots) {
    if (abs(a) < EPS) { // Não é uma cúbica, resolve como uma quadrática
        vec2 quad_roots;
        int num = solve_quadratic(b, c, d, quad_roots);
        for(int i=0; i<num; ++i) roots[i] = quad_roots[i];
        return num;
    }
    
    // Normaliza para uma cúbica mônica: x^3 + a'x^2 + b'x + c' = 0
    float inv_a = 1.0 / a;
    float an = b * inv_a;
    float bn = c * inv_a;
    float cn = d * inv_a;
    
    // Deprime a cúbica: y^3 + py + q = 0, onde x = y - an/3
    float an2 = an * an;
    float p = bn - an2 / 3.0;
    float q = cn - an * bn / 3.0 + 2.0 * an2 * an / 27.0;
    float offset = an / 3.0;
    
    // Resolve a cúbica deprimida
    float half_q = 0.5 * q;
    float p3_27 = p*p*p / 27.0;
    float discriminant = half_q*half_q + p3_27;

    if (discriminant >= -EPS) { // 1 raiz real (fórmula de Cardano)
        discriminant = max(0.0, discriminant);
        float sqrt_d = sqrt(discriminant);
        float A = -half_q + sqrt_d;
        float B = -half_q - sqrt_d;
        float rootA = sign(A) * pow(abs(A), 1.0/3.0);
        float rootB = sign(B) * pow(abs(B), 1.0/3.0);
        
        roots[0] = rootA + rootB - offset;
        return 1;
    } else { // 3 raízes reais (solução trigonométrica)
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
 * @brief Resolve uma equação quârtica mônica x^4 + ax^3 + bx^2 + cx + d = 0 para raízes reais.
 * Usa o método de Ferrari, que envolve resolver uma cúbica resolvente.
 * @param a O coeficiente do termo x^3.
 * @param b O coeficiente do termo x^2.
 * @param c O coeficiente do termo x.
 * @param d O termo constante.
 * @param roots Um vetor de saída para armazenar as raízes reais.
 * @return O número de raízes reais encontradas (0, 1, 2, 3 ou 4).
 */
int solve_quartic(float a, float b, float c, float d, out vec4 roots) {
    // Deprime a quârtica: x = y - a/4 -> y^4 + py^2 + qy + r = 0
    float a2 = a * a;
    float p  = b - 3.0/8.0 * a2;
    float q  = c - 0.5 * a * b + 1.0/8.0 * a2 * a;
    float r  = d - 0.25 * a * c + 1.0/16.0 * a2 * b - 3.0/256.0 * a2 * a2;
    float offset = 0.25 * a;
    
    int numRoots = 0;

    // --- Caso 1: Biquadrática (q ≈ 0) ---
    // A equação se torna y^4 + py^2 + r = 0, que é uma quadrática em y^2.
    if (abs(q) < EPS) {
        vec2 y2_roots;
        int num_y2_roots = solve_quadratic(1.0, p, r, y2_roots);
        for (int i = 0; i < num_y2_roots; ++i) {
            float z = y2_roots[i];
            if (z >= -EPS) {
                float y = sqrt(max(0.0, z));
                roots[numRoots++] =  y - offset;
                if (y > EPS) { // Evita adicionar a mesma raiz duas vezes se y=0
                    roots[numRoots++] = -y - offset;
                }
            }
        }
        return numRoots;
    }
    
    // --- Caso 2: Quârtica geral (método de Ferrari) ---
    // Resolve a cúbica resolvente: u^3 + 2pu^2 + (p^2 - 4r)u - q^2 = 0
    vec3 cubic_roots;
    solve_cubic(1.0, 2.0*p, p*p - 4.0*r, -q*q, cubic_roots);
    
    // Pega a maior raiz real para u. É garantido que seja não-negativa
    // se a quârtica original tiver raízes reais.
    float u = cubic_roots[0];

    // u deve ser não-negativo para prosseguir.
    if (u < 0.0) return 0;
    
    float w = sqrt(u);

    // Resolve as duas equações quadráticas para y:
    float term_A = 0.5 * p + 0.5 * u;
    float term_B = 0.5 * q / (w + EPS); // Adiciona EPS para estabilidade se w for próximo de zero
    
    vec2 quad_roots1;
    int num1 = solve_quadratic(1.0, w, term_A - term_B, quad_roots1);
    for(int i=0; i<num1; ++i) roots[numRoots++] = quad_roots1[i] - offset;
    
    vec2 quad_roots2;
    int num2 = solve_quadratic(1.0, -w, term_A + term_B, quad_roots2);
    for(int i=0; i<num2; ++i) roots[numRoots++] = quad_roots2[i] - offset;
    
    return numRoots;
}

// --- Interseção Raio-Toro ---
// Calcula a interseção de um raio com um objeto toro.
HitInfo intersect_torus(Ray ray, SceneObject torus) {
    HitInfo info;
    info.hit = false;

    // Transforma o raio para o espaço local do toro (onde o toro está centrado na origem e alinhado com os eixos).
    vec3 local_ray_origin = ray.origin - torus.center;
    vec3 w_axis = normalize(torus.normal);
    vec3 u_axis = normalize(cross(w_axis, abs(w_axis.y) > 0.99 ? vec3(1,0,0) : vec3(0,1,0) ));
    vec3 v_axis = cross(w_axis, u_axis);
    vec3 ro = vec3(dot(local_ray_origin, u_axis), dot(local_ray_origin, w_axis), dot(local_ray_origin, v_axis));
    vec3 rd = vec3(dot(ray.direction, u_axis), dot(ray.direction, w_axis), dot(ray.direction, v_axis));

    // Resolve a equação quârtica para a interseção no espaço local.
    float R = torus.major_radius;
    float r = torus.minor_radius;
    
    float m = dot(ro, ro);
    float n = dot(ro, rd);
    
    float rd_dxz = rd.x*rd.x + rd.z*rd.z;
    float ro_rd_dxz = ro.x*rd.x + ro.z*rd.z;
    float ro_dxz = ro.x*ro.x + ro.z*ro.z;
    
    float k = m + R*R - r*r;

    // Coeficientes da equação quârtica At^4 + Bt^3 + Ct^2 + Dt + E = 0
    // Como a direção do raio é normalizada, o coeficiente de t^4 é 1.
    float A_coeff = 4.0 * n;
    float B_coeff = 2.0 * k + 4.0 * n * n - 4.0 * R*R * rd_dxz;
    float C_coeff = 4.0 * n * k - 8.0 * R*R * ro_rd_dxz;
    float D_coeff = k * k - 4.0 * R*R * ro_dxz;

    vec4 roots;
    int num_roots = solve_quartic(A_coeff, B_coeff, C_coeff, D_coeff, roots);

    // Encontra a menor raiz positiva, que corresponde à interseção mais próxima.
    float t = 1e20;
    bool found_root = false;
    for (int i = 0; i < num_roots; i++) {
        if (roots[i] > 0.001 && roots[i] < t) {
            t = roots[i];
            found_root = true;
        }
    }

    if (!found_root) return info;

    // Se houver interseção, calcula as propriedades no espaço do mundo.
    info.hit = true;
    info.t = t;
    info.position = ray.origin + t * ray.direction;
    
    // Calcula a normal no ponto de interseção
    vec3 hit_pos_local = ro + t * rd;
    float alpha = R / sqrt(hit_pos_local.x*hit_pos_local.x + hit_pos_local.z*hit_pos_local.z);
    vec3 normal_local = normalize(vec3(hit_pos_local.x * (1.0 - alpha), hit_pos_local.y, hit_pos_local.z * (1.0 - alpha)));

    // Transforma a normal de volta para o espaço do mundo
    info.normal = normalize(normal_local.x * u_axis + normal_local.y * w_axis + normal_local.z * v_axis);

    info.color = torus.color;
    info.reflectivity = torus.reflectivity;
    info.transparency = torus.transparency;
    info.refractive_index = torus.refractive_index;

    return info;
}

// --- Traçado de Raio Principal ---
// Lança um raio na cena e encontra a interseção mais próxima com qualquer objeto.
HitInfo trace(Ray ray) {
    HitInfo closest_hit;
    closest_hit.hit = false;
    closest_hit.t = 1e30; // Um número muito grande (infinito)

    // Verifica a interseção com todos os objetos na cena
    for (int i = 0; i < NUM_OBJECTS; ++i) {
        // --- Otimização: Verificação da Esfera Delimitadora (Bounding Sphere) ---
        // Antes de fazer o teste de interseção preciso e caro (especialmente para o toro),
        // fazemos um teste barato contra a esfera delimitadora do objeto.
        // Se o raio não puder atingir essa esfera mais perto do que o acerto mais próximo
        // que já encontramos, podemos pular o teste caro com segurança.
        float bounding_radius;
        if (scene[i].type == SHAPE_SPHERE) {
            bounding_radius = scene[i].radius;
        } else { // SHAPE_TORUS
            bounding_radius = scene[i].major_radius + scene[i].minor_radius;
        }

        // Teste de interseção de esfera simplificado que retorna apenas 't'.
        vec3 oc = ray.origin - scene[i].center;
        float b = dot(oc, ray.direction);
        float c = dot(oc, oc) - bounding_radius * bounding_radius;
        float discriminant = b*b - c; // Como a direção do raio é normalizada, a=1

        // Se o raio erra a esfera delimitadora, pule para o próximo objeto.
        if (discriminant < 0.0) {
            continue;
        }

        // Se a esfera delimitadora está mais longe que o objeto mais próximo já encontrado, pule.
        float t_bound = -b - sqrt(discriminant);
        if (t_bound > closest_hit.t) {
            continue;
        }
        // --- Fim da Verificação da Esfera Delimitadora ---

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

    // Verifica a interseção com o plano do chão
    HitInfo plane_hit = intersect_plane(ray);
    if (plane_hit.hit && plane_hit.t < closest_hit.t) {
        closest_hit = plane_hit;
    }

    return closest_hit;
}

// --- Cálculo de Atenuação da Luz (Sombras) ---
// Calcula quanta luz chega a um ponto, considerando objetos transparentes no caminho.
vec3 calculate_light_attenuation(vec3 point, vec3 light_pos) {
    // Começa com um filtro de luz branco puro (100% de passagem de luz).
    vec3 light_filter = vec3(1.0); 

    float light_dist = length(light_pos - point);
    
    // Lança um "raio de sombra" do ponto em direção à luz.
    Ray shadow_ray;
    shadow_ray.origin = point + normalize(light_pos - point) * 0.001; 
    shadow_ray.direction = normalize(light_pos - point);

    float distance_traveled = 0.0;

    // Itera para encontrar múltiplos objetos transparentes no caminho da luz.
    for(int i = 0; i < 2; i++) { 
        HitInfo hit = trace(shadow_ray);

        // Se atingir um objeto que está entre o ponto e a luz...
        if (hit.hit && (hit.t + distance_traveled < light_dist)) {
            
            // Calcula o filtro de luz para este objeto. A luz que passa é
            // tingida pela cor do objeto e atenuada por sua transparência.
            // Um objeto totalmente opaco (transparency=0) terá um filtro preto.
            vec3 object_filter = hit.color * hit.transparency;

            // Multiplica nosso filtro de luz acumulado pelo filtro deste objeto.
            light_filter *= object_filter;
            
            // Se a luz foi completamente bloqueada, podemos parar mais cedo.
            if (dot(light_filter, light_filter) == 0.0) {
                return vec3(0.0);
            }
            
            // Continua o traçado a partir da nova posição.
            distance_traveled += hit.t;
            shadow_ray.origin = hit.position + shadow_ray.direction * 0.001;

        } else {
            // Não há mais objetos no caminho para a luz.
            break;
        }
    }

    return light_filter;
}

// --- Cálculo de Iluminação de Phong ---
// Calcula a cor de um ponto na superfície usando o modelo de iluminação de Phong.
// Agora usa a cor de atenuação para sombras sofisticadas e filtradas.
vec3 phong_lighting(HitInfo info, vec3 light_pos, vec3 camera_pos) {
    // Componente ambiente: uma luz de fundo fraca para que as sombras não sejam totalmente pretas.
    vec3 ambient = 0.05 * info.color;

    // Calcula a cor de atenuação da luz usando nossa nova função avançada.
    vec3 attenuation = calculate_light_attenuation(info.position, light_pos);

    // Se a atenuação for preta, estamos em uma sombra totalmente opaca.
    if (dot(attenuation, attenuation) == 0.0) {
        return ambient;
    }

    // Componente difusa: reflete a luz uniformemente em todas as direções.
    vec3 light_dir = normalize(light_pos - info.position);
    float diff = max(dot(info.normal, light_dir), 0.0);
    vec3 diffuse = diff * info.color;

    // Componente especular: cria o "brilho" ou destaque.
    vec3 view_dir = normalize(camera_pos - info.position);
    vec3 reflect_dir = reflect(-light_dir, info.normal);
    float spec = pow(max(dot(view_dir, reflect_dir), 0.0), 32.0);
    vec3 specular = 0.5 * spec * vec3(1.0, 1.0, 1.0); // Destaques brancos

    // A cor final é a soma das componentes ambiente, difusa e especular.
    // As componentes difusa e especular são filtradas pela cor de atenuação da sombra.
    return ambient + (diffuse + specular) * attenuation;
}

/**
 * @brief Calcula o coeficiente de reflexão usando a aproximação de Schlick (Efeito Fresnel).
 * Determina a proporção de luz refletida em relação à luz refratada para um material dielétrico.
 * @param cos_theta O cosseno do ângulo entre o vetor de luz incidente e a normal.
 * @param n1 O índice de refração do meio em que o raio está.
 * @param n2 O índice de refração do meio em que o raio está entrando.
 * @return A quantidade de luz que é refletida (um valor entre 0.0 e 1.0).
 */
float calculate_fresnel(float cos_theta, float n1, float n2) {
    float r0 = (n1 - n2) / (n1 + n2);
    r0 = r0 * r0;
    // Lida com a reflexão interna total
    float n = n1 / n2;
    float sin_t2 = n * n * (1.0 - cos_theta * cos_theta);
    if (sin_t2 > 1.0) {
        return 1.0; // Reflexão interna total
    }
    return r0 + (1.0 - r0) * pow(1.0 - cos_theta, 5.0);
}

/**
 * @brief Uma versão iterativa da função RayTrace que usa uma pilha manual
 * para lidar com reflexão e refração sem recursão.
 * @param initial_ray O primeiro raio lançado da câmera.
 * @param max_depth O número máximo de vezes que um raio pode ricochetear.
 * @param camera_pos A posição da câmera, necessária para o cálculo de iluminação.
 * @return A cor final calculada para o caminho do raio.
 */
vec3 RayTraceIterative(Ray initial_ray, int max_depth, vec3 camera_pos) {
    vec3 final_color = vec3(0.0);

    const int STACK_SIZE = 3; // Tamanho máximo da pilha de raios
    RayState stack[STACK_SIZE];
    int stack_ptr = 0;

    const float RAY_EPSILON = 0.001; // Pequeno deslocamento para evitar auto-interseção

    // Empilha o raio inicial
    stack[stack_ptr].ray = initial_ray;
    stack[stack_ptr].throughput = vec3(1.0, 1.0, 1.0); // O raio inicial carrega 100% da energia
    stack[stack_ptr].depth = 0;
    stack[stack_ptr].current_refractive_index = 1.0; // Começa no "ar"
    stack_ptr++;

    // Processa a pilha de raios até que ela esteja vazia
    while (stack_ptr > 0) {
        // Desempilha o estado atual do raio
        stack_ptr--;
        RayState current_state = stack[stack_ptr];

        // Traça o raio na cena
        HitInfo hit = trace(current_state.ray);

        // Se o raio não atingir nenhum objeto, calcula uma cor de céu procedural.
        if (!hit.hit) {
            vec3 sky_color_zenith = vec3(0.5, 0.7, 1.0);  // Azul claro para o zênite
            vec3 sky_color_horizon = vec3(0.8, 0.9, 1.0); // Branco/azulado para o horizonte

            // Interpola entre as cores do céu com base na direção y do raio.
            float t = 0.5 + 0.5 * current_state.ray.direction.y;
            vec3 sky_color = mix(sky_color_horizon, sky_color_zenith, t);

            final_color += sky_color * current_state.throughput;
            continue; // Vai para o próximo raio na pilha
        }

        // Determina a normal da superfície e os índices de refração
        vec3 outward_normal;
        float n1, n2;
        if (dot(current_state.ray.direction, hit.normal) < 0.0) { // O raio está entrando no objeto
            outward_normal = hit.normal;
            n1 = current_state.current_refractive_index;
            n2 = hit.refractive_index;
        } else { // O raio está saindo do objeto
            outward_normal = -hit.normal;
            n1 = hit.refractive_index;
            n2 = 1.0; // Assume que está saindo para o "ar"
        }

        // Calcula a refletância de Fresnel
        float cos_theta = abs(dot(current_state.ray.direction, outward_normal));
        float fresnel_reflectance = calculate_fresnel(cos_theta, n1, n2);

        // Calcula a contribuição da cor local (iluminação de Phong)
        float local_coef = 1.0 - hit.reflectivity - hit.transparency;
        if (local_coef > 0.0) {
            vec3 local_color = phong_lighting(hit, u_light_pos, camera_pos);
            final_color += local_color * local_coef * current_state.throughput;
        }

        // Se a profundidade máxima foi atingida, para por aqui.
        if (current_state.depth >= max_depth - 1) {
            continue;
        }

        // Lida com a refração (transparência)
        if (hit.transparency > 0.0) {
            vec3 refraction_dir = refract(current_state.ray.direction, outward_normal, n1 / n2);
            // Se a refração é válida e a pilha não está cheia...
            if (dot(refraction_dir, refraction_dir) > 0.0 && stack_ptr < STACK_SIZE) {
                // Empilha um novo raio para a refração
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

        // Lida com a reflexão
        float total_reflectivity = hit.reflectivity + (1.0 - hit.reflectivity) * fresnel_reflectance;
        if (total_reflectivity > 0.0) {
            // Se a pilha não está cheia...
            if (stack_ptr < STACK_SIZE) {
                // Empilha um novo raio para a reflexão
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

// --- Função Principal ---
// O ponto de entrada do fragment shader. É executado para cada pixel.
void main() {
    // Converte as coordenadas do pixel (gl_FragCoord) para coordenadas de tela normalizadas (-1 a 1).
    vec2 uv = (gl_FragCoord.xy * 2.0 - u_resolution.xy) / u_resolution.y;
    const int max_depth = 4; // Profundidade máxima de recursão para os raios

    // Desconstrói a matriz da câmera para obter sua posição e vetores de orientação.
    vec3 camera_pos   = u_camera_to_world[3].xyz;
    vec3 camera_right = u_camera_to_world[0].xyz;
    vec3 camera_up    = u_camera_to_world[1].xyz;
    vec3 camera_fwd   = u_camera_to_world[2].xyz;

    // Gera o raio primário a partir da posição da câmera através do pixel atual.
    Ray primary_ray;
    primary_ray.origin = camera_pos;
    primary_ray.direction = normalize(
        uv.x * camera_right + 
        uv.y * camera_up + 
        u_focal_length * camera_fwd
    );
    
    // Traça o raio e calcula a cor final.
    vec3 final_color = RayTraceIterative(primary_ray, max_depth, camera_pos);

    // Define a cor do fragmento.
    FragColor = vec4(final_color, 1.0);
}
