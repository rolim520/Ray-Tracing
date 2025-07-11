import sys
import os
import glfw
from OpenGL.GL import *
import numpy as np

# --- Helper Functions (no changes) ---

def read_shader_file(filename):
    if not os.path.exists(filename):
        print(f"ERROR: Shader file not found: {filename}")
        sys.exit()
    with open(filename, 'r') as f:
        return f.read()

def compile_shader(source, shader_type):
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)
    if not glGetShaderiv(shader, GL_COMPILE_STATUS):
        error = glGetShaderInfoLog(shader).decode()
        print(f"ERROR compiling shader: {error}")
        sys.exit()
    return shader

def framebuffer_size_callback(window, width, height):
    global window_width, window_height
    window_width = width
    window_height = height
    glViewport(0, 0, width, height)

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

# --- FINAL CORRECTED: Input Processing for Ground-Parallel Movement ---

def process_input(window, eye, target, up_vector, delta_time):
    """
    Handles keyboard input for camera movement that is parallel to the ground plane.
    """
    camera_speed = 3.0 * delta_time

    new_eye = eye.copy()
    new_target = target.copy()

    # --- THE FIX IS HERE ---
    # 1. Calculate the true forward vector (which may point up or down)
    forward_vector = normalize(target - eye)

    # 2. Project the forward vector onto the XZ plane to get a ground-parallel direction
    #    This ensures W and S keys move you forward/backward without changing altitude.
    parallel_forward = normalize(np.array([forward_vector[0], 0, forward_vector[2]]))
    
    # 3. The right vector is calculated from the parallel forward vector
    right_vector = normalize(np.cross(parallel_forward, up_vector))
    
    # The world up vector is used for space/shift to move straight up/down
    world_up_vector = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    # Apply displacement using the new parallel vectors
    if glfw.get_key(window, glfw.KEY_W) == glfw.PRESS:
        # Use the parallel vector for forward/backward movement
        displacement = parallel_forward * camera_speed
        new_eye += displacement
        new_target += displacement
        
    if glfw.get_key(window, glfw.KEY_S) == glfw.PRESS:
        displacement = -parallel_forward * camera_speed
        new_eye += displacement
        new_target += displacement

    if glfw.get_key(window, glfw.KEY_A) == glfw.PRESS:
        # The right vector is already parallel to the ground
        displacement = -right_vector * camera_speed
        new_eye += displacement
        new_target += displacement
        
    if glfw.get_key(window, glfw.KEY_D) == glfw.PRESS:
        displacement = right_vector * camera_speed
        new_eye += displacement
        new_target += displacement
        
    if glfw.get_key(window, glfw.KEY_SPACE) == glfw.PRESS:
        displacement = world_up_vector * camera_speed
        new_eye += displacement
        new_target += displacement
        
    if glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS:
        displacement = -world_up_vector * camera_speed
        new_eye += displacement
        new_target += displacement
        
    return new_eye, new_target


# --- Main Application ---

window_width = 1000
window_height = 800

def main():
    if not glfw.init():
        print("Could not initialize GLFW")
        return
        
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)

    window = glfw.create_window(window_width, window_height, "Python GLFW Ray Tracer", None, None)
    if not window:
        print("Failed to create GLFW window.")
        glfw.terminate()
        return

    glfw.make_context_current(window)
    glfw.set_framebuffer_size_callback(window, framebuffer_size_callback)
    
    # --- Code is the same until the Main Loop ---
    vertex_shader = compile_shader(read_shader_file("vertex_shader.glsl"), GL_VERTEX_SHADER)
    fragment_shader = compile_shader(read_shader_file("fragment_shader.glsl"), GL_FRAGMENT_SHADER)
    shader_program = glCreateProgram()
    glAttachShader(shader_program, vertex_shader)
    glAttachShader(shader_program, fragment_shader)
    glLinkProgram(shader_program)
    if not glGetProgramiv(shader_program, GL_LINK_STATUS):
        print("Linker failure: " + str(glGetProgramInfoLog(shader_program)))
        sys.exit()
    glDeleteShader(vertex_shader)
    glDeleteShader(fragment_shader)
    quad_vertices = np.array([-1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0], dtype=np.float32)
    VAO = glGenVertexArrays(1)
    VBO = glGenBuffers(1)
    glBindVertexArray(VAO)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, quad_vertices.nbytes, quad_vertices, GL_STATIC_DRAW)
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * 4, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glBindVertexArray(0)
    locations = {
        "u_resolution": glGetUniformLocation(shader_program, "u_resolution"),
        "u_light_pos": glGetUniformLocation(shader_program, "u_light_pos"),
        "u_camera_to_world": glGetUniformLocation(shader_program, "u_camera_to_world")
    }
    eye = np.array([3.0, 1.0, 1.5], dtype=np.float32)
    target = np.array([0.0, 0.0, -3.0], dtype=np.float32)
    up_vector = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    last_frame_time = 0.0
    
    print("Running ray tracer... Use WASD, Space, and Shift to move.")
    print("Press ESC or close window to exit.")
    
    # --- Main Loop ---
    while not glfw.window_should_close(window):
        current_time = glfw.get_time()
        delta_time = current_time - last_frame_time
        last_frame_time = current_time

        eye, target = process_input(window, eye, target, up_vector, delta_time)
        
        if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
            glfw.set_window_should_close(window, True)
        
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)
        glUseProgram(shader_program)

        z_axis = normalize(eye - target)
        x_axis = normalize(np.cross(up_vector, z_axis))
        y_axis = normalize(np.cross(z_axis, x_axis))
        view_dir = -z_axis
        camera_to_world = np.array([
            [x_axis[0], y_axis[0], view_dir[0], eye[0]],
            [x_axis[1], y_axis[1], view_dir[1], eye[1]],
            [x_axis[2], y_axis[2], view_dir[2], eye[2]],
            [0.0,       0.0,       0.0,        1.0]
        ], dtype=np.float32)

        glUniform2f(locations["u_resolution"], window_width, window_height)
        glUniform3f(locations["u_light_pos"], -2.0, 4.0, 1.0)
        
        # --- THE ONLY CHANGE IS HERE ---
        glUniformMatrix4fv(locations["u_camera_to_world"], 1, GL_TRUE, camera_to_world)
        
        glBindVertexArray(VAO)
        glDrawArrays(GL_TRIANGLES, 0, 6)

        glfw.swap_buffers(window)
        glfw.poll_events()

    # --- Terminate ---
    glDeleteVertexArrays(1, [VAO])
    glDeleteBuffers(1, [VBO])
    glDeleteProgram(shader_program)
    glfw.terminate()

if __name__ == "__main__":
    main()