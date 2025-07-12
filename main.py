import sys
import os
import glfw
from OpenGL.GL import *
import numpy as np

# Add this global variable near the top with the other globals
focal_length = 2.0

window_width = 1000
window_height = 800

# --- NEW: Globals for Mouse Look ---
last_x, last_y = window_width / 2, window_height / 2
first_mouse = True
yaw = -90.0  # Yaw is initialized to -90.0 degrees because a yaw of 0.0 results in a direction vector pointing to the right.
pitch = 0.0

# --- NEW: State for mouse capture ---
mouse_captured = True

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

# --- NEW: Key callback for toggling mouse capture ---
def key_callback(window, key, scancode, action, mods):
    """
    Handles key presses for toggling states.
    Specifically, it toggles mouse capture on/off with the ESC key.
    """
    global mouse_captured, first_mouse
    
    # We only care about the key press event, not release or repeat
    if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
        mouse_captured = not mouse_captured # Toggle the state
        
        if mouse_captured:
            # Re-capture the mouse
            glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)
            # Reset first_mouse to prevent camera jump on re-capture
            first_mouse = True 
        else:
            # Release the mouse
            glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_NORMAL)

# Add this new function anywhere before the main() function
def scroll_callback(window, x_offset, y_offset):
    """
    Handles mouse scroll wheel input to adjust the camera's focal length for zooming.
    """
    global focal_length
    # y_offset is positive for scroll up (zoom in), negative for scroll down (zoom out)
    zoom_speed = 0.2
    focal_length += y_offset * zoom_speed

    
    # Clamp the focal length to a reasonable range to prevent issues
    if focal_length < 0.5:
        focal_length = 0.5
    if focal_length > 10.0:
        focal_length = 10.0

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

def cursor_pos_callback(window, xpos, ypos):
    """
    Handles mouse movement to change the camera's view direction (yaw and pitch).
    """
    global mouse_captured, first_mouse, last_x, last_y, yaw, pitch

    # If the mouse is not captured, do nothing with camera rotation
    if not mouse_captured:
        return

    if first_mouse:
        last_x = xpos
        last_y = ypos
        first_mouse = False

    xoffset = xpos - last_x
    yoffset = last_y - ypos # reversed since y-coordinates go from bottom to top
    last_x = xpos
    last_y = ypos

    sensitivity = 0.1
    xoffset *= sensitivity
    yoffset *= sensitivity

    yaw += xoffset
    pitch += yoffset

    # Make sure that when pitch is out of bounds, screen doesn't get flipped
    if pitch > 89.0:
        pitch = 89.0
    if pitch < -89.0:
        pitch = -89.0


# --- Main Application ---

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
    glfw.set_scroll_callback(window, scroll_callback)
    glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED) # Capture the cursor
    glfw.set_cursor_pos_callback(window, cursor_pos_callback) # Register the new callback
    glfw.set_key_callback(window, key_callback)
    
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
        "u_camera_to_world": glGetUniformLocation(shader_program, "u_camera_to_world"),
        "u_focal_length": glGetUniformLocation(shader_program, "u_focal_length")
    }
    
    eye = np.array([3.0, 1.0, 1.5], dtype=np.float32)
    initial_direction = np.array([0.0, 0.0, -3.0], dtype=np.float32) - eye
    initial_direction = normalize(initial_direction)
    
    global yaw, pitch
    yaw = np.degrees(np.arctan2(initial_direction[2], initial_direction[0]))
    pitch = np.degrees(np.arcsin(initial_direction[1]))
    
    target = eye + initial_direction 

    up_vector = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    last_frame_time = 0.0
    
    # --- MODIFIED: Update instructions to the user ---
    print("Running ray tracer... Use WASD, Space, and Shift to move.")
    print("Use the mouse to look around. Scroll to zoom.")
    print("Press ESC to release/capture the mouse.")
    print("Close the window with the 'X' button to exit.")
    
    # --- Main Loop ---
    while not glfw.window_should_close(window):
        current_time = glfw.get_time()
        delta_time = current_time - last_frame_time
        last_frame_time = current_time

        # --- MODIFIED: Only process keyboard input if the window is focused ---
        # (This is good practice when the mouse is released)
        if glfw.get_window_attrib(window, glfw.FOCUSED):
            eye, target = process_input(window, eye, target, up_vector, delta_time)
        
        # --- The rest of the camera logic is the same ---
        direction = np.empty(3, dtype=np.float32)
        direction[0] = np.cos(np.radians(yaw)) * np.cos(np.radians(pitch))
        direction[1] = np.sin(np.radians(pitch))
        direction[2] = np.sin(np.radians(yaw)) * np.cos(np.radians(pitch))
        view_dir = normalize(direction)
        target = eye + view_dir
        
        # --- REMOVED: The check for the ESCAPE key is gone from here ---
        # if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
        #     glfw.set_window_should_close(window, True)
        
        # ... (Rendering code remains exactly the same) ...
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)
        glUseProgram(shader_program)
        z_axis = -view_dir 
        x_axis = normalize(np.cross(up_vector, z_axis))
        y_axis = normalize(np.cross(z_axis, x_axis))
        camera_to_world = np.array([
            [x_axis[0], y_axis[0], view_dir[0], eye[0]],
            [x_axis[1], y_axis[1], view_dir[1], eye[1]],
            [x_axis[2], y_axis[2], view_dir[2], eye[2]],
            [0.0,       0.0,       0.0,        1.0]
        ], dtype=np.float32)
        glUniform2f(locations["u_resolution"], window_width, window_height)
        glUniform3f(locations["u_light_pos"], -2.0, 4.0, 1.0)
        glUniformMatrix4fv(locations["u_camera_to_world"], 1, GL_TRUE, camera_to_world)
        glUniform1f(locations["u_focal_length"], focal_length)
        glBindVertexArray(VAO)
        glDrawArrays(GL_TRIANGLES, 0, 6)
        glfw.swap_buffers(window)
        glfw.poll_events()

    # --- Terminate --- (no changes here)
    glDeleteVertexArrays(1, [VAO])
    glDeleteBuffers(1, [VBO])
    glDeleteProgram(shader_program)
    glfw.terminate()

if __name__ == "__main__":
    main()