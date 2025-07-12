import sys
import os
import glfw
from OpenGL.GL import *
import numpy as np
import ctypes

# --- CONSTANTS ---
# Define constants here for easy tweaking
CAMERA_SPEED = 3.0
MOUSE_SENSITIVITY = 0.1
ZOOM_SPEED = 0.2
MIN_FOCAL_LENGTH = 0.5
MAX_FOCAL_LENGTH = 10.0

def read_shader_file(filename):
    # ... (this helper function is perfect, no changes)
    if not os.path.exists(filename):
        print(f"ERROR: Shader file not found: {filename}")
        sys.exit()
    with open(filename, 'r') as f:
        return f.read()

def compile_shader(source, shader_type):
    # ... (this helper function is perfect, no changes)
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)
    if not glGetShaderiv(shader, GL_COMPILE_STATUS):
        error = glGetShaderInfoLog(shader).decode()
        print(f"ERROR compiling shader: {error}")
        sys.exit()
    return shader

def normalize(v):
    # ... (this helper function is perfect, no changes)
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

# --- Main Application Class ---
class App:
    def __init__(self, width, height):
        # Window dimensions
        self.width = width
        self.height = height

        # Camera state
        self.eye = np.array([3.0, 1.0, 1.5], dtype=np.float32)
        self.up_vector = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self.focal_length = 2.0
        
        # Mouse and orientation state
        self.yaw = -90.0
        self.pitch = 0.0
        self.last_x = width / 2
        self.last_y = height / 2
        self.first_mouse = True
        self.mouse_captured = True
        
        # Timing
        self.last_frame_time = 0.0
        self.delta_time = 0.0

        # Initialize GLFW and create a window
        if not glfw.init():
            raise Exception("GLFW could not be initialized")

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)
        
        self.window = glfw.create_window(self.width, self.height, "Python GLFW Ray Tracer", None, None)
        if not self.window:
            glfw.terminate()
            raise Exception("GLFW window could not be created")

        glfw.make_context_current(self.window)
        
        # Set callbacks
        glfw.set_window_user_pointer(self.window, self)
        glfw.set_framebuffer_size_callback(self.window, self.framebuffer_size_callback_proxy)
        glfw.set_key_callback(self.window, self.key_callback_proxy)
        glfw.set_cursor_pos_callback(self.window, self.cursor_pos_callback_proxy)
        glfw.set_scroll_callback(self.window, self.scroll_callback_proxy)
        glfw.set_mouse_button_callback(self.window, self.mouse_button_callback_proxy)

        # Start with mouse captured
        self.capture_mouse(True)

    def run(self):
        # Compile shaders and set up GPU resources
        self.setup_rendering()

        print("Running ray tracer... Use WASD, Space, and Shift to move.")
        print("Use the mouse to look around. Scroll to zoom.")
        print("Press ESC to toggle mouse capture.")
        print("Click in the window to re-capture the mouse.")
        print("Close the window with the 'X' button to exit.")

        # Main loop
        while not glfw.window_should_close(self.window):
            self.update_timing()
            self.process_input()
            self.render()
            glfw.poll_events()
            
        self.cleanup()

    def update_timing(self):
        current_time = glfw.get_time()
        self.delta_time = current_time - self.last_frame_time
        self.last_frame_time = current_time
        
    def setup_rendering(self):
        # Shader Program
        vertex_shader = compile_shader(read_shader_file("vertex_shader.glsl"), GL_VERTEX_SHADER)
        fragment_shader = compile_shader(read_shader_file("fragment_shader.glsl"), GL_FRAGMENT_SHADER)
        self.shader_program = glCreateProgram()
        glAttachShader(self.shader_program, vertex_shader)
        glAttachShader(self.shader_program, fragment_shader)
        glLinkProgram(self.shader_program)
        if not glGetProgramiv(self.shader_program, GL_LINK_STATUS):
            raise Exception("Linker failure: " + str(glGetProgramInfoLog(self.shader_program)))
        glDeleteShader(vertex_shader)
        glDeleteShader(fragment_shader)

        # Fullscreen Quad
        quad_vertices = np.array([-1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0], dtype=np.float32)
        self.VAO = glGenVertexArrays(1)
        self.VBO = glGenBuffers(1)
        glBindVertexArray(self.VAO)
        glBindBuffer(GL_ARRAY_BUFFER, self.VBO)
        glBufferData(GL_ARRAY_BUFFER, quad_vertices.nbytes, quad_vertices, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * 4, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)

        # Uniform locations
        self.locations = {
            "u_resolution": glGetUniformLocation(self.shader_program, "u_resolution"),
            "u_light_pos": glGetUniformLocation(self.shader_program, "u_light_pos"),
            "u_camera_to_world": glGetUniformLocation(self.shader_program, "u_camera_to_world"),
            "u_focal_length": glGetUniformLocation(self.shader_program, "u_focal_length")
        }

    def render(self):
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)
        glUseProgram(self.shader_program)

        # Calculate view direction and camera matrix
        direction = np.empty(3, dtype=np.float32)
        direction[0] = np.cos(np.radians(self.yaw)) * np.cos(np.radians(self.pitch))
        direction[1] = np.sin(np.radians(self.pitch))
        direction[2] = np.sin(np.radians(self.yaw)) * np.cos(np.radians(self.pitch))
        view_dir = normalize(direction)

        z_axis = -view_dir 
        x_axis = normalize(np.cross(self.up_vector, z_axis))
        y_axis = normalize(np.cross(z_axis, x_axis))
        
        camera_to_world = np.array([
            [x_axis[0], y_axis[0], view_dir[0], self.eye[0]],
            [x_axis[1], y_axis[1], view_dir[1], self.eye[1]],
            [x_axis[2], y_axis[2], view_dir[2], self.eye[2]],
            [0.0,       0.0,       0.0,        1.0]
        ], dtype=np.float32)

        # Send uniforms
        glUniform2f(self.locations["u_resolution"], self.width, self.height)
        glUniform3f(self.locations["u_light_pos"], -2.0, 4.0, 1.0)
        glUniformMatrix4fv(self.locations["u_camera_to_world"], 1, GL_TRUE, camera_to_world)
        glUniform1f(self.locations["u_focal_length"], self.focal_length)
        
        # Draw
        glBindVertexArray(self.VAO)
        glDrawArrays(GL_TRIANGLES, 0, 6)

        glfw.swap_buffers(self.window)

    def process_input(self):
        # Simplified: this function no longer needs to return anything
        # It directly modifies self.eye based on the current view direction
        if not glfw.get_window_attrib(self.window, glfw.FOCUSED):
            return

        speed = CAMERA_SPEED * self.delta_time

        # Calculate current view direction
        direction = np.empty(3, dtype=np.float32)
        direction[0] = np.cos(np.radians(self.yaw)) * np.cos(np.radians(self.pitch))
        direction[1] = np.sin(np.radians(self.pitch))
        direction[2] = np.sin(np.radians(self.yaw)) * np.cos(np.radians(self.pitch))
        view_dir = normalize(direction)
        
        # Use it to define movement vectors
        parallel_forward = normalize(np.array([view_dir[0], 0, view_dir[2]]))
        right_vector = normalize(np.cross(parallel_forward, self.up_vector))

        if glfw.get_key(self.window, glfw.KEY_W) == glfw.PRESS:
            self.eye += parallel_forward * speed
        if glfw.get_key(self.window, glfw.KEY_S) == glfw.PRESS:
            self.eye -= parallel_forward * speed
        if glfw.get_key(self.window, glfw.KEY_A) == glfw.PRESS:
            self.eye -= right_vector * speed
        if glfw.get_key(self.window, glfw.KEY_D) == glfw.PRESS:
            self.eye += right_vector * speed
        if glfw.get_key(self.window, glfw.KEY_SPACE) == glfw.PRESS:
            self.eye += self.up_vector * speed
        if glfw.get_key(self.window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS:
            self.eye -= self.up_vector * speed

    def capture_mouse(self, should_capture):
        """Helper function to handle mouse capture logic."""
        self.mouse_captured = should_capture
        if should_capture:
            glfw.set_input_mode(self.window, glfw.CURSOR, glfw.CURSOR_DISABLED)
            self.first_mouse = True
        else:
            glfw.set_input_mode(self.window, glfw.CURSOR, glfw.CURSOR_NORMAL)

    def cleanup(self):
        glDeleteVertexArrays(1, [self.VAO])
        glDeleteBuffers(1, [self.VBO])
        glDeleteProgram(self.shader_program)
        glfw.terminate()

    # --- Callback Proxies and Methods ---
    # These static methods retrieve the App instance and call the real method.
    @staticmethod
    def framebuffer_size_callback_proxy(window, width, height):
        app = glfw.get_window_user_pointer(window)
        app.framebuffer_size_callback(width, height)

    def framebuffer_size_callback(self, width, height):
        self.width, self.height = width, height
        glViewport(0, 0, width, height)

    @staticmethod
    def key_callback_proxy(window, key, scancode, action, mods):
        app = glfw.get_window_user_pointer(window)
        app.key_callback(key, scancode, action, mods)

    def key_callback(self, key, scancode, action, mods):
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            self.capture_mouse(not self.mouse_captured) # Toggle

    @staticmethod
    def mouse_button_callback_proxy(window, button, action, mods):
        app = glfw.get_window_user_pointer(window)
        app.mouse_button_callback(button, action, mods)
        
    def mouse_button_callback(self, button, action, mods):
        if not self.mouse_captured and action == glfw.PRESS:
            self.capture_mouse(True) # Re-capture

    @staticmethod
    def scroll_callback_proxy(window, x_offset, y_offset):
        app = glfw.get_window_user_pointer(window)
        app.scroll_callback(x_offset, y_offset)

    def scroll_callback(self, x_offset, y_offset):
        self.focal_length += y_offset * ZOOM_SPEED
        self.focal_length = np.clip(self.focal_length, MIN_FOCAL_LENGTH, MAX_FOCAL_LENGTH)

    @staticmethod
    def cursor_pos_callback_proxy(window, xpos, ypos):
        app = glfw.get_window_user_pointer(window)
        app.cursor_pos_callback(xpos, ypos)
        
    def cursor_pos_callback(self, xpos, ypos):
        if not self.mouse_captured:
            return

        if self.first_mouse:
            self.last_x, self.last_y = xpos, ypos
            self.first_mouse = False

        xoffset = (xpos - self.last_x) * MOUSE_SENSITIVITY
        yoffset = (self.last_y - ypos) * MOUSE_SENSITIVITY
        self.last_x, self.last_y = xpos, ypos
        
        self.yaw += xoffset
        self.pitch += yoffset
        self.pitch = np.clip(self.pitch, -89.0, 89.0)

if __name__ == "__main__":
    try:
        app = App(1000, 800)
        app.run()
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit()