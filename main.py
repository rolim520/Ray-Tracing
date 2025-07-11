import sys
import glfw
from OpenGL.GL import *
import numpy as np

def read_shader_file(filename):
    with open(filename, 'r') as f:
        return f.read()


# --- Python Host Code (Using GLFW) ---

window_width = 1000
window_height = 800

def compile_shader(source, shader_type):
    """ Helper function to compile a shader. """
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)
    if not glGetShaderiv(shader, GL_COMPILE_STATUS):
        error = glGetShaderInfoLog(shader).decode()
        print(f"ERROR compiling shader: {error}")
        sys.exit()
    return shader

def framebuffer_size_callback(window, width, height):
    """ The function called whenever a window is resized """
    global window_width, window_height
    window_width = width
    window_height = height
    glViewport(0, 0, width, height)

def main():
    """ Main function to set up GLFW and run the main loop. """
    
    # --- 1. Initialize GLFW ---
    if not glfw.init():
        print("Could not initialize GLFW")
        return

    # --- 2. Create a window and its OpenGL context ---
    window = glfw.create_window(window_width, window_height, "Python GLFW Ray Tracer", None, None)
    if not window:
        glfw.terminate()
        return

    glfw.make_context_current(window)
    glfw.set_framebuffer_size_callback(window, framebuffer_size_callback)

    # --- 3. Compile and link shaders ---
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

    # --- 4. Set up vertex data for a screen-filling quad ---
    # We create a simple rectangle that covers the entire screen. The fragment
    # shader will then run on every pixel of this rectangle.
    quad_vertices = np.array([
        # positions   
        -1.0,  1.0,
        -1.0, -1.0,
         1.0, -1.0,
        
        -1.0,  1.0,
         1.0, -1.0,
         1.0,  1.0
    ], dtype=np.float32)

    VAO = glGenVertexArrays(1)
    VBO = glGenBuffers(1)
    glBindVertexArray(VAO)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, quad_vertices.nbytes, quad_vertices, GL_STATIC_DRAW)
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * 4, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glBindVertexArray(0)
    
    print("Running ray tracer... Press ESC or close window to exit.")
    
    # --- 5. Main Loop ---
    while not glfw.window_should_close(window):
        # Input
        if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
            glfw.set_window_should_close(window, True)
        
        # Rendering
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)

        glUseProgram(shader_program)

        # Pass uniforms to the shader
        glUniform2f(glGetUniformLocation(shader_program, "u_resolution"), window_width, window_height)
        # Set the desired side-view camera position from Python
        glUniform3f(glGetUniformLocation(shader_program, "u_camera_pos"), 5.0, 1.0, -2.0)
        glUniform3f(glGetUniformLocation(shader_program, "u_light_pos"), -2.0, 2.0, 1.0)
        
        # Draw the quad
        glBindVertexArray(VAO)
        glDrawArrays(GL_TRIANGLES, 0, 6)

        # Swap buffers and poll events
        glfw.swap_buffers(window)
        glfw.poll_events()

    # --- 6. Terminate ---
    glDeleteVertexArrays(1, [VAO])
    glDeleteBuffers(1, [VBO])
    glDeleteProgram(shader_program)
    glfw.terminate()

if __name__ == "__main__":
    main()