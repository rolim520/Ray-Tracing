# -*- coding: utf-8 -*-
"""
================================================================================
Ray Tracing com Python e OpenGL
================================================================================

Descrição:
Este arquivo implementa um visualizador 3D simples usando ray tracing em tempo real.
A cena é renderizada em um quad que ocupa a tela inteira, e os cálculos de
interseção de raios e sombreamento são feitos no fragment shader (fragment_shader.glsl).
Este script Python é responsável por:
- Configurar uma janela usando GLFW.
- Compilar e vincular os shaders GLSL (vertex e fragment).
- Criar um quad de tela cheia para a renderização.
- Gerenciar a câmera e a entrada do usuário (teclado e mouse).
- Enviar os dados necessários (uniforms) para os shaders a cada quadro.

Link do Repositório: https://github.com/rolim520/Ray-Tracing

Pré-requisitos:
- Python 3.x
- Placa de vídeo com suporte a OpenGL 3.3+

Bibliotecas necessárias:
- PyOpenGL
- GLFW
- NumPy

# Como instalar as bibliotecas:
# pip install PyOpenGL PyOpenGL-accelerate glfw numpy
#
# Como executar o programa:
python main.py
# (Em alguns sistemas Linux/macOS, pode ser necessário usar python3 main.py)

Controles:
- W, A, S, D: Mover a câmera para frente, esquerda, trás e direita.
- Espaço: Mover a câmera para cima.
- Shift Esquerdo: Mover a câmera para baixo.
- Mouse: Controlar a orientação da câmera.
- Scroll do Mouse: Ajustar o zoom (distância focal).
- ESC: Capturar/Liberar o cursor do mouse.
- Clique na janela: Re-capturar o mouse se ele estiver liberado.

IMPORTANTE: Se o programa apresentar baixo desempenho ou fechar inesperadamente,
tente reduzir a resolução da janela. A performance é diretamente
impactada pelo número de pixels a serem renderizados.
Altere os valores de largura e altura na linha 'app = App(width, height)'
no final deste arquivo.
"""

import sys
import os
import glfw
from OpenGL.GL import *
import numpy as np
import ctypes

# --- CONSTANTES ---
# Define constantes para facilitar o ajuste de parâmetros da aplicação.
CAMERA_SPEED = 3.0          # Velocidade de movimento da câmera
MOUSE_SENSITIVITY = 0.1     # Sensibilidade do mouse para rotação da câmera
ZOOM_SPEED = 0.2            # Velocidade do zoom (ajuste da distância focal)
MIN_FOCAL_LENGTH = 0.5      # Distância focal mínima (zoom in)
MAX_FOCAL_LENGTH = 10.0     # Distância focal máxima (zoom out)

def read_shader_file(filename):
    """Lê o conteúdo de um arquivo de shader."""
    if not os.path.exists(filename):
        print(f"ERRO: Arquivo de shader não encontrado: {filename}")
        sys.exit()
    with open(filename, 'r') as f:
        return f.read()

def compile_shader(source, shader_type):
    """Compila um shader a partir do código fonte."""
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)
    # Verifica se a compilação foi bem-sucedida
    if not glGetShaderiv(shader, GL_COMPILE_STATUS):
        error = glGetShaderInfoLog(shader).decode()
        print(f"ERRO ao compilar shader: {error}")
        sys.exit()
    return shader

def normalize(v):
    """Normaliza um vetor NumPy (calcula o vetor unitário)."""
    norm = np.linalg.norm(v)
    if norm == 0:
       return v
    return v / norm

# --- Classe Principal da Aplicação ---
class App:
    def __init__(self, width, height):
        """Inicializa a aplicação, a janela GLFW e o estado da câmera."""
        # Dimensões da janela
        self.width = width
        self.height = height

        # Estado da câmera
        self.eye = np.array([3.0, 1.0, 1.5], dtype=np.float32)  # Posição inicial da câmera (olho)
        self.up_vector = np.array([0.0, 1.0, 0.0], dtype=np.float32) # Vetor 'up' do mundo
        self.focal_length = 2.0  # Distância focal inicial

        # Estado do mouse e orientação da câmera
        self.yaw = -90.0  # Rotação horizontal (guinada)
        self.pitch = 0.0  # Rotação vertical (inclinação)
        self.last_x = width / 2
        self.last_y = height / 2
        self.first_mouse = True  # Flag para o primeiro movimento do mouse
        self.mouse_captured = True # Flag para controlar se o mouse está capturado

        # Controle de tempo (para movimento independente de framerate)
        self.last_frame_time = 0.0
        self.delta_time = 0.0

        # Controle de FPS
        self.frame_count = 0
        self.last_fps_time = 0.0
        self.fps = 0

        # Definição da cena
        self.scene = [
            # Esferas
            {'type': 1, 'center': [0.0, 0.0, -0.6], 'radius': 1.0, 'color': [1.0, 1.0, 1.0], 'reflectivity': 0.1, 'transparency': 0.9, 'refractive_index': 1.5},
            {'type': 1, 'center': [-0.5, -0.5, -3.0], 'radius': 0.5, 'color': [0.2, 1.0, 0.2], 'reflectivity': 0.05, 'transparency': 0.0, 'refractive_index': 1.5},
            {'type': 1, 'center': [0.5, -0.5, -3.0], 'radius': 0.5, 'color': [0.2, 0.2, 1.0], 'reflectivity': 0.05, 'transparency': 0.0, 'refractive_index': 1.5},
            {'type': 1, 'center': [0.0, 0.366, -3.0], 'radius': 0.5, 'color': [1.0, 0.2, 0.2], 'reflectivity': 0.05, 'transparency': 0.0, 'refractive_index': 1.5},
            {'type': 2, 'center': [0.0, 1.2, -3.0], 'normal': [0.0, 1.0, 0.0], 'major_radius': 0.8, 'minor_radius': 0.2, 'color': [1.0, 0.8, 0.2], 'reflectivity': 0.4, 'transparency': 0.0, 'refractive_index': 1.0},
        ]

        # Inicializa a biblioteca GLFW
        if not glfw.init():
            raise Exception("Não foi possível inicializar o GLFW")
        # Define a versão do OpenGL (3.3 Core)
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)

        # Cria a janela
        self.window = glfw.create_window(self.width, self.height, "Ray Tracer com Python e GLFW", None, None)
        if not self.window:
            glfw.terminate()
            raise Exception("Não foi possível criar a janela GLFW")

        glfw.make_context_current(self.window)
        glfw.swap_interval(0) # Desativa o V-Sync para não limitar o FPS

        # Define as funções de callback para eventos
        glfw.set_window_user_pointer(self.window, self)
        glfw.set_framebuffer_size_callback(self.window, self.framebuffer_size_callback_proxy)
        glfw.set_key_callback(self.window, self.key_callback_proxy)
        glfw.set_cursor_pos_callback(self.window, self.cursor_pos_callback_proxy)
        glfw.set_scroll_callback(self.window, self.scroll_callback_proxy)
        glfw.set_mouse_button_callback(self.window, self.mouse_button_callback_proxy)

        # Inicia com o mouse capturado para controle da câmera
        self.capture_mouse(True)

    def run(self):
        """Inicia o loop principal da aplicação."""
        # Compila os shaders e configura os recursos da GPU
        self.setup_rendering()

        print("Executando o ray tracer... Use WASD, Espaço e Shift para mover.")
        print("Use o mouse para olhar ao redor. Use o scroll para dar zoom.")
        print("Pressione ESC para capturar/liberar o mouse.")
        print("Clique na janela para re-capturar o mouse.")
        print("Feche a janela no botão 'X' para sair.")

        # Loop principal de renderização
        while not glfw.window_should_close(self.window):
            self.update_timing()   # Atualiza o delta time
            self.process_input()   # Processa a entrada do usuário
            self.render()          # Renderiza a cena
            glfw.poll_events()     # Verifica por novos eventos

        self.cleanup() # Libera os recursos ao sair

    def update_timing(self):
        """Calcula o tempo decorrido desde o último quadro (delta time) e o FPS."""
        current_time = glfw.get_time()
        self.delta_time = current_time - self.last_frame_time
        self.last_frame_time = current_time

        # Calcula o FPS
        self.frame_count += 1
        if current_time - self.last_fps_time >= 1.0:
            self.fps = self.frame_count
            self.frame_count = 0
            self.last_fps_time = current_time

    def setup_rendering(self):
        """Configura o programa de shader e os buffers de vértices."""
        # Compila e linca os shaders
        vertex_shader = compile_shader(read_shader_file("vertex_shader.glsl"), GL_VERTEX_SHADER)
        fragment_shader = compile_shader(read_shader_file("fragment_shader.glsl"), GL_FRAGMENT_SHADER)
        self.shader_program = glCreateProgram()
        glAttachShader(self.shader_program, vertex_shader)
        glAttachShader(self.shader_program, fragment_shader)
        glLinkProgram(self.shader_program)
        if not glGetProgramiv(self.shader_program, GL_LINK_STATUS):
            raise Exception("Falha no linker: " + str(glGetProgramInfoLog(self.shader_program)))
        # Os shaders já foram lincados, então podemos deletá-los
        glDeleteShader(vertex_shader)
        glDeleteShader(fragment_shader)

        # Cria um quad que preenche a tela inteira
        # Dois triângulos formam um retângulo: (-1,1), (-1,-1), (1,-1) e (-1,1), (1,-1), (1,1)
        quad_vertices = np.array([-1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0], dtype=np.float32)
        self.VAO = glGenVertexArrays(1) # Vertex Array Object
        self.VBO = glGenBuffers(1)      # Vertex Buffer Object
        glBindVertexArray(self.VAO)
        glBindBuffer(GL_ARRAY_BUFFER, self.VBO)
        glBufferData(GL_ARRAY_BUFFER, quad_vertices.nbytes, quad_vertices, GL_STATIC_DRAW)
        # Define como o OpenGL deve interpretar os dados do vértice (layout 0: vec2)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * 4, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)

        # Obtém a localização das variáveis uniform dos shaders para posterior atualização
        self.locations = {
            "u_resolution": glGetUniformLocation(self.shader_program, "u_resolution"),
            "u_light_pos": glGetUniformLocation(self.shader_program, "u_light_pos"),
            "u_camera_to_world": glGetUniformLocation(self.shader_program, "u_camera_to_world"),
            "u_focal_length": glGetUniformLocation(self.shader_program, "u_focal_length"),
            "scene": []
        }
        for i in range(len(self.scene)):
            self.locations["scene"].append({
                "type": glGetUniformLocation(self.shader_program, f"scene[{i}].type"),
                "center": glGetUniformLocation(self.shader_program, f"scene[{i}].center"),
                "radius": glGetUniformLocation(self.shader_program, f"scene[{i}].radius"),
                "normal": glGetUniformLocation(self.shader_program, f"scene[{i}].normal"),
                "major_radius": glGetUniformLocation(self.shader_program, f"scene[{i}].major_radius"),
                "minor_radius": glGetUniformLocation(self.shader_program, f"scene[{i}].minor_radius"),
                "color": glGetUniformLocation(self.shader_program, f"scene[{i}].color"),
                "reflectivity": glGetUniformLocation(self.shader_program, f"scene[{i}].reflectivity"),
                "transparency": glGetUniformLocation(self.shader_program, f"scene[{i}].transparency"),
                "refractive_index": glGetUniformLocation(self.shader_program, f"scene[{i}].refractive_index"),
            })

    def render(self):
        """Função de renderização chamada a cada quadro."""
        # Atualiza o título da janela com o FPS
        glfw.set_window_title(self.window, f"Ray Tracer com Python e GLFW - FPS: {self.fps}")

        glClearColor(0.0, 0.0, 0.0, 1.0) # Cor de fundo preta
        glClear(GL_COLOR_BUFFER_BIT)
        glUseProgram(self.shader_program) # Ativa o programa de shader

        # --- Cálculo da Matriz da Câmera ---
        # Calcula o vetor de direção da visão a partir dos ângulos de yaw e pitch
        direction = np.empty(3, dtype=np.float32)
        direction[0] = np.cos(np.radians(self.yaw)) * np.cos(np.radians(self.pitch))
        direction[1] = np.sin(np.radians(self.pitch))
        direction[2] = np.sin(np.radians(self.yaw)) * np.cos(np.radians(self.pitch))
        view_dir = normalize(direction)

        # Cria um sistema de coordenadas ortonormal para a câmera (base da câmera)
        z_axis = -view_dir # O eixo Z da câmera aponta para trás da direção de visão
        x_axis = normalize(np.cross(self.up_vector, z_axis)) # Eixo X da câmera (direita)
        y_axis = normalize(np.cross(z_axis, x_axis)) # Eixo Y da câmera (cima)

        # Monta a matriz de transformação de "câmera para mundo"
        # Esta matriz transforma as coordenadas do espaço da câmera para o espaço do mundo
        # e é usada no shader para calcular a origem e direção dos raios.
        camera_to_world = np.array([
            [x_axis[0], y_axis[0], view_dir[0], self.eye[0]],
            [x_axis[1], y_axis[1], view_dir[1], self.eye[1]],
            [x_axis[2], y_axis[2], view_dir[2], self.eye[2]],
            [0.0,       0.0,       0.0,        1.0]
        ], dtype=np.float32)

        # Envia os dados (uniforms) para a GPU
        glUniform2f(self.locations["u_resolution"], self.width, self.height)
        glUniform3f(self.locations["u_light_pos"], -2.0, 4.0, 1.0) # Posição da luz fixa
        glUniformMatrix4fv(self.locations["u_camera_to_world"], 1, GL_TRUE, camera_to_world)
        glUniform1f(self.locations["u_focal_length"], self.focal_length)

        # Envia os dados da cena para a GPU
        for i, obj in enumerate(self.scene):
            loc = self.locations["scene"][i]
            glUniform1i(loc["type"], obj["type"])
            glUniform3fv(loc["center"], 1, obj["center"])
            glUniform3fv(loc["color"], 1, obj["color"])
            glUniform1f(loc["reflectivity"], obj["reflectivity"])
            glUniform1f(loc["transparency"], obj["transparency"])
            glUniform1f(loc["refractive_index"], obj["refractive_index"])
            if obj["type"] == 1: # Esfera
                glUniform1f(loc["radius"], obj["radius"])
            elif obj["type"] == 2: # Toro
                glUniform3fv(loc["normal"], 1, obj["normal"])
                glUniform1f(loc["major_radius"], obj["major_radius"])
                glUniform1f(loc["minor_radius"], obj["minor_radius"])

        # Desenha o quad de tela cheia
        glBindVertexArray(self.VAO)
        glDrawArrays(GL_TRIANGLES, 0, 6)

        # Troca os buffers de exibição (double buffering)
        glfw.swap_buffers(self.window)

    def process_input(self):
        """Processa as teclas pressionadas para mover a câmera."""
        # Não processa entrada se a janela não estiver em foco
        if not glfw.get_window_attrib(self.window, glfw.FOCUSED):
            return

        speed = CAMERA_SPEED * self.delta_time # Velocidade de movimento ajustada pelo delta time

        # Calcula o vetor de direção da visão atual
        direction = np.empty(3, dtype=np.float32)
        direction[0] = np.cos(np.radians(self.yaw)) * np.cos(np.radians(self.pitch))
        direction[1] = np.sin(np.radians(self.pitch))
        direction[2] = np.sin(np.radians(self.yaw)) * np.cos(np.radians(self.pitch))
        view_dir = normalize(direction)

        # Define os vetores de movimento baseados na direção da câmera
        # Movimento para frente/trás ignora a componente Y para não "voar" para cima/baixo
        parallel_forward = normalize(np.array([view_dir[0], 0, view_dir[2]]))
        right_vector = normalize(np.cross(parallel_forward, self.up_vector))

        # Atualiza a posição da câmera com base nas teclas pressionadas
        if glfw.get_key(self.window, glfw.KEY_W) == glfw.PRESS:
            self.eye += parallel_forward * speed
        if glfw.get_key(self.window, glfw.KEY_S) == glfw.PRESS:
            self.eye -= parallel_forward * speed
        if glfw.get_key(self.window, glfw.KEY_A) == glfw.PRESS:
            self.eye -= right_vector * speed
        if glfw.get_key(self.window, glfw.KEY_D) == glfw.PRESS:
            self.eye += right_vector * speed
        if glfw.get_key(self.window, glfw.KEY_SPACE) == glfw.PRESS:
            self.eye += self.up_vector * speed # Movimento vertical para cima
        if glfw.get_key(self.window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS:
            self.eye -= self.up_vector * speed # Movimento vertical para baixo

    def capture_mouse(self, should_capture):
        """Função auxiliar para capturar ou liberar o cursor do mouse."""
        self.mouse_captured = should_capture
        if should_capture:
            # Esconde o cursor e o mantém dentro da janela
            glfw.set_input_mode(self.window, glfw.CURSOR, glfw.CURSOR_DISABLED)
            self.first_mouse = True # Reseta para evitar um salto na câmera
        else:
            # Mostra o cursor normalmente
            glfw.set_input_mode(self.window, glfw.CURSOR, glfw.CURSOR_NORMAL)

    def cleanup(self):
        """Libera os recursos da OpenGL e termina o GLFW."""
        glDeleteVertexArrays(1, [self.VAO])
        glDeleteBuffers(1, [self.VBO])
        glDeleteProgram(self.shader_program)
        glfw.terminate()

    # --- Funções de Callback e Proxies ---
    # Métodos estáticos que atuam como "proxies". Eles obtêm a instância da
    # classe 'App' a partir da janela GLFW e chamam o método de instância real.
    # Isso é necessário porque as funções de callback do GLFW são funções C
    # que não conhecem o 'self' do Python.

    @staticmethod
    def framebuffer_size_callback_proxy(window, width, height):
        """Proxy para o callback de redimensionamento da janela."""
        app = glfw.get_window_user_pointer(window)
        app.framebuffer_size_callback(width, height)

    def framebuffer_size_callback(self, width, height):
        """Callback chamado quando a janela é redimensionada."""
        self.width, self.height = width, height
        glViewport(0, 0, width, height) # Atualiza o viewport do OpenGL

    @staticmethod
    def key_callback_proxy(window, key, scancode, action, mods):
        """Proxy para o callback de teclado."""
        app = glfw.get_window_user_pointer(window)
        app.key_callback(key, scancode, action, mods)

    def key_callback(self, key, scancode, action, mods):
        """Callback chamado quando uma tecla é pressionada."""
        # Pressionar ESC alterna a captura do mouse
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            self.capture_mouse(not self.mouse_captured)

    @staticmethod
    def mouse_button_callback_proxy(window, button, action, mods):
        """Proxy para o callback de clique do mouse."""
        app = glfw.get_window_user_pointer(window)
        app.mouse_button_callback(button, action, mods)

    def mouse_button_callback(self, button, action, mods):
        """Callback chamado quando um botão do mouse é pressionado."""
        # Se o mouse não estiver capturado, um clique o captura novamente.
        if not self.mouse_captured and action == glfw.PRESS:
            self.capture_mouse(True)

    @staticmethod
    def scroll_callback_proxy(window, x_offset, y_offset):
        """Proxy para o callback de scroll do mouse."""
        app = glfw.get_window_user_pointer(window)
        app.scroll_callback(x_offset, y_offset)

    def scroll_callback(self, x_offset, y_offset):
        """Callback chamado quando o scroll do mouse é usado (para zoom)."""
        self.focal_length += y_offset * ZOOM_SPEED
        # Limita a distância focal aos valores mínimo e máximo definidos
        self.focal_length = np.clip(self.focal_length, MIN_FOCAL_LENGTH, MAX_FOCAL_LENGTH)

    @staticmethod
    def cursor_pos_callback_proxy(window, xpos, ypos):
        """Proxy para o callback de posição do cursor."""
        app = glfw.get_window_user_pointer(window)
        app.cursor_pos_callback(xpos, ypos)

    def cursor_pos_callback(self, xpos, ypos):
        """Callback chamado quando o mouse se move."""
        # Ignora o movimento se o mouse não estiver capturado
        if not self.mouse_captured:
            return

        # No primeiro movimento do mouse após a captura, atualiza a última posição
        # para evitar um salto repentino da câmera.
        if self.first_mouse:
            self.last_x, self.last_y = xpos, ypos
            self.first_mouse = False

        # Calcula o deslocamento do mouse desde o último quadro
        xoffset = (xpos - self.last_x) * MOUSE_SENSITIVITY
        yoffset = (self.last_y - ypos) * MOUSE_SENSITIVITY # Invertido, pois as coordenadas Y vão de baixo para cima
        self.last_x, self.last_y = xpos, ypos

        # Atualiza os ângulos de yaw e pitch
        self.yaw += xoffset
        self.pitch += yoffset

        # Limita o pitch para evitar que a câmera vire de cabeça para baixo
        self.pitch = np.clip(self.pitch, -89.0, 89.0)

# --- Ponto de Entrada da Aplicação ---
if __name__ == "__main__":
    try:
        # Cria e executa a aplicação
        app = App(800, 800)
        app.run()
    except Exception as e:
        # Captura e exibe qualquer erro que possa ocorrer
        print(f"Ocorreu um erro: {e}")
        sys.exit()
