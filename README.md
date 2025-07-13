# Projeto de Ray Tracing em Tempo Real com Python e OpenGL

![Imagem 1](Imagens/imagem%201.png)

Este projeto é uma implementação de um renderizador 3D que utiliza a técnica de **Ray Tracing em tempo real**, executado inteiramente na GPU através de um Fragment Shader em GLSL. O script Python atua como um "host" para a aplicação, gerenciando a janela, a câmera e enviando os dados da cena para a GPU, mas toda a complexa lógica de renderização ocorre no shader.

## Destaques da Implementação

Este não é um projeto OpenGL comum que utiliza a pipeline de rasterização padrão. Em vez disso, ele simula como a luz realmente funciona, traçando o caminho de raios de luz a partir da câmera para a cena, pixel por pixel.

- **Ray Tracing no Fragment Shader:** Toda a lógica de traçado de raios, interseção com objetos e cálculo de cores é implementada do zero em GLSL. O Vertex Shader apenas desenha um retângulo do tamanho da tela, e o Fragment Shader faz todo o trabalho pesado.

- **Implementação Iterativa:** Para superar as limitações de recursão em GLSL, o traçado de raios para reflexões e refrações é feito de forma iterativa, utilizando uma pilha de raios gerenciada manualmente no shader.

- **Interseções Matemáticas Complexas:** O shader inclui resolvedores para equações quadráticas, cúbicas e quârticas para calcular com precisão a interseção de raios com diferentes formas geométricas, como esferas e **toros** (que exigem a resolução de uma equação quârtica).

- **Efeitos Ópticos Avançados:**
  - **Reflexão e Refração:** Objetos podem ser reflexivos ou transparentes.
  - **Efeito Fresnel:** A quantidade de luz refletida versus refratada muda realisticamente com base no ângulo de visão, simulando o efeito Fresnel.
  - **Sombras com Transparência:** As sombras não são apenas pretas. O shader calcula a atenuação da luz através de objetos transparentes, permitindo a criação de "sombras coloridas".

- **Cena Definida em Python:** A cena 3D, incluindo a posição, forma, cor e propriedades do material de cada objeto, é definida de forma simples em uma lista Python, que é então enviada para a GPU.

## Tecnologias Utilizadas

*   **Python 3**: Linguagem principal da aplicação host.
*   **PyOpenGL**: Biblioteca para utilizar as funções da API OpenGL em Python.
*   **GLFW**: Biblioteca para criar a janela, o contexto OpenGL e gerenciar entradas (teclado/mouse).
*   **NumPy**: Utilizada para cálculos matemáticos e manipulação de vetores/matrizes no lado do CPU.
*   **GLSL (OpenGL Shading Language)**: Linguagem utilizada para escrever os shaders que rodam na GPU.

## Como Executar

### Pré-requisitos

- Python 3.x
- Placa de vídeo com suporte a OpenGL 3.3 ou superior.

### Instalação de Dependências

Abra seu terminal (ou **Command Prompt/PowerShell** no Windows) e instale as bibliotecas necessárias com o seguinte comando:

```bash
pip install PyOpenGL glfw numpy
```

### Execução

Para rodar a aplicação, execute o script principal a partir do diretório do projeto:

```bash
python main.py
```
> **Nota:** Se o comando acima não funcionar, tente usar `python3` em vez de `python`. Isso é comum em sistemas Linux e macOS onde múltiplas versões do Python estão instaladas.


## Controles

*   **W, A, S, D**: Mover a câmera para frente, esquerda, trás e direita.
*   **Espaço**: Mover a câmera para cima.
*   **Shift Esquerdo**: Mover a câmera para baixo.
*   **Mouse**: Controlar a orientação (olhar) da câmera.
*   **Scroll do Mouse**: Ajustar o zoom (distância focal).
*   **ESC**: Capturar/Liberar o cursor do mouse para interagir com a janela.
*   **Clique na janela**: Re-capturar o mouse se ele estiver liberado.

## Links

**Repositório no GitHub:** [https://github.com/rolim520/Ray-Tracing](https://github.com/rolim520/Ray-Tracing)
