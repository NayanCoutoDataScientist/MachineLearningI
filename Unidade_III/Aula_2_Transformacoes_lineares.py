# Introdução às transformações lineares
"""
TRANSFORMAÇÕES LINEARES PARA MACHINE LEARNING

O Que é a Algebra Linear?
> A algeba linear é um campo da matemática universalmente considerado pré-requisito para uma compreensão mais
profunda do aprendizado de máquina.
> A álgebra linear trata de combinações lineares. Isto é, usando aritmética em colunas de números chamadas vetores
e arranjos de números chamados matrizes, para criar colunas e arranjos de números. Álgebra linear é o estudo de linhas
e planos, espaços vetoriais e mapeamentos necessários para transformações lineares.

O que são transformações lineares?
> Uma transformação linear é uma função de um espaço vetorial para outro que respeita a estrutura subjacente (linear)
de cada espaço vetorial. Uma transformação linear também é conhecida como operador ou mapa linear.

    Transf: \R² → \R³
    (x, y) → (x, 2y, x+y)
"""

# Métodos de Transformações Lineares
"""
Transformações Lineares são funções que operam entre espaços vetoriais, preservando as operações de adição de vetores e
multiplicação de vetores por escalares. Vamos explorar os principais conceitos e métodos relacionados a
transformações lineares:

> Definição:
    Uma função (T: V /to W), onde (V) e (W) são espaços vetoriais, é considerada uma transformação linear se ela
    satisfaz as seguintes propriedades:
        (T(x + y) = T(x) + T(y)) para quaisquer vetores (x) e (y) em (V).
        (T(kx) = kT(x)) para todo vetor (x) em (V) e escalar (k).
> Exemplos:
    Função 1: (T: \mathbb{R}^2 /to \mathbb{R}), dada por (T(x, y) = x + y).
        Verificamos que a propriedade vale para essa função, tornando-a uma transformação linear de (\mathbb{R}^2) em (\mathbb{R}).
    Função 2: (T: \mathbb{R} /to \mathbb{R}), dada por (T(x) = 10x).
        Também verificamos que essa função é uma transformação linear de (\mathbb{R}) em (\mathbb{R}).
    Propriedade Importante:
    Se (T: V /to W) é uma transformação linear, então (T(0) = 0). Isso significa que se (T(0) /neq 0), a função não é linear.
> Transformação Linear Injetora:
    Uma transformação linear (T: V /to W) é injetora se, para quaisquer vetores (x) e (y) em (V), se (x /neq y), então (T(x) /neq T(y)).
> Sobrejetora:
    Neste caso uma transformação linear é sobrejetora quando os valores são exatamente iguais ao contradomínio.
> Bijetora: Quando nós temos uma transformação linear tanto intejora quanto sobrejetora ao mesmo tempo.
"""

# Aplicações de Transformações Lineares
"""
> A base dos sistemas de aprendizado de máquina e aprendizado profundo é totalmente baseada nos princípios e conceitos
matemáticos. É imperativo compreender os fundamentos dos princípios matemáticos.

Aplicações
> Estas são algumas das áreas da álgebra linear que usamos no aprendizado de máquina(ML) e aprendizado profundo:
    - Vetor e matriz.
    - Sistema de equações lineares.
    - Espaço vetorial.

> Especificamente com relação as transformações lineares, costumam ser usadas em aplicações de aprendizado de máquina.
Eles são úteis na modelagem de animação 2D e 3D, onde o tamanho e a forma de um objeto precisam ser transformados
de um ângulo de visão para o outro. Um objeto pode ser girado e escalado dentro de um espaço usando um tipo de
transformação linear conhecida como transformação geométrica, bem como aplicando matrizes de transformação.
"""

import numpy as np
import matplotlib.pyplot as plt

# Função para plotar os vetores no gráfico
def plot_vectors(vectors, colors):
    for vector, color in zip(vectors, colors):
        plt.quiver(0, 0, vector[0], vector[1], angles='xy', scale_units='xy', scale=1, color=color)
        plt.xlim(-10, 10)
        plt.ylim(-10, 10)

# Vetores de exemplo
vectors = np.array([[2, 2], [-3, 1], [1, -3]])
colors = ['r', 'g', 'b']

# Transformação: Rotação de 45 graus no sentido anti-horário
rotation_matrix = np.array([[np.cos(np.pi/4), -np.sin(np.pi/4)], [np.sin(np.pi/4), np.cos(np.pi/4)]])
rotated_vectors = np.dot(vectors, rotation_matrix)

# Transformação: Escala por um fator de 2
scale_factor = 2
scaled_vectors = vectors * scale_factor

# Transformação: Reflexão em relação ao eixo x
reflection_matrix = np.array([[1, 0], [0, -1]])
reflected_vectors = np.dot(vectors, reflection_matrix)

# Plotar os vetores originais
plt.subplot(2, 2, 1)
plt.title('Vetores Originais')
plot_vectors(vectors, colors)

# Plotar os vetores rotacionados
plt.subplot(2, 2, 2)
plt.title('Vetores Rotacionados')
plot_vectors(rotated_vectors, colors)

# Plotar os vetores escalados
plt.subplot(2, 2, 3)
plt.title('Vetores Escalados')
plot_vectors(scaled_vectors, colors)

# Plotar os vetores refletidos
plt.subplot(2, 2, 4)
plt.title('Vetores Refletidos')
plot_vectors(reflected_vectors, colors)

plt.tight_layout()
plt.show()