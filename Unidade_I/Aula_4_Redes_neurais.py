# Introdução às redes neurais, neurônios biológicos e neurônios artificiais
"""
Redes Neurais
> Sistemas de computação com nos interconectados;
> Neurônios do cérebro humano;
> Quando se utilizam algorítmos, podem reconhecer padrões escondidos e correlações em dados;
> Podem agrupá-los e classificá-los;
> Aprendem continuamente;

Um nó é modelado conforme o comportamento de um neurônio humano.
Os nos são ativados quando há estímulos ou entradas suficientes. Essa ativação se espalha através da rede, criando uma
resposta ao estímulo (resultado).
As conexões entre esses neurônios artificiais agem como sinapses simples, fazendo os sinais serem transmitidos de
um para o outro.
"""

import matplotlib.pyplot as plt

# Define the neural network architecture
input_size = 2
hidden_size = 4
output_size = 1

# Create the figure and axes for plotting
fig, ax = plt.subplots()

# Plot the neural network architecture
ax.axis('off')
ax.set_aspect('equal')

# Draw the input layer
input_layer = plt.Circle((0.5, 0.7), 0.1, color='blue')
ax.add_patch(input_layer)
ax.text(0.5, 0.7, 'Input', ha='center', va='center')

# Draw the hidden layer
hidden_layer = plt.Circle((0.5, 0.5), 0.1, color='green')
ax.add_patch(hidden_layer)
ax.text(0.5, 0.5, 'Hidden', ha='center', va='center')

# Draw the output layer
output_layer = plt.Circle((0.5, 0.3), 0.1, color='red')
ax.add_patch(output_layer)
ax.text(0.5, 0.3, 'Output', ha='center', va='center')

# Draw the connections between layers
ax.arrow(0.5, 0.7, 0.5, 0, head_width=0.05, head_length=0.05, fc='black', ec='black')
ax.arrow(0.5, 0.5, 0.5, 0, head_width=0.05, head_length=0.05, fc='black', ec='black')

# Set the limits of the plot
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])

# Show the plot
plt.show()


# Classificador Perceptron
"""
Perceptron
> Unidade base de representação de redees neurais.
> Não transforma os dados para fazer classificação.
> Procura encontrar a melhor fronteira lineae para a separação de dados
"""

import matplotlib.pyplot as plt

# Define the perceptron architecture
input_size = 2
output_size = 1

# Create the figure and axes for plotting
fig, ax = plt.subplots()

# Plot the perceptron architecture
ax.axis('off')
ax.set_aspect('equal')

# Draw the input layer
input_layer = plt.Circle((0.5, 0.5), 0.1, color='blue')
ax.add_patch(input_layer)
ax.text(0.5, 0.5, 'Input', ha='center', va='center')

# Draw the output layer
output_layer = plt.Circle((0.8, 0.5), 0.1, color='red')
ax.add_patch(output_layer)
ax.text(0.8, 0.5, 'Output', ha='center', va='center')

# Draw the connection between layers
ax.arrow(0.5, 0.5, 0.3, 0, head_width=0.05, head_length=0.05, fc='black', ec='black')

# Set the limits of the plot
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])

# Show the plot
plt.show()


# Backpropagation
"""
> Algoritmo de aprendizagem supervisionada;
> O mais inportante algoritmo de redes neurais;

Duas fases principais:
> Passo a frente (forward pass): entradas serão passadas através da rede;
> Passo para trás: calcular o gradiente da função de perda da camada final. Utiliza-se esse gradiente para a aplicação
da regra da cadeia com a finalidade de atualizar os pesos da rede.
> Objetivo Geral: otimização e pesos para que a rede neural aprenda corretamente a mapear as entradas para as saídas.
"""

import numpy as np

# Define the sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define the derivative of the sigmoid function
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Define the backpropagation algorithm
def backpropagation(inputs, targets, learning_rate, num_epochs):
    # Initialize random weights and biases
    input_size = inputs.shape[1]
    hidden_size = 4
    output_size = targets.shape[1]
    W1 = np.random.randn(input_size, hidden_size)
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size)
    b2 = np.zeros((1, output_size))

    # Training loop
    for epoch in range(num_epochs):
        # Forward pass
        hidden_layer = sigmoid(np.dot(inputs, W1) + b1)
        output_layer = sigmoid(np.dot(hidden_layer, W2) + b2)

        # Backward pass
        output_error = targets - output_layer
        output_delta = output_error * sigmoid_derivative(output_layer)
        hidden_error = np.dot(output_delta, W2.T)
        hidden_delta = hidden_error * sigmoid_derivative(hidden_layer)

        # Update weights and biases
        W2 += learning_rate * np.dot(hidden_layer.T, output_delta)
        b2 += learning_rate * np.sum(output_delta, axis=0, keepdims=True)
        W1 += learning_rate * np.dot(inputs.T, hidden_delta)
        b1 += learning_rate * np.sum(hidden_delta, axis=0, keepdims=True)

    return W1, b1, W2, b2

# Example usage
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([[0], [1], [1], [0]])
learning_rate = 0.1
num_epochs = 10000

W1, b1, W2, b2 = backpropagation(inputs, targets, learning_rate, num_epochs)

# Print the final weights and biases
print("Final weights and biases:")
print("W1:", W1)
print("b1:", b1)
print("W2:", W2)
print("b2:", b2)

