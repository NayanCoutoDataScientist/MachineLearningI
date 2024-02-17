# Introdução aos Modelos de Seleção
"""
> Uma base de dados pode ser definida como uma coleção de objetos que podem ser eventos, observações ou registros.
Estes objetos de dados são caracterizados por valores em um conjunto de características pré-determinadas chamadas
de atributos.
"""

# Aplicação dos Modelos de Seleção
"""
> Uma das técnicas muito utilizadas para a seleção de características é a Determinação Automática de Relevância (DAR),
sendo baseada na ESTATÍSTICA BAYESIANA e que utiliza as densidades de probabilidades.
> Nesta técnica, é utilizada a teoria de Bayes para a inferência do conjunto ótimo de características que devem ser
consideradas para a base de dados de um determinado problema
"""

# Algoritmos de Seleção
"""
> No aprendizado de máquina, o ALGORITMO GENÉTICO (GA) é um método amplamente utilizado para a seleção das melhores
características de uma base de dados.
> O GA gera algumas soluções aleatórias possíveis (chamadas de população), que representam variáveis diferentes,
para então combinar as melhores soluções em um processo iterativo.

Funcionamento do algoritmo em algumas operações básicas:
> Seleção: escolha dos indivíduos mais aptos em uma geração.
> Crossover: criação de dois novos indiv´duos, com base nos genes de duas soluções. Esses indivíduos aparecerão para
a próxima geração.
> Mutação: altera aleatóriamente um gene no indivíduo (ou seja, se for 0 altera para 1; se for 1 altera para 0).
"""

# Estudo de Caso

import random

# Função de avaliação com peso
def evaluate_with_weight(individual, weights):
    # Avaliar o desempenho do indivíduo e atribuir uma pontuação ponderada
    score = sum(individual[i] * weights[i] for i in range(len(individual)))
    return score

# Função de seleção por torneio
def tournament_selection(population, k, weights):
    # Selecionar k indivíduos aleatórios da população
    tournament = random.sample(population, k)
    # Retornar o indivíduo com a maior pontuação ponderada
    return max(tournament, key=lambda x: evaluate_with_weight(x, weights))


# Função de crossover
def crossover(parent1, parent2):
    # Realizar o crossover entre os pais para gerar dois filhos
    crossover_point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

# Função de mutação
def mutation(individual, mutation_rate):
    # Aplicar mutação em um gene do indivíduo com uma determinada taxa de mutação
    mutated_individual = individual.copy()
    for i in range(len(mutated_individual)):
        if random.random() < mutation_rate:
            mutated_individual[i] = 1 - mutated_individual[i]
    return mutated_individual

# Parâmetros do algoritmo genético
population_size = 4
individual_length = 6
mutation_rate = 2
num_generations = 5

# Pesos para cada atributo
weights = [0.9, 0.5, 0.4, 0.3, 0.2, 0.1]

# Passo 1: Produzir uma população inicial
population = []
for _ in range(population_size):
    individual = [random.randint(0, 1) for _ in range(individual_length)]
    population.append(individual)

# Repetir pelas gerações
for generation in range(num_generations):
    print("Generation:", generation)

    # Passo 2: Atribuir uma pontuação aos membros
    scores = [evaluate_with_weight(individual, weights) for individual in population]

    # Passo 3: Selecionar um subconjunto por torneio e enviar para seleção
    selected_population = [tournament_selection(population, k=2, weights=weights) for _ in range(population_size)]

    # Passo 4: Aplicar mutações e crossover
    new_population = []
    for i in range(0, population_size, 2):
        parent1 = selected_population[i]
        parent2 = selected_population[i+1]
        child1, child2 = crossover(parent1, parent2)
        child1 = mutation(child1, mutation_rate)
        child2 = mutation(child2, mutation_rate)
        new_population.extend([child1, child2])

        # Atualizar a população
    population = new_population

    # Imprimir a população atual
    for individual in population:
        print(individual)

    print()

"""
Neste exemplo, a função evaluate_with_weight recebe um indivíduo e uma lista de pesos e calcula a pontuação ponderada do indivíduo.
A função tournament_selection agora recebe também a lista de pesos e seleciona o indivíduo com a maior pontuação ponderada.
A função crossover realiza o crossover entre dois pais para gerar dois filhos.
A função mutation aplica mutação em um gene do indivíduo com uma determinada taxa de mutação.
O algoritmo começa gerando uma população inicial de indivíduos binários.
Em seguida, ele itera pelas gerações, avaliando os indivíduos, selecionando um subconjunto por torneio,
aplicando mutações e crossover, e atualizando a população.
"""