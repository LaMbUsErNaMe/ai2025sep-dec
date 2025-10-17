import numpy as np
import matplotlib.pyplot as plt

# Количество пунктов производства и городов
n = 5  # Пункты производства
k = 3  # Города

# Объемы производства и потребления
production = np.array([100, 150, 200, 250, 300])  # Возможности пунктов производства
demand = np.array([180, 220, 280])  # Потребности городов

# Транспортные расходы (случайные значения, можно заменить на реальные)
transport_costs = np.random.randint(10, 100, size=(n, k))

# Определение вероятностей
MUTATION_PROBABILITY = 0.5
CROSSOVER_PROBABILITY = 0.5

class Individual:
    def __init__(self, genome):
        self.genome = genome
        self.fitness = self.calculate_fitness()

    def calculate_fitness(self):
        total_cost = 0
        for i in range(n):  # Для каждого пункта производства
            for j in range(k):  # Для каждого города
                total_cost += self.genome[i, j] * transport_costs[i, j]

        # Штраф за превышение или нехватку продуктов
        total_supplied = np.sum(self.genome, axis=0)
        penalty = np.sum(np.abs(total_supplied - demand)) * 100  # Штраф 100 за единицу отклонения

        return total_cost + penalty


# Создаем случайного индивида (геном - это распределение продуктов между пунктами производства и городами)
def create_random_individual():
    genome = np.zeros((n, k))
    for i in range(n):
        remaining_production = production[i]
        for j in range(k):
            if remaining_production > 0:
                supply = np.random.randint(0, remaining_production + 1)
                genome[i, j] = supply
                remaining_production -= supply
    return Individual(genome)


# Выбор родителей
def select_parents(population):
    random_parents = np.random.choice(population, size=5, replace=False)
    random_parents = sorted(random_parents, key=lambda x: x.fitness)
    return random_parents[:2]


# Методы кроссовера:
def single_point_crossover(parent1, parent2):
    crossover_point = np.random.randint(1, parent1.genome.shape[0])
    child1_genome = np.concatenate((parent1.genome[:crossover_point], parent2.genome[crossover_point:]))
    child2_genome = np.concatenate((parent2.genome[:crossover_point], parent1.genome[crossover_point:]))
    return Individual(child1_genome), Individual(child2_genome)


def two_point_crossover(parent1, parent2):
    idx1, idx2 = sorted(np.random.randint(1, parent1.genome.size, size=2))
    flat_genome1 = parent1.genome.flatten()
    flat_genome2 = parent2.genome.flatten()
    child1_genome = np.concatenate((flat_genome1[:idx1], flat_genome2[idx1:idx2], flat_genome1[idx2:])).reshape(n, k)
    child2_genome = np.concatenate((flat_genome2[:idx1], flat_genome1[idx1:idx2], flat_genome2[idx2:])).reshape(n, k)
    return Individual(child1_genome), Individual(child2_genome)


def uniform_crossover(parent1, parent2):
    mask = np.random.randint(0, 2, size=(n, k))
    child1_genome = np.where(mask, parent1.genome, parent2.genome)
    child2_genome = np.where(mask, parent2.genome, parent1.genome)
    return Individual(child1_genome), Individual(child2_genome)


# Методы мутации:
def random_replacement(individual):
    idx1, idx2 = np.random.randint(n), np.random.randint(k)
    individual.genome[idx1, idx2] = np.random.randint(0, production[idx1] + 1)
    individual.fitness = individual.calculate_fitness()
    return individual


def swap_mutation(individual):
    idx1, idx2 = np.random.randint(n), np.random.randint(k)
    idx3, idx4 = np.random.randint(n), np.random.randint(k)
    individual.genome[idx1, idx2], individual.genome[idx3, idx4] = individual.genome[idx3, idx4], individual.genome[idx1, idx2]
    individual.fitness = individual.calculate_fitness()
    return individual


def inversion_mutation(individual):
    idx1, idx2 = sorted(np.random.randint(0, n * k, size=2))
    flat_genome = individual.genome.flatten()
    flat_genome[idx1:idx2 + 1] = flat_genome[idx1:idx2 + 1][::-1]
    individual.genome = flat_genome.reshape(n, k)
    individual.fitness = individual.calculate_fitness()
    return individual


# Функция эволюции популяции с учетом вероятностей мутации и кроссовера
def evolve_population(population, crossover_method, mutation_method):
    new_population = []
    while len(new_population) < len(population):
        parent1, parent2 = select_parents(population)

        # Кроссовер
        if np.random.rand() < CROSSOVER_PROBABILITY:
            child1, child2 = crossover_method(parent1, parent2)
        else:
            child1, child2 = parent1, parent2  # Без кроссовера

        # Мутация
        if np.random.rand() < MUTATION_PROBABILITY:
            child1 = mutation_method(child1)

        if np.random.rand() < MUTATION_PROBABILITY:
            child2 = mutation_method(child2)

        new_population.append(child1)
        new_population.append(child2)
    return new_population


# Функция для вывода результатов
def print_results(final_population, label):
    best_individual = min(final_population, key=lambda ind: ind.fitness)
    avg_fitness = np.mean([ind.fitness for ind in final_population])
    print(f"Результаты для {label}:")
    print(f"Лучший индивид: Геном = \n{best_individual.genome}")
    print(f"Лучший фитнес: {best_individual.fitness}")
    print(f"Средний фитнес в популяции: {avg_fitness}")
    print("-" * 50)


# Функция для тестирования одной комбинации и вывода графика
def run_experiment(crossover_method, mutation_method, label, color):
    np.random.seed(0)  # Для воспроизводимости
    initial_population = [create_random_individual() for _ in range(10)]
    best_fitness_over_time = []

    for generation in range(50):  # 50 поколений
        initial_population = evolve_population(initial_population, crossover_method, mutation_method)
        best_fitness = min(ind.fitness for ind in initial_population)
        best_fitness_over_time.append(best_fitness)

    plt.plot(best_fitness_over_time, label=label, color=color)

    # Вывод результатов после завершения эволюции
    print_results(initial_population, label)


# Запуск всех 9 комбинаций с выводом результатов в консоль:
plt.figure(figsize=(10, 8))

# 1. Random Replacement + разные кроссоверы
run_experiment(single_point_crossover, random_replacement, "Random Replacement + Single Point", 'red')
run_experiment(two_point_crossover, random_replacement, "Random Replacement + Two Point", 'green')
run_experiment(uniform_crossover, random_replacement, "Random Replacement + Uniform", 'blue')

# 2. Swap Mutation + разные кроссоверы
run_experiment(single_point_crossover, swap_mutation, "Swap Mutation + Single Point", 'orange')
run_experiment(two_point_crossover, swap_mutation, "Swap Mutation + Two Point", 'purple')
run_experiment(uniform_crossover, swap_mutation, "Swap Mutation + Uniform", 'brown')

# 3. Inversion Mutation + разные кроссоверы
run_experiment(single_point_crossover, inversion_mutation, "Inversion Mutation + Single Point", 'pink')
run_experiment(two_point_crossover, inversion_mutation, "Inversion Mutation + Two Point", 'cyan')
run_experiment(uniform_crossover, inversion_mutation, "Inversion Mutation + Uniform", 'gray')

# Настройка графиков
plt.title('Эволюция фитнеса для разных комбинаций мутаций и кроссоверов')
plt.xlabel('Поколение')
plt.ylabel('Лучший фитнес')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()
