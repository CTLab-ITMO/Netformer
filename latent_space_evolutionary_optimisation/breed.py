import random
# Функция для выполнения операторов кроссовера и мутации
def breed_population(parents, population_size, childs_per_parent=2):
    offspring = []
    while len(offspring) < population_size * childs_per_parent:
        parent1 = random.choice(parents)
        parent2 = random.choice(parents)
        child = parent1.crossover(parent2)
        child.mutate()
        offspring.append(child)
    return offspring