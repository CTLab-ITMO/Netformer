import random
import torch

class Individual:
    def __init__(self, chromosome):
        self.chromosome = chromosome
        self.fitness = []

    def evaluate(self, fit_func):
        # Функция для вычисления значения целевой функции для особи
        self.fitness = fit_func(self.chromosome.unsqueeze(0).to(device)).to('cpu').tolist()[0]

    def crossover(self, other):
        # Оператор кроссовера для создания потомка
        child_chromosome = []
        for i in range(len(self.chromosome)):
            if random.random() < 0.5:
                child_chromosome.append(self.chromosome[i])
            else:
                child_chromosome.append(other.chromosome[i])
        child_chromosome = torch.FloatTensor(child_chromosome)
        return Individual(child_chromosome)

    def mutate(self):
        # Оператор мутации для изменения генов особи
        for i in range(len(self.chromosome)):
            if random.random() < 0.1:
                self.chromosome[i] = torch.distributions.Normal(0, 1).sample()