import torch
from latent_space_evolutionary_optimisation.individual import Individual

def create_initial_population(population_size, chromosome_length):
    population = []
    for _ in range(population_size):
        chromosome = torch.distributions.Normal(0, 1).sample([1, latent_dims]).to(device)[0]
        individual = Individual(chromosome)
        population.append(individual)
    return population