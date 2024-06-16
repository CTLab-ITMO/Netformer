from latent_space_evolutionary_optimisation.population_creation import create_initial_population
from latent_space_evolutionary_optimisation.breed import breed_population
from latent_space_evolutionary_optimisation.sort_individuals import sort_by_crowding_distance
from latent_space_evolutionary_optimisation.ranked_individuals import get_ranked_individuals
from tqdm import tqdm

# Основная функция для запуска алгоритма NSGA-II
def nsga_ii(population_size, chromosome_length, num_generations, fit_func):
    population = create_initial_population(population_size, chromosome_length)

    for j in tqdm(range(num_generations)):
        new_population = []

        population += breed_population(population, population_size)
        for individual in population:
            individual.evaluate(fit_func)

        rank2individuals = get_ranked_individuals(population)

        i = 1
        while len(new_population) + len(rank2individuals[i]) < population_size:
            new_population += rank2individuals[i]
            i += 1

        rank2individuals_i = sort_by_crowding_distance(rank2individuals[i])
        new_population += rank2individuals_i[:population_size - len(new_population)]
        population = new_population

    return population