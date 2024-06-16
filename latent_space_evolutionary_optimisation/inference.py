from latent_space_evolutionary_optimisation.main_func import nsga_ii


def main(fit_func, latent_space_size):
    population_size = 100
    chromosome_length = latent_space_size
    num_generations = 100
    result = nsga_ii(population_size, chromosome_length, num_generations, fit_func)
    return result
