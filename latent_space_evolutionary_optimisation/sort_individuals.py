# Функция для вычисления расстояния тесноты
def sort_by_crowding_distance(ranks_i_individuals):
    inds_and_dists = [{'ind': ind, 'dist': 0.0} for ind in ranks_i_individuals]

    num_objectives = len(ranks_i_individuals[0].fitness)
    for objective in range(num_objectives):
        inds_and_dists.sort(key=lambda x: x['ind'].fitness[objective])
        inds_and_dists[0]['dist'] = float('inf')
        inds_and_dists[-1]['dist'] = float('inf')

        min_fitness = inds_and_dists[0]['ind'].fitness[objective]
        max_fitness = inds_and_dists[-1]['ind'].fitness[objective]
        if max_fitness == min_fitness:
            continue

        for i in range(1, len(inds_and_dists)-1):
            inds_and_dists[i]['dist'] += (inds_and_dists[i+1]['ind'].fitness[objective] - inds_and_dists[i-1]['ind'].fitness[objective]) / (max_fitness - min_fitness)

    inds_and_dists.sort(key=lambda x: x['dist'])
    inds_and_dists = inds_and_dists[::-1]
    individuals = [ind['ind'] for ind in inds_and_dists]

    return individuals