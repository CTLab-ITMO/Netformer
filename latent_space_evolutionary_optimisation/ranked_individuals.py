import numpy as np

def get_ranked_individuals(population):
    rank2indexes = {}
    dominated_solutions = [[] for _ in range(len(population))]
    dominated_by_counters = [0] * len(population)

    rank2indexes[1] = []
    for p_idx, p_ind in enumerate(population):
        for q_idx, q_ind in enumerate(population):
            if p_idx == q_idx:
                continue

            p_fit = np.array(p_ind.fitness)
            q_fit = np.array(q_ind.fitness)
            is_q_dominated = np.all(p_fit >= q_fit) & np.any(p_fit > q_fit)
            is_p_dominated = np.all(q_fit >= p_fit) & np.any(q_fit > p_fit)
            if is_q_dominated:
                dominated_solutions[p_idx].append(q_idx)
            if is_p_dominated:
                dominated_by_counters[p_idx] += 1

        if dominated_by_counters[p_idx] == 0:
            rank2indexes[1].append(p_idx)

    i = 1
    while len(rank2indexes[i]) != 0:
        rank2indexes[i+1] = []

        for p_idx in rank2indexes[i]:
            for q_idx in dominated_solutions[p_idx]:
                dominated_by_counters[q_idx] -= 1
                if dominated_by_counters[q_idx] == 0:
                    rank2indexes[i+1].append(q_idx)
        i += 1
    del rank2indexes[i]

    rank2individuals = {}
    for rank, individual_indexes in rank2indexes.items():
        rank2individuals[rank] = [population[idx] for idx in individual_indexes]

    return rank2individuals