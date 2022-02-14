import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt

np.random.seed(0)

def infection(p, Sn, In):
    prob = 1-((1-p)**In)
    I_next = np.random.binomial(Sn, prob)
    S_next = Sn - I_next
    return I_next, S_next


def cost_containment(p):
    cost = (0.003/p)**9 -1
    return cost


def contagion_process(N, d, p, S_0, I_0):
    cost_hc = 0
    inf = np.zeros(d)
    sus = np.zeros(d)
    rem = np.zeros(d)
    todo = np.zeros((d, 2))
    I_k = I_0
    S_k = S_0
    R_k = N - S_0 - I_0
    for i in range(d):
        rem[i] = I_k
        todo[i, :] = infection(p, S_k, I_k)

        I_k = todo[i, 0]
        S_k = todo[i, 1]

        R_k = R_k + rem[i]
        total_pop = S_k + I_k + R_k


        # print(f'Susceptibles in day {i + 1} ::::: {todo[i, 1]}')
        # print(f'Infected in day {i + 1}:::: {todo[i, 0]}')
        # print(f'The removed in day {i + 1} are::: {rem[i]}')
        # print(f'Total of population always equal to  N:: {total_pop}')
        # print(f'The total recovered are::: {R_k}\n')

    # print(f'After the first {i+1} days: {todo}')
    # print(f'The vector removed are{rem}')
    cost_hc = R_k
    cost_cc = cost_containment(p)
    total_cost = cost_hc + cost_cc
    # print(f'The cost of ricovered is {cost_hc}')
    # print(f'The cost of containment is {cost_cc}')

    return total_cost

if __name__ == '__main__':
    simulations = 1000
    population = 1000
    days = 60
    probability_grid = np.array([0.003, 0.00275, 0.0025, 0.00225, 0.002, 0.00175,
                                0.0015, 0.00125, 0.001, 0.00075, 0.0005])
    initial_susceptibles = 999
    initial_infected = 1
    cost_in_probability = np.zeros((probability_grid.size, simulations))
    average_cost = np.zeros(probability_grid.size)
    for i in range(probability_grid.size):
        for j in range(simulations):
            cost_in_probability[i, j] = contagion_process(population, days,
                                                       probability_grid[i],
                                                       initial_susceptibles,
                                                       initial_infected)

        # print(f'With probability {probability_grid[i]} '
        #         f'the array of total cost is {cost_in_probability[i]}')
        av = np.average(cost_in_probability[i])
        average_cost[i] = av
        print(f'The average for probability {probability_grid[i]} is :{average_cost[i]}')

    plt.scatter(probability_grid, average_cost)
    plt.yscale('log')
    plt.show()

