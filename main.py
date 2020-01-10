from src.IndividualDrum import IndividualDrum
from algo_gen.classes import Population
from algo_gen.tools.plot import show_stats

from src.VAE_orig import Vae


def final_condition(pop):
    if pop.nb_turns >= pop.parameters['stop after no change']:
        last_max = pop.stats['max_fitness'][-pop.parameters['stop after no change']:]
        # last_min = self.stats['min_fitness'][-self.parameters['stop after no change']:]
        max_change = not all(x >= y for x, y in zip(last_max, last_max[1:]))
        # min_change = not all(x >= y for x, y in zip(last_min, last_min[1:]))
        return (not (pop.nb_turns == pop.parameters['nb turn max'])) and max_change
        # and not (not max_change and not min_change)
    else:
        return not (pop.nb_turns == pop.parameters['nb turn max'])




def function_each_turn(pop):
    #print(pop.stats['max_fitness'])
    for p in pop.individuals:
        print(p)


def function_end(pop):
    print(f'fitness max : {pop.stats["max_fitness"][-1]}')

import copy

#svae = Vae()

parameters = {
    'configuration name': 'config1',
    'individual': IndividualDrum,
    'population size': 2,  # 100 200 500
    'chromosome size': 12,  # 5 10 50 100
    'termination_condition': final_condition,
    'function_each_turn': function_each_turn,
    'function_end': function_end,
    'nb turn max': 500,
    'stop after no change': 5000000,
    'selection': ['select_best'],
    'proportion selection': 1,
    'crossover': ['individual'],
    'proportion crossover': 1,
    'mutation': ['individual'],
    'proportion mutation': 1,
    'insertion': 'fitness',  # 'age' 'fitness'
}
population = Population(parameters)
population.start()
show_stats(population.stats)

#from VAE.VAE import train

#train()