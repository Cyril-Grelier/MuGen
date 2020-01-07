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


best_file_path="/ressources/example_midi_file.mid"
best_fitness= -10000

def function_each_turn(pop):
    #if pop.stats['max_fitness']
    for p in pop.individuals:
        print(p)


def function_end(pop):
    print(f'fitness max : {pop.stats["max_fitness"][-1]}')

import copy

#svae = Vae()

parameters = {
    'configuration name': 'config1',
    'individual': IndividualDrum,

    #'vae' : vae,

    'population size': 10,  # 100 200 500
    'chromosome size': 12,  # 5 10 50 100

    'termination_condition': final_condition,

    'function_each_turn': function_each_turn,
    'function_end': function_end,

    'nb turn max': 50,
    'stop after no change': 5000000,  # int(config['nb turn max']*0.10),

    'selection':
        ['select_best'],
    #     ['adaptative',
    #      'UCB',
    #      [
    #          [0.25, 'select_random'],
    #          [0.25, 'select_best'],
    #          [0.25, 'select_tournament'],
    #          [0.25, 'select_wheel']
    #      ]],
    'proportion selection': 1,  # 0.04,  # 2 / population_size

    'crossover':
        ['mono-point'],
    #     ['adaptative',
    #      'UCB',
    #      [
    #          [0.25, 'mono-point'],
    #          [0.25, 'uniforme'],
    #      ],
    #      0.5],
    'proportion crossover': 0,

    'mutation':
    ['individual'],
    #     ['adaptative',
    #      'UCB',
    #      # 'fixed roulette wheel' 'adaptive roulette wheel' 'adaptive pursuit' 'UCB'
    #      [
    #          [0.25, '1-flip'],
    #          [0.25, '3-flip'],
    #          [0.25, '5-flip'],
    #          [0.25, 'bit-flip']
    #      ],
    #      0.05,  # pmin for adaptive roulette wheel and adaptive poursuite
    #      0.5,  # beta for adaptive poursuit
    #      ],
    'proportion mutation': 1,  # 0.1 0.2 0.5 0.8

    'insertion': 'fitness',  # 'age' 'fitness'
}

population = Population(parameters)
population.start()
show_stats(population.stats)

#from VAE.VAE import train

#train()