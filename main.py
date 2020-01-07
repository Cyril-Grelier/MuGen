# from individuals.IndividualDrum import IndividualDrum
# from algo_gen.classes import Population
# from algo_gen.tools.plot import show_stats
#
#
# def final_condition(pop):
#     if pop.nb_turns >= pop.parameters['stop after no change']:
#         last_max = pop.stats['max_fitness'][-pop.parameters['stop after no change']:]
#         # last_min = self.stats['min_fitness'][-self.parameters['stop after no change']:]
#         max_change = not all(x >= y for x, y in zip(last_max, last_max[1:]))
#         # min_change = not all(x >= y for x, y in zip(last_min, last_min[1:]))
#         return (not (pop.nb_turns == pop.parameters['nb turn max'])) and max_change
#         # and not (not max_change and not min_change)
#     else:
#         return not (pop.nb_turns == pop.parameters['nb turn max'])
#
#
# def function_each_turn(pop):
#     print(pop.nb_turns)
#
#
# def function_end(pop):
#     print(f'fitness max : {pop.stats["max_fitness"][-1]}')
#
#
# parameters = {
#     'configuration name': 'config1',
#     'individual': IndividualDrum,
#
#     'population size': 10,  # 100 200 500
#     'chromosome size': 12,  # 5 10 50 100
#
#     'termination_condition': final_condition,
#
#     'function_each_turn': function_each_turn,
#     'function_end': function_end,
#
#     'nb turn max': 10,
#     'stop after no change': 5000000,  # int(config['nb turn max']*0.10),
#
#     'selection':
#         ['select_best'],
#     #     ['adaptative',
#     #      'UCB',
#     #      [
#     #          [0.25, 'select_random'],
#     #          [0.25, 'select_best'],
#     #          [0.25, 'select_tournament'],
#     #          [0.25, 'select_wheel']
#     #      ]],
#     'proportion selection': 1,  # 0.04,  # 2 / population_size
#
#     'crossover':
#         ['mono-point'],
#     #     ['adaptative',
#     #      'UCB',
#     #      [
#     #          [0.25, 'mono-point'],
#     #          [0.25, 'uniforme'],
#     #      ],
#     #      0.5],
#     'proportion crossover': 0,
#
#     'mutation':
#     # ['3-flip'],
#         ['adaptative',
#          'UCB',
#          # 'fixed roulette wheel' 'adaptive roulette wheel' 'adaptive pursuit' 'UCB'
#          [
#              [0.25, '1-flip'],
#              [0.25, '3-flip'],
#              [0.25, '5-flip'],
#              [0.25, 'bit-flip']
#          ],
#          0.05,  # pmin for adaptive roulette wheel and adaptive poursuite
#          0.5,  # beta for adaptive poursuit
#          ],
#     'proportion mutation': 1,  # 0.1 0.2 0.5 0.8
#
#     'insertion': 'fitness',  # 'age' 'fitness'
# }
# population = Population(parameters)
# population.start()
# show_stats(population.stats)

from VAE.VAE import train

train()