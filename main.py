import signal

from algo_gen.classes import Population
from algo_gen.tools.plot import show_stats

from src.IndividualDrumRNN import IndividualDrum


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
    # if pop.stats['max_fitness']
    # for p in pop.individuals:
    #     print(p)
    print(f'\rturn, {pop.nb_turns}, max : {pop.stats["max_fitness"][-1]} ', end="")



def function_end(pop):
    print()
    print(pop.individuals)
    print(f'fitness max : {max(pop.stats["max_fitness"])}')


parameters = {
    'configuration name': 'config1',
    'individual': IndividualDrum,
    'population size': 50,  # 100 200 500
    'chromosome size': 12,  # 5 10 50 100
    'termination_condition': final_condition,
    'function_each_turn': function_each_turn,
    'function_end': function_end,

    'nb turn max': 50,
    'stop after no change': 5000000,
    'selection': ['select_best'],
    'proportion selection': 0.2,
    'crossover': ['individual'],
    'proportion crossover': 1,
    'mutation': ['individual'],
    'proportion mutation': 1,
    'insertion': 'fitness',  # 'age' 'fitness'
}
population = Population(parameters)


def stop(signum, frame):
    print("The process have been terminated (singal : " + str(signum) + ")")
    print("Please wait for the end of this turn so the current data will be saved")
    print(frame)
    global population
    population.final_condition = lambda pop: False


def signal_handler():
    """
    Handle signal for SIGINT and SIGTERM when the MCTS is running
    :return: None
    """
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)


signal_handler()

population.start()
show_stats(population.stats)


# [(140, 9, 46), (161, 9, 43), (177, 9, 42), (187, 9, 41), (249, 9, 35), (255, 9, 34), (257, 9, 34), (261, 9, 33), (275, 9, 32), (283, 9, 31), (287, 9, 31), (293, 9, 30), (303, 9, 29), (307, 9, 29), (309, 9, 29), (310, 9, 29), (313, 9, 28), (315, 9, 28), (318, 9, 28), (321, 9, 27), (325, 9, 27), (328, 9, 27), (330, 9, 27), (336, 9, 26), (351, 9, 24), (357, 9, 24), (360, 9, 24), (362, 9, 23), (367, 9, 23), (378, 9, 22), (381, 9, 21), (383, 9, 21), (385, 9, 21), (386, 9, 21), (389, 9, 21), (393, 9, 20), (395, 9, 20), (401, 9, 19), (402, 9, 19), (409, 9, 19), (414, 9, 18), (419, 9, 18), (421, 9, 17), (424, 9, 17), (427, 9, 17), (429, 9, 17), (430, 9, 17), (432, 9, 16), (433, 9, 16), (436, 9, 16), (439, 9, 16), (440, 9, 16), (441, 9, 15), (444, 9, 15), (445, 9, 15), (446, 9, 15), (448, 9, 15), (449, 9, 15), (450, 9, 15), (460, 9, 14), (462, 9, 13), (464, 9, 13), (471, 9, 12), (472, 9, 12), (474, 9, 12), (475, 9, 12), (477, 9, 12), (478, 9, 12), (479, 9, 12), (482, 9, 11), (483, 9, 11), (484, 9, 11), (485, 9, 11), (487, 9, 11), (489, 9, 11), (491, 9, 10), (496, 9, 10), (497, 9, 10), (498, 9, 10), (501, 9, 9), (502, 9, 9), (505, 9, 9), (506, 9, 9), (513, 9, 8), (514, 9, 8), (515, 9, 8), (516, 9, 8), (520, 9, 8), (521, 9, 7), (522, 9, 7), (592, 9, 0), (594, 9, 0), (596, 9, 0), (591, 6, 0), (593, 5, 0), (595, 5, 0), (599, 5, 0), (598, 4, 0), (597, 3, 0), (600, 2, 0)]
