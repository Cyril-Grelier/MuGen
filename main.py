
import os
import signal
from glob import glob

import matplotlib.pyplot as plt
from algo_gen.classes import Population

from src.IndividualDrumRNN import IndividualDrum
# from src.IndividualDrum import IndividualDrum

def show_stats(stats, title=None):
    fig, ax = plt.subplots(figsize=(8, 8))

    if title:
        plt.title(title)
    plt.xlabel("tours")
    plt.ylabel('fitness')
    plt.xlim(0, len(stats['max_fitness']))

    ax.plot(stats['max_fitness'], color='red', label='max')
    ax.plot(stats['min_fitness'], color='green', label='min')
    ax.plot(stats['mean_fitness'], color='blue', label='mean')
    plt.legend(title='fitness', loc='lower right')
    plt.show()


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
    files = glob('output/worse*.mid')
    if len(files) == 1:
        worse = float(files[0].split('_')[1].split('.')[0])
        worse_fit = pop.individuals[-1][1]
        if worse_fit < worse:
            os.remove(files[0])
            worse_indiv = pop.individuals[-1][0]
            worse_indiv.create_midi_file(file_name='worse_' + str(worse_fit))
    else:
        worse_indiv = pop.individuals[-1][0]
        worse_fit = pop.individuals[-1][1]
        worse_indiv.create_midi_file(file_name='worse_' + str(worse_fit))

    # if pop.nb_turns == 0:
    #     worst_indiv = pop.individuals[-1]
    # if worst_indiv[1] >= pop.individuals[-1][1]:
    #     worst_indiv = pop.individuals[-1]
    #     # print("WORST ", worst_indiv)

    # if pop.stats['max_fitness']
    # for p in pop.individuals:
    # print(p)
    print(f'\rturn, {pop.nb_turns}, max : {pop.stats["max_fitness"][-1]} ', end="")


def function_end(pop):
    print(pop.individuals)
    pop.individuals[0][0].create_midi_file()
    # pop.individuals[len(pop.individuals) - 1][0].create_midi_file()

    print("best: ", pop.individuals[0])
    pop.individuals[0][0].fitness()
    pop.individuals[len(pop.individuals) - 1][0].fitness()


parameters = {
        'configuration name'   : 'config1',
        'individual'           : IndividualDrum,
        'population size'      : 50,  # 100 200 500
        'chromosome size'      : 12,  # 5 10 50 100
        'termination_condition': final_condition,
        'function_each_turn'   : function_each_turn,
        'function_end'         : function_end,

        'nb turn max'          : 1000,
        'stop after no change' : 5000,
        'selection'            : ['select_best'],
        'proportion selection' : 4,
        'crossover'            : ['individual'],
        'proportion crossover' : 1,
        'mutation'             : ['individual'],
        'proportion mutation'  : 1,
        'insertion'            : 'fitness',  # 'age' 'fitness'
}
population = Population(parameters, description=True)


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
