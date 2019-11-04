import copy
import random
import statistics

from gen_algo.tools.tools import get_import


class Population:

    def __init__(self, parameters):
        self.individuals = []
        self.parameters = parameters
        self.stats = {
            'max_fitness': [], 'min_fitness': [], 'mean_fitness': [], 'fitness_diversity': [], "total_fitness": [],
            'diversity': [], 'max_age': [], 'mean_age': [], 'parameters': parameters, 'time': 0.0, 'utility': []
        }
        self.selected = []
        self.crossed = []
        self.mutated = []
        self.nb_turns = 0
        self.individual_class = get_import(parameters, 'individual')

        individuals = []
        for _ in range(self.parameters['population size']):
            individuals.append(self.individual_class(parameters))

        fits = [i.fitness() for i in individuals]
        self.individuals = [(i, f, 0) for i, f in sorted(zip(individuals, fits), key=lambda z: z[1])]

        self.nb_select = int(len(self.individuals) * self.parameters['proportion selection'])
        description = f'Population parameters :' \
                      f'\n\tIndividuals : {self.individual_class}' \
                      f'\n\tSize of an individual : {self.parameters["chromosome size"]}' \
                      f'\n\tSize of the population : {self.parameters["population size"]}' \
                      f'\n\tNumber of individuals selected each turns : {self.nb_select}' \
                      f'\n\tSelection : {self.parameters["selection"]}' \
                      f'\n\tCrossover : {self.parameters["crossover"]} ({self.parameters["proportion crossover"] * 100}%)' \
                      f'\n\tMutation : {self.parameters["mutation"]} ({self.parameters["proportion mutation"] * 100}%)' \
                      f'\n\tInsertion : {self.parameters["insertion"]}'
        print(description)

    def sort_individuals_fitness(self):
        self.individuals.sort(key=lambda i: i[1], reverse=True)

    def sort_individuals_age(self):
        self.individuals.sort(key=lambda i: i[2], reverse=False)

    def population_get_older(self):
        self.nb_turns += 1
        self.individuals = [(i, f, (a + 1)) for i, f, a in self.individuals]

    def __repr__(self):
        s = f"Population size : {len(self.individuals)}"
        for i in self.individuals:
            s += "\n" + str(i)
        return s

    def final_condition(self):
        if self.nb_turns >= self.parameters['stop after no change']:
            last_max = self.stats['max_fitness'][-self.parameters['stop after no change']:]
            last_min = self.stats['min_fitness'][-self.parameters['stop after no change']:]
            max_change = not all(x >= y for x, y in zip(last_max, last_max[1:]))
            min_change = not all(x >= y for x, y in zip(last_min, last_min[1:]))
            return (not (self.nb_turns == self.parameters['nb turn max'])) and not (not max_change and not min_change)
        else:
            return not (self.nb_turns == self.parameters['nb turn max'])

    def start(self):
        self.statistic()
        # TODO : remove this
        if self.parameters['individual'] == ['gen_algo.individuals.music', 'IndividualMusic']:
            from gen_algo.tools.midi_utils import convert_to_midi
            for i, indiv in enumerate(self.individuals):
                convert_to_midi(indiv[0], str(i) + "ORI" + ".mid")
        while self.final_condition():
            self.population_get_older()
            self.sort_individuals_fitness()
            self.turn()
            self.statistic()
        # TODO : remove this
        if self.parameters['individual'] == ['gen_algo.individuals.music', 'IndividualMusic']:
            from gen_algo.tools.midi_utils import convert_to_midi
            for i, indiv in enumerate(self.individuals):
                convert_to_midi(indiv[0], str(i) + ".mid")
                # play_midi_file(str(i) + ".mid")
        return self.termination()

    def turn(self):
        self.selection()
        self.crossover()
        self.mutation()
        self.insertion()
        # from pprint import pprint
        # pprint(self.stats['utility'])

    def termination(self):
        return self.stats

    ###############################################################
    #                       Selection                             #
    ###############################################################

    def selection(self):
        """
        chose the type of selection to use and select individuals
        :return:
        """
        self.selected = []
        switch = {
            'select_random': self.select_random,
            'select_best': self.select_best,
            'select_tournament': self.select_tournament,
            'select_wheel': self.select_wheel,
            'adaptative': self.select_adaptative
        }
        switch[self.parameters['selection'][0]]()

    def select_adaptative(self):
        switch = {
            'fixed roulette wheel': 'coucou',
            'adaptive roulette wheel': 'coucou',
            'adaptive pursuit': 'coucou',
            'UCB': 'coucou',
            'DMAB': 'coucou',
        }

    def select_random(self):
        selected = random.sample(self.individuals, self.nb_select)
        self.stats['utility'].append([[f for _, f, _ in selected]])
        self.selected = copy.deepcopy([i for i, _, _ in selected])

    def select_best(self):
        selected = self.individuals[0:self.nb_select]
        self.stats['utility'].append([[f for _, f, _ in selected]])
        self.selected = copy.deepcopy([i for i, _, _ in selected])

    def select_tournament(self):
        nb_winners = self.parameters['selection'][1]
        nb_players = self.parameters['selection'][2]
        self.stats['utility'].append([[]])
        while len(self.selected) < self.nb_select:
            selected = random.sample(self.individuals, nb_players)
            selected.sort(key=lambda i: i[1], reverse=True)
            selected = selected[0:nb_winners]
            self.stats['utility'][-1][0].extend([f for _, f, _ in selected])
            self.selected.extend(copy.deepcopy([i for i, _, _ in selected]))
        self.selected = self.selected[0:self.nb_select]

    def select_wheel(self):
        individuals, fits, _ = list(zip(*self.individuals))
        total = sum(fits)
        wheel = [1] * len(fits) if total == 0 else [f / total for f in fits]
        probabilities = [sum(wheel[:i + 1]) for i in range(len(wheel))]
        self.stats['utility'].append([[]])
        for n in range(self.nb_select):
            r = random.random()
            for (i, individual) in enumerate(individuals):
                if r <= probabilities[i]:
                    self.stats['utility'][-1][0].append(individual.fitness())
                    self.selected.append(copy.deepcopy(individual))
                    break

    ###############################################################
    #                       Crossover                             #
    ###############################################################

    def crossover(self):
        self.crossed = []
        if self.parameters['proportion crossover'] == 0:
            self.crossed = self.selected
        else:
            switch = {
                'mono-point': self.crossover_monopoint,
                'uniforme': self.crossover_uniforme,
            }
            for i in range(0, len(self.selected), 2):
                if random.random() <= self.parameters['proportion crossover']:
                    if len(self.selected) <= i + 1:
                        rand = random.choice(self.selected[0:-1])
                        first_child, second_child = switch[self.parameters['crossover']](self.selected[-1], rand)
                    else:
                        first_child, second_child = switch[self.parameters['crossover']](self.selected[i],
                                                                                         self.selected[i + 1])
                else:
                    first_child = self.individual_class(self.parameters)
                    second_child = self.individual_class(self.parameters)
                    first_child.sequence = copy.deepcopy(self.selected[i][::])
                    # the second child will be a copy of a random parents from the selected parents if the number
                    # of parents is odd
                    sc = random.randrange(len(self.selected[0:-1])) if len(self.selected) <= i + 1 else i + 1
                    second_child.sequence = copy.deepcopy(self.selected[sc].sequence)
                self.stats['utility'][-1].append([first_child.fitness(), second_child.fitness()])
                self.crossed.extend([first_child, second_child])

    def crossover_monopoint(self, i1, i2):
        first_child = self.individual_class(self.parameters)
        second_child = self.individual_class(self.parameters)
        rand = random.randint(1, len(i1.sequence))
        first_child[::] = i1[0:rand] + i2[rand:]
        second_child[::] = i2[0:rand] + i1[rand:]
        return first_child, second_child

    def crossover_uniforme(self, i1, i2):
        first_child = self.individual_class(self.parameters)
        second_child = self.individual_class(self.parameters)
        for i in range(self.parameters['chromosome size']):
            first_child[i], second_child[i] = (i1[i], i2[i]) if random.random() <= 0.5 else (i2[i], i1[i])
        return first_child, second_child

    ###############################################################
    #                       Mutation                              #
    ###############################################################

    def mutation(self):
        self.mutated = []
        switch = {
            'n-flip': self.mutation_nfip,
            'bit-flip': self.mutation_bitfip,
        }
        for i in self.crossed:
            if random.random() <= self.parameters['proportion mutation']:
                switch[self.parameters['mutation'][0]](i)
            self.mutated.append(i)

    def mutation_nfip(self, indiv):
        for i in random.sample(range(len(indiv.sequence)), self.parameters['mutation'][1]):
            indiv.sequence[i].bit = 1 - indiv.sequence[i].bit

    def mutation_bitfip(self, indiv):
        length = self.parameters['chromosome size']
        p = 1 / length
        for i in range(length):
            if random.random() <= p:
                indiv.sequence[i].mutate()

    ###############################################################
    #                       Insertion                             #
    ###############################################################

    def insertion(self):
        if self.parameters['insertion'] == 'fitness':
            self.sort_individuals_fitness()
        elif self.parameters['insertion'] == 'age':
            self.sort_individuals_age()
        if len(self.mutated):
            self.individuals = self.individuals[:-len(self.mutated)]
            self.stats['utility'][-1].append([])
            for i in self.mutated:
                f = i.fitness()
                self.stats['utility'][-1][-1].append(f)
                self.individuals.append((i, f, 0))

    ###############################################################
    #                       Statistic                             #
    ###############################################################

    def statistic(self):
        self.sort_individuals_fitness()
        i = [i for i, _, _ in self.individuals]
        fits = [f for _, f, _ in self.individuals]
        ages = [a for _, _, a in self.individuals]
        self.stats['total_fitness'].append(sum(fits))
        self.stats['max_fitness'].append(fits[0])
        self.stats['min_fitness'].append(fits[-1])
        self.stats['mean_fitness'].append(statistics.mean(fits))
        self.stats['fitness_diversity'].append(len(set(i)))
        self.stats['diversity'].append(len(set(self.individuals)))
        self.stats['max_age'].append(max(ages))
        self.stats['mean_age'].append(statistics.mean(ages))


if __name__ == '__main__':
    parameters = {
        'configuration name': 'config1',
        'individual': ['gen_algo.individuals.onemax', 'IndividualOneMax'],

        'population size': 100,  # 100 200 500
        'chromosome size': 50,  # 5 10 50 100

        'nb turn max': 1000,
        'stop after no change': 10000,  # int(config['nb turn max']*0.10),

        # ('select_random')
        # ('select_best')
        # ('select_tournament', 2, 5)
        # ('select_wheel')
        # ('adaptative' ,[
        #            (0.25, 'select_random'), (0.25, 'select_best'),
        #            (0.25, 'select_tournament', 2 , 5), (0.25, 'select_wheel')])

        'selection': ('select_tournament', 2, 5),
        'proportion selection': 0.02,  # 2 / config['population size']

        'crossover': 'mono-point',  # 'mono-point' 'uniforme'
        'proportion crossover': 1,

        # ['n-flip', 1] ['n-flip', 3] ['n-flip', 5] ['bit-flip']
        'mutation': ['bit-flip'],
        'proportion mutation': 0.2,  # 0.1 0.2 0.5 0.8

        'insertion': 'age',  # 'age' 'fitness'
    }
    population = Population(parameters)
    stats = population.start()

    from gen_algo.tools.plot import show_stats

    show_stats(stats)
