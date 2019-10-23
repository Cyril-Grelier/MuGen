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
            'diversity': [], 'max_age': [], 'mean_age': [], 'parameters': parameters, 'time': 0.0
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

    def sort_individuals_fitness(self):
        self.individuals.sort(key=lambda i: i[1], reverse=True)

    def sort_individuals_age(self):
        self.individuals.sort(key=lambda i: i[2], reverse=False)

    def population_get_older(self):
        self.nb_turns += 1
        self.individuals = [(i, f, (a + 1)) for i, f, a in self.individuals]

    def start(self):
        self.statistic()
        while self.final_condition():
            self.population_get_older()
            self.sort_individuals_fitness()
            self.turn()
            self.statistic()
        # TODO : remove this
        from gen_algo.tools.midi_utils import convert_to_midi, play_midi_file
        for i, indiv in enumerate(self.individuals):
            convert_to_midi(indiv[0], str(i) + ".mid")
            # play_midi_file(str(i) + ".mid")
        return self.termination()

    def turn(self):
        self.selection()
        self.crossover()
        self.mutation()
        self.insertion()

    def final_condition(self):
        if self.nb_turns >= self.parameters['stop after no change']:
            last_max = self.stats['max_fitness'][-self.parameters['stop after no change']:]
            last_min = self.stats['min_fitness'][-self.parameters['stop after no change']:]
            max_change = not all(x >= y for x, y in zip(last_max, last_max[1:]))
            min_change = not all(x >= y for x, y in zip(last_min, last_min[1:]))
            return (not (self.nb_turns == self.parameters['nb turn max'])) and not (not max_change and not min_change)
        else:
            return not (self.nb_turns == self.parameters['nb turn max'])

    def choose_parameter(self):
        pass

    def selection(self):
        """
        chose the type of selection to use and select individuals
        :return:
        """
        self.selected = []
        nb_select = int(len(self.individuals) * self.parameters['proportion selection'])
        if self.parameters['selection'] == 'select_random':
            self.select_random(nb_select)
        elif self.parameters['selection'] == 'select_best':
            self.select_best(nb_select)
        elif self.parameters['selection'] == 'select_tournament':
            print("faut revoir le code")
        #     self.select_tournament(nb_select, self.parameters['nb selected tournament'])
        elif self.parameters['selection'] == 'select_wheel':
            print("faut revoir le code")
        #     self.select_wheel(nb_select)
        elif self.parameters['selection'] == 'adaptative':
            print("TODO")
        #     self.select_random(nb_select)
        #     self.select_best(nb_select)
        #     self.select_tournament(nb_select, self.parameters['nb selected tournament'])
        #     self.select_wheel(nb_select)

    def select_random(self, nb_select):
        selected = random.sample(self.individuals, nb_select)
        self.selected = copy.deepcopy([i for i, _, _ in selected])

    def select_best(self, nb_select):
        selected = self.individuals[0:nb_select]
        self.selected = copy.deepcopy([i for i, _, _ in selected])

    def select_tournament(self, nb_select, nb_players):
        nb_turns = nb_select // 2 if nb_select % 2 == 0 else (nb_select + 1) // 2
        print(nb_select)
        print(nb_turns)
        for _ in range(nb_turns):
            selected = random.sample(self.individuals, nb_players)
            print(selected)
            selected.sort(key=lambda i: i[1], reverse=True)
            print(selected)
            selected = selected[nb_select:]
            print(selected)
            self.selected.append(copy.deepcopy([i for i, _, _ in selected]))

    def select_wheel(self, nb_select):
        individuals, fits, _ = list(zip(*self.individuals))
        total = sum(fits)
        wheel = []
        if total == 0:
            wheel = [1] * len(fits)
        else:
            wheel = [f / total for f in fits]
        probabilities = [sum(wheel[:i + 1]) for i in range(len(wheel))]
        for n in range(nb_select):
            r = random.random()
            for (i, individual) in enumerate(individuals):
                if r <= probabilities[i]:
                    self.selected.append(copy.deepcopy(individual))
                    break

    def crossover(self):
        self.crossed = []
        if self.parameters['proportion crossover'] == 0:
            self.crossed = self.selected
        else:
            for i in range(0, len(self.selected), 2):
                if random.random() <= self.parameters['proportion crossover']:
                    if len(self.selected) <= i + 1:
                        rand = random.choice(self.selected[0:-1])
                        first_child, second_child = self.crossover_individual(self.selected[-1], rand)
                    else:
                        first_child, second_child = self.crossover_individual(self.selected[i], self.selected[i + 1])
                else:
                    first_child = self.individual_class(self.parameters)
                    second_child = self.individual_class(self.parameters)
                    first_child.sequence = copy.deepcopy(self.selected[i][::])
                    # the second child will be a copy of a random parents from the selected parents if the number
                    # of parents is odd
                    sc = random.randrange(len(self.selected[0:-1])) if len(self.selected) <= i + 1 else i + 1
                    second_child.sequence = copy.deepcopy(self.selected[sc].sequence)
                self.crossed.extend([first_child, second_child])

    def crossover_individual(self, i1, i2):
        first_child = self.individual_class(self.parameters)
        second_child = self.individual_class(self.parameters)
        if i1.parameters['type crossover'] == 'mono-point':
            rand = random.randint(1, len(i1.sequence))
            first_child[::] = i1[0:rand] + i2[rand:]
            second_child[::] = i2[0:rand] + i1[rand:]
        elif i1.parameters['type crossover'] == 'uniforme':
            for i in range(i1.parameters['chromosome size']):
                first_child[i], second_child[i] = (i1[i], i2[i]) if random.random() <= 0.5 else (i2[i], i1[i])
        return first_child, second_child

    def mutation(self):
        self.mutated = []
        for i in self.crossed:
            if random.random() <= self.parameters['proportion mutation']:
                self.mutation_individual(i)
            self.mutated.append(i)

    def mutation_individual(self, indiv):
        if self.parameters['mutation'][0] == 'n-flip':
            for i in random.sample(range(len(indiv.sequence)), self.parameters['mutation'][1]):
                indiv.sequence[i].bit = 1 - indiv.sequence[i].bit
        if self.parameters['mutation'][0] == 'bit-flip':
            length = len(indiv.sequence)
            p = 1 / length
            for i in range(length):
                if random.random() <= p:
                    indiv.sequence[i].mutate()

    def insertion(self):
        if self.parameters['insertion'] == 'fitness':
            self.sort_individuals_fitness()
        elif self.parameters['insertion'] == 'age':
            self.sort_individuals_age()
        self.individuals = self.individuals[:-len(self.mutated)]
        for i in self.mutated:
            self.individuals.append((i, i.fitness(), 0))

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

    def termination(self):
        return self.stats

    def __repr__(self):
        s = f"Population size : {len(self.individuals)}"
        for i in self.individuals:
            s += "\n" + str(i)
        return s

# def main():
#     parameters = load_parameters('config1')
#     population = Population(parameters)
#     return population.start()
#
# if __name__ == '__main__':
#     parameters = {
#         'configuration name': 'config1',
#         'individual': ['onemax', 'IndividualOneMax'],
#
#         'population size': 50,  # 100 200 500
#         'chromosome size': 50,  # 5 10 50 100
#
#         'nb turn max': 1000,
#         'stop after no change': 1000,  # int(config['nb turn max']*0.10)
#
#         'selection': 'select_random',  # 'select_random' 'select_best' 'select_tournament'
#         'proportion selection': 0.5,  # 2 / config['population size']
#         'nb selected tournament': 20,  # int(config['population size']*0.4)
#
#         'proportion crossover': 0,
#         'type crossover': 'uniforme',  # 'mono-point' 'uniforme'
#
#         'mutation': ['n-flip', 3],  # ['n-flip', 1] ['n-flip', 3] ['n-flip', 5] ['bit-flip']
#         'proportion mutation': 0.2,  # 0.1 0.2 0.5 0.8
#
#         'insertion': 'fitness',  # 'age' 'fitness'
#     }
#
#     population = Population(parameters)
#     population.individuals[0][0] = 5
#     print(population.individuals[0][0])
