import pygad
import numpy as np
import random
import time

class GATSP:
    """
    Modelează Rezolvarea TSP printr-un Algoritm Genetic, utilizând biblioteca PyGAD.
    
    Problema necesită cromozomi de tip permutare (fiecare oraș trebuie vizitat exact o dată).
    Deoarece operatorii standard de recombinare produc duplicate, folosim operatori personalizați:
    Order Crossover (OX) și Swap Mutation pentru a menține validitatea generațiilor.
    """
    def __init__(self, matrice, pop_size=100, generations=300, mutation_rate=40):
        self.matrice = matrice
        self.n = len(matrice)
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate

    def distanta_ruta(self, solutie):
        total = sum(self.matrice[int(solutie[i])][int(solutie[(i + 1) % self.n])] for i in range(self.n))
        return total

    def fitness_func(self, ga_instance, solutie, solutie_idx):
        """Calculează fitness-ul. PyGAD maximizează, deci trebuie returnat costul rutelor cu minus."""
        return -self.distanta_ruta(solutie)

    def ox_crossover(self, parinti, offspring_size, ga_instance):
        """
        Operatorul personalizat Order Crossover (OX).
        Copiază un segment aleator din părintele 1, apoi completează golurile
        în ordine cu genele din părintele 2, omițând genele deja existente.
        Astfel cromozomii sunt mereu valizi (fără duplicate).
        """
        offspring = []
        idx = 0
        while len(offspring) < offspring_size[0]:
            p1 = parinti[idx % parinti.shape[0]].astype(int).tolist()
            p2 = parinti[(idx + 1) % parinti.shape[0]].astype(int).tolist()
            
            cx1, cx2 = sorted(random.sample(range(self.n), 2))
            copil = [-1] * self.n
            copil[cx1:cx2 + 1] = p1[cx1:cx2 + 1]
            
            set_segment = set(copil[cx1:cx2 + 1])
            gene_ramase = [g for g in p2 if g not in set_segment]
            
            pozitii_libere = [i for i in range(self.n) if copil[i] == -1]
            for pos, gena in zip(pozitii_libere, gene_ramase):
                copil[pos] = gena
                
            offspring.append(copil)
            idx += 1
        return np.array(offspring, dtype=int)

    def swap_mutation(self, offspring, ga_instance):
        """
        Mutație prin interschimbare (Swap Mutation).
        Alege aleator 2 gene din descendent și le interschimbă cu probabilitatea `mutation_rate`.
        """
        rata = self.mutation_rate / 100.0
        for i in range(offspring.shape[0]):
            if random.random() < rata:
                idx1, idx2 = random.sample(range(self.n), 2)
                offspring[i][idx1], offspring[i][idx2] = offspring[i][idx2], offspring[i][idx1]
        return offspring

    def solve(self):
        timp_start = time.perf_counter()
        pop_initiala = [random.sample(range(self.n), self.n) for _ in range(self.pop_size)]
        
        ga = pygad.GA(
            num_generations=self.generations,
            num_parents_mating=max(2, self.pop_size // 2),
            fitness_func=self.fitness_func,
            initial_population=pop_initiala,
            crossover_type=self.ox_crossover,
            mutation_type=self.swap_mutation,
            keep_elitism=2,
            suppress_warnings=True
        )
        
        ga.run()
        solutie, fitness, _ = ga.best_solution()
        
        timp_executie = time.perf_counter() - timp_start
        istoric = [-f for f in ga.best_solutions_fitness]
        
        return list(solutie.astype(int)), -fitness, istoric, timp_executie