import math
import random
import time

class SimulatedAnnealingTSP:
    """
    Implementează meta-euristica Simulated Annealing (Călirea Simulată) pură în Python.
    
    Spre deosebire de Hill Climbing, SA permite temporar acceptarea unor soluții
    mai slabe (mai costisitoare) pentru a scăpa din minimele locale. Probabilitatea 
    acceptării unei soluții slabe depinde de Criteriul Metropolis: P = exp(-ΔE/T),
    fiind mai mare la început (T mare) și scăzând odată cu "răcirea" sistemului.
    """
    def __init__(self, matrice, t_max=10000, t_min=1, alpha=0.95, iters_per_temp=100):
        self.matrice = matrice
        self.n = len(matrice)
        self.t_max = t_max
        self.t_min = t_min
        self.alpha = alpha
        self.iters_per_temp = iters_per_temp

    def cost(self, traseu):
        return sum(self.matrice[traseu[i]][traseu[(i + 1) % self.n]] for i in range(self.n))

    def get_neighbor(self, traseu):
        """Generează o soluție vecină aplicând operatorul de 2-opt (inversare de subsegment)."""
        i, j = sorted(random.sample(range(self.n), 2))
        vecin = traseu[:]
        vecin[i:j+1] = reversed(vecin[i:j+1])
        return vecin

    def solve(self):
        timp_start = time.perf_counter()
        
        # Stare inițială aleatoare
        current_tour = list(range(self.n))
        random.shuffle(current_tour)
        current_cost = self.cost(current_tour)
        
        best_tour = current_tour[:]
        best_cost = current_cost
        temp = self.t_max
        istoric_costuri = []

        while temp > self.t_min:
            for _ in range(self.iters_per_temp):
                neighbor = self.get_neighbor(current_tour)
                neighbor_cost = self.cost(neighbor)
                delta = neighbor_cost - current_cost
                
                # Criteriul Metropolis de acceptare
                if delta <= 0 or math.exp(-delta / temp) > random.random():
                    current_tour = neighbor
                    current_cost = neighbor_cost
                    
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_tour = current_tour[:]
            
            istoric_costuri.append(best_cost)
            # Răcirea sistemului (Răcire Geometrică)
            temp *= self.alpha
            
        timp_executie = time.perf_counter() - timp_start
        return best_tour, best_cost, istoric_costuri, timp_executie