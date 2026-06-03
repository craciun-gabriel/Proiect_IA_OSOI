import time
import random


class HillClimbingTSP:
    def __init__(self, n, matrice, restarts=10):
        self.n = n
        self.matrice = matrice
        self.restarts = restarts
        self.k_vecini = min(200, n * (n - 1) // 2)

    def _cost(self, traseu):
        return sum(self.matrice[traseu[i]][traseu[(i + 1) % self.n]] for i in range(self.n))

    def _vecin_2opt(self, traseu, i, j):
        neighbor = traseu[:]
        neighbor[i:j + 1] = neighbor[i:j + 1][::-1]
        return neighbor

    def solve(self):
        timp_start = time.perf_counter()

        best_traseu = None
        best_cost = float('inf')

        for _ in range(self.restarts):
            current = list(range(self.n))
            random.shuffle(current)
            c_current = self._cost(current)

            improved = True
            while improved:
                improved = False
                for _ in range(self.k_vecini):
                    i, j = sorted(random.sample(range(self.n), 2))
                    neighbor = self._vecin_2opt(current, i, j)
                    c_neighbor = self._cost(neighbor)

                    if c_neighbor < c_current:
                        current = neighbor
                        c_current = c_neighbor
                        improved = True
                        break

            if c_current < best_cost:
                best_cost = c_current
                best_traseu = current[:]

        timp_executie = time.perf_counter() - timp_start
        return best_traseu, best_cost, timp_executie


def rezolva_tsp_hc(n, matrice, restarts=10):
    hc = HillClimbingTSP(n, matrice, restarts=restarts)
    return hc.solve()