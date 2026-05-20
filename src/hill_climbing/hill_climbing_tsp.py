import time
import random
from simpleai.search import SearchProblem, hill_climbing_random_restarts

class TSPProblem(SearchProblem):
    """
    Definește problema Comis-Voiajorului pentru a fi compatibilă cu biblioteca `simpleai`.
    Starea problemei este un tuplu imutabil reprezentând o permutare a orașelor.
    """
    def __init__(self, n, matrice):
        self.n = n
        self.matrice = matrice
        initial_state = list(range(n))
        random.shuffle(initial_state)
        super().__init__(initial_state=tuple(initial_state))

    def actions(self, state):
        acts = []
        for i in range(self.n):
            for j in range(i + 1, self.n):
                acts.append((i, j))
        return acts

    def result(self, state, action):
        state_list = list(state)
        i, j = action
        state_list[i:j+1] = reversed(state_list[i:j+1])
        return tuple(state_list)

    def value(self, state):
        cost = sum(self.matrice[state[i]][state[(i + 1) % self.n]] for i in range(self.n))
        return -cost 

    def generate_random_state(self):
        """
        Funcție OBLIGATORIE pentru hill_climbing_random_restarts în SimpleAI.
        Explică algoritmului cum să genereze un nou punct de pornire valid (o nouă permutare).
        """
        stare_noua = list(range(self.n))
        random.shuffle(stare_noua)
        return tuple(stare_noua)

def rezolva_tsp_hc(n, matrice, restarts=10):
    """
    Aplică algoritmul alpinistului (Hill Climbing) folosind biblioteca `simpleai`.
    """
    timp_start = time.perf_counter()
    problema = TSPProblem(n, matrice)
    rezultat = hill_climbing_random_restarts(problema, restarts_limit=restarts)
    timp_executie = time.perf_counter() - timp_start
    return list(rezultat.state), -problema.value(rezultat.state), timp_executie