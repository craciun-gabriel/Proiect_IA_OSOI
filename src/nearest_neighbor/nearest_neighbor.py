import time
import sys

def rezolva_tsp_nn(n, matrice, start=0):
    """
    Construiește o soluție TSP folosind euristica constructivă Greedy (Nearest Neighbor).
    
    La fiecare pas, algoritmul alege să se deplaseze către cel mai apropiat oraș 
    nevizitat față de orașul curent, ignorând complet viziunea de ansamblu.
    Avantajul este complexitatea foarte redusă O(N^2), dar nu garantează
    soluția optimă globală.
    """
    timp_start = time.perf_counter()
    vizitat = [False] * n
    traseu = [start]
    vizitat[start] = True
    cost_total = 0
    oras_curent = start

    for _ in range(n - 1):
        dist_min = sys.maxsize
        cel_mai_aproape = -1
        for oras in range(n):
            if not vizitat[oras] and 0 < matrice[oras_curent][oras] < dist_min:
                dist_min = matrice[oras_curent][oras]
                cel_mai_aproape = oras
        
        cost_total += dist_min
        traseu.append(cel_mai_aproape)
        vizitat[cel_mai_aproape] = True
        oras_curent = cel_mai_aproape

    cost_total += matrice[oras_curent][start]
    timp_executie = time.perf_counter() - timp_start
    return traseu, cost_total, timp_executie

def rezolva_tsp_nn_multistart(n, matrice):
    """
    Rezolvă problema sensibilității față de punctul de start din Nearest Neighbor.
    
    Deoarece calitatea traseului variază dramatic în funcție de nodul de pornire,
    acest algoritm rulează NN din absolut toate cele N puncte posibile și 
    îl returnează pe cel mai bun (Multistart NN).
    """
    timp_start = time.perf_counter()
    best_cost = sys.maxsize
    best_tour = []
    
    for start in range(n):
        traseu, cost, _ = rezolva_tsp_nn(n, matrice, start)
        if cost < best_cost:
            best_cost = cost
            best_tour = traseu
            
    timp_executie = time.perf_counter() - timp_start
    return best_tour, best_cost, timp_executie