import time
import sys

def rezolva_tsp_backtracking(n, matrice, mod='toate', timp_max=60, y_max=10):
    """
    Rezolvă problema TSP folosind Backtracking cu prunere (Branch and Bound).
    
    Args:
        n: numărul de orașe
        matrice: matricea de distanțe NxN
        mod: 'prima' (prima soluție), 'toate' (optimul global), 
             'timp' (cel mai bun în timp limită), 'y_solutii' (cel mai bun din Y)
        timp_max: limita de timp în secunde (pentru modul 'timp')
        y_max: numărul maxim de soluții (pentru modul 'y_solutii')
        
    Returns:
        (best_tour, best_cost, nr_solutii, timp_executie)
    """
    best_cost = sys.maxsize
    best_tour = []
    nr_solutii = 0
    oprire = False
    timp_start = time.perf_counter()

    def bkt(oras_curent, vizitat, traseu, cost):
        nonlocal best_cost, best_tour, nr_solutii, oprire

        if oprire:
            return

        # Cazul de bază: am vizitat toate orașele
        if len(traseu) == n:
            cost_total = cost + matrice[oras_curent][traseu[0]]
            nr_solutii += 1

            if cost_total < best_cost:
                best_cost = cost_total
                best_tour = traseu.copy()

            if mod == 'prima':
                oprire = True
            elif mod == 'y_solutii' and nr_solutii >= y_max:
                oprire = True
            return

        # Parcurgerea vecinilor
        for urmator in range(n):
            if oprire:
                return
                
            if vizitat[urmator]:
                continue

            # Verificare limită de timp (dacă este activată)
            if mod == 'timp' and (time.perf_counter() - timp_start) >= timp_max:
                oprire = True
                return

            cost_nou = cost + matrice[oras_curent][urmator]

            # Prunere branch-and-bound: dacă depășim costul minim, abandonăm ramura
            if mod in ['toate', 'prima', 'y_solutii'] and cost_nou >= best_cost:
                continue

            # Pasul înainte
            vizitat[urmator] = True
            traseu.append(urmator)
            
            bkt(urmator, vizitat, traseu, cost_nou)
            
            traseu.pop()
            vizitat[urmator] = False

    # Startul din orașul 0 
    vizitat = [False] * n
    vizitat[0] = True
    
    bkt(0, vizitat, [0], 0)
    
    timp_executie = time.perf_counter() - timp_start

    return best_tour, best_cost, nr_solutii, timp_executie