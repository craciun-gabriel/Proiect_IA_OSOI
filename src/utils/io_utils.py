import random

def citeste_matrice(cale_fisier):
    """
    Citește matricea de distanțe dintr-un fișier text.
    Format: N (număr orașe) pe prima linie, urmat de matricea NxN.
    """
    with open(cale_fisier, 'r') as f:
        linii = [linie.strip() for linie in f if linie.strip()]
    n = int(linii[0])
    matrice = [[int(x) for x in linii[i + 1].split()] for i in range(n)]
    return n, matrice

def genereaza_matrice_aleatorie(n, seed=42):
    """
    Generează o matrice de distanțe NxN simetrică cu valori în [1, 100].
    Diagonala va fi 0.
    """
    random.seed(seed)
    matrice = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            dist = random.randint(1, 100)
            matrice[i][j] = dist
            matrice[j][i] = dist
    return matrice