#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 20:40:34 2025

@author: nicolas
"""

"""
Equation de Poisson 1D :  -u''(x) = f(x) : sur [0, L] avec f(x) = 1
Problème de Dirichlet : u(0) = u_0 et u(L) = u_L
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix

#=============================================================================#
# Parametres du problème
#=============================================================================#

E = 3 # Nombre d'elements
n = E + 1 # Nombre de noeuds

L = 1
x_0, x_L = 0, L # Domaine de dfinition

u_0, u_L = 1, 1 # Conditions de Dirichlet non-homogene

#=============================================================================#
# Definition de la fonction source f(x)
#=============================================================================#
def f(x):
    return 1

#=============================================================================#
# Definition du maillage
#=============================================================================#

noeud = np.linspace(x_0, x_L, n) # discretisation de l'intervalle [x0,xL] en n noeuds

# Definition de la matrice element #
vector1 = np.arange(1, n) # Creer un vecteur ligne contenant les entiers de 1 a n-1
vector2 = np.arange(2, n + 1) # Creer un vecteur ligne contenant les entiers de 2 a n
element = np.array([vector1, vector2])

# Nombre d'elements
nombre_element = element.shape[1] # element.shape retourne un tuple reprrsentant les dimensions du tableau
                                  # shape[1] donne la taille de la seconde dimension, qui correspond au nombre de colonne

#=============================================================================#
# Assemblage du membre de gauche
#=============================================================================#

# Assemblage de la matrice de rigidité K
K = np.zeros((n, n)) # Initialisation
for i in range(nombre_element):
    # Selection des noeuds bordant l'element i
    
    # Indices des noeuds bordant de l'element i
    N = element[:2, i] # selectionne les deux premieres lignes de la i-eme colonne de la matrice element
    
    # Coordonnee des noeuds bordant l'element
    x = noeud[N-1] # Recupere les coordonnees des deux noeuds definis par N dans la matrice noeud, ou chaque ligne represente les coordonnees d'un noeud
    xa, xb = x[0], x[1] # Coordonnees des deux noeuds bordant l'element
    
    # Longueur de l'element courant
    h = abs(xb -xa)
    
    # Assemblage de la matrice élémentaire
    K_el = (1 / h) * np.array([[1, -1],[-1, 1]])
    
    # Assemblage de la matrice globale    
    K[np.ix_(N-1, N-1)] += K_el

#=============================================================================#
# Second membre
#=============================================================================#

# Assemblage du vecteur de charge b
b = np.zeros(n) # Initialisation
for i in range(nombre_element): # La boucle parcourt chaque element
    # Selection des noeuds bordant l'element i
    
    # Indices des noeuds bordant l'element i
    N = element[:2, i] # selectionne les deux premieres lignes de la i-eme colonne de la matrice element
    
    # Coordonnee des noeuds bordant l'element i
    x = noeud[N-1]  # Recuperation des coordonnees des noeuds bordant l'element courant
    xa, xb = x[0], x[1]  # Coordonnees des noeuds bordant l'element courant
    
    # Longueur de l'element courant
    h = abs(xb - xa)

    # Assemblage du vecteur élémentaire
    b_el = (h/ 2) * np.array([1, 1])
    
    # Assemblage du vecteur global
    b[N-1] += b_el

#=======================
# Contribution des effets de bords du aux conditions de Dirichlet
#=======================

l = np.zeros(n) # Initialisation
for j in range(n): # la boucle parcours les noeuds
    l[j] += u_0 * K[j, 0] + u_L * K[j, -1]

#=======================

B = b - l

#=============================================================================#
# Prise en compte des conditions limites
#=============================================================================#

# Liste des noeuds ou la solution est imposee (bords)
noeuds_bords = np.array([0, n-1])

# Liste complementaire des noeuds ou la solution sera calculee (interieur)
noeuds_int = np.arange(1, n-1)  # Creer une sequence de 1  n-1

#=============================================================================#
# Resolution du systeme lineaire
#=============================================================================#

# Extraction de la sous-matrice A_int
K_int = K[noeuds_int, :][:, noeuds_int]

# Extraction du sous vecteur b_int
B_int = B[noeuds_int]

# Résolution du système linéaire homogène interne
K_sparse = csr_matrix(K_int)
w_int = spsolve(K_sparse, B_int)

# Ajout des conditions aux bords homogène
w = np.concatenate(([0], w_int, [0]))

# Définition du vecteur u_tilde 
u_tilde = np.zeros(n)
u_tilde[0] = u_0
u_tilde[-1] = u_L

u = w + u_tilde

#=============================================================================#
# Solution analytique de l'equation differentielle
#=============================================================================#
x_range = np.linspace(x_0, x_L, 100)
def u_exact(x_range):
    C_1 = (1/ L) * (u_L - u_0) + (L/ 2)
    C_2 = u_0
    return - x_range**2 / 2 + C_1 * x_range + C_2

#=============================================================================#
# Tracer la solution numerique
#=============================================================================#

# Affichage de la solution MEF
plt.plot(noeud, u,
         label="MEF",
         color="blue",
         linestyle="--")

# Affichage de la solution analytique
plt.plot(x_range, u_exact(x_range),
         label="Solution analytique",
         color="red",
         linestyle="-")

plt.xlabel("x")
plt.ylabel("u(x)")
plt.title("Solutions de l'equation differentielle")
plt.grid(True)
plt.legend()
plt.show()