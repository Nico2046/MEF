#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 22:03:57 2025

@author: nicolas
"""

"""
Equation de Poisson 1D :
                         -u''(x) = f(x) : sur [0, L] avec f(x) = sin(x)
                         
Problème Dirichlet-Neumann :
    
    - condition de Dirichlet sur le bord gauche {0} : u(0) = u_0
    
    - condition de Neumann sur le bord droit {L} : u'(L) = g
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix

#=============================================================================#
# Parametres du problème
#=============================================================================#

E = 4 # Nombre d'elements
n = E + 1 # Nombre de noeuds

L = 2
x_0, x_L = 0, L # Domaine de dfinition

u_0 = 1 # Conditions Dirichlet
g = -1 # Condition Neumann

#=============================================================================#
# Definition de la fonction source f(x)
#=============================================================================#
def f(x):
    return np.sin(x)

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
    
    # Longueur de l'element
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
for i in range(nombre_element): # La boucle parcourt chaque élément
    # Séléction des noeuds bordant l'élément i
    
    # Indices des nœuds bordant de l'élément i
    N = element[:2, i] # séléctionne les deux premières lignes de la i-ème colonne de la matrice element
    
    # Coordonnée des noeuds bordant l'élément
    x = noeud[N-1]  # Récupération des coordonnées des nœuds bordant l'élément courant
    xa, xb = x[0], x[1]  # Coordonnées des nœuds bordant l'élément courant
    
    # Longueur de l'élément courant
    h = abs(xb - xa)
    """
    La méthode d'intégration pour la source est la méthode de Simpson composite
    """
    # Assemblage du vecteur élémentaire
    b_el = ((f(xa) + 4 * f((xa + xb) / 2) + f(xb)) / 6) * (h / 2) * np.array([1, 1])
    
    # Assemblage du vecteur global
    b[N-1] += b_el

#=================
# Prise en compte de la condition de Neumann au bord droit {1}
#=================
b[-1] += g

#=================
# ontribution des effets de bords du à la condition de Dirichlet
#=================

l = np.zeros(n) # Initialisation
for j in range(n): # la boucle parcours les noeuds
    l[j] += u_0 * K[j, 0]

#=================

B = b - l

#=============================================================================#
# Prise en compte des conditions aux bords
#=============================================================================#

# Liste des noeuds ou la solution est imposee (bord gauche uniquement)
noeuds_bords = np.array([0])

# Liste complementaire des noeuds ou la solution sera calculee (interieur + bord droit)
noeuds_int = np.arange(1, n)  # Creer une sequence de 1 à n

#=============================================================================#
# Resolution du systeme lineaire
#=============================================================================#

# Extraction de la sous-matrice A_int
K_int = K[noeuds_int, :][:, noeuds_int]

# Extraction du sous vecteur b_int
B_int = B[noeuds_int]

# Résolution du système linéaire homogène pour w
K_sparse = csr_matrix(K_int)
w_int = spsolve(K_sparse, B_int)

# Ajout des conditions limites fixees
w = np.concatenate(([0], w_int))

# Définition du vecteur u_tilde 
u_tilde = np.zeros(n)
u_tilde[0] = u_0

u = w + u_tilde

#=============================================================================#
# Solution analytique de l'equation differentielle
#=============================================================================#
x_range = np.linspace(x_0, x_L, 100)
def u_exact(x_range):
    C_2 = g - np.cos(L)
    return u_0 + C_2 * x_range + np.sin(x_range)

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