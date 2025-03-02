#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 23:22:26 2025

@author: nicolas
"""

"""
Equation de Laplace 1D :  -u''(x) = f(x) : sur [0, L] avec f(x) = sin(x)
Problème de Neumann non-uniforme : u'(0) = g_0 et u'(L) = g_L
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

L = 3
x_0, x_L = 0, L # Domaine de dfinition

# Relation de compatibilité entre la source f(x) et les conditions de Neumann
g_0 = 1
g_L = g_0 - 1 + np.cos(L)

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

"""
Avec des conditions de Neumann aux deux extrémités, la matrice K a une ligne de sommes nulles, ce qui signifie qu'elle est singulière (non inversible).
Cette singularité vient du fait que l’équation de Laplace avec des conditions de Neumann admet une famille infinie de solutions : 
    si u(x) est une solution alors u(x) + C aussi.
L’ajout de K[0, 0] += epsilon revient à ajouter un terme dans le système linéaire de la forme :
    
    int_{Omega} u'(x) * v(x) dx + epsilon * u(0) = int_{Omega} f(x) * v(x) dx + g_L v(L) - g_0 v(0)

Cela modifie la formulation variationnelle du problème en introduisant un terme de régularisation, ici epsilon. Cela permet de lever la singularité
de la matrice K.
"""
epsilon = 0.01
K[0, 0] += epsilon

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
# Prise en compte des conditions aux bords de Neumann
#=================

# Contribution au bord gauche {0}
b[0] -= g_0

# Contribution au bord droit {L}
b[-1] += g_L

#=============================================================================#
# Resolution du systeme lineaire
#=============================================================================#
K_sparse = csr_matrix(K)
u = spsolve(K_sparse, b)

#=============================================================================#
# Solution analytique de l'equation differentielle
#=============================================================================#
x_range = np.linspace(x_0, x_L, 100)
def u_exact(x_range):
    C_1 = g_0 - 1
    return  np.sin(x_range) + C_1 * x_range

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