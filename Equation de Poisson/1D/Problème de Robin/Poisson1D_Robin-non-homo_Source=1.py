#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 29 15:39:52 2025

@author: nicolas-fournie
"""

"""
Equation de Poisson 1D :
                         -u''(x) = f(x) sur 𝛀 = ]0, L[ (L > 0) avec f(x) = 1
                         
Problème de Robin non-homogène :
    
    - Sur le bord gauche {0} : 𝜶_0 * u(0) - 𝛃_0 * u'(0) = g_0
    
    - Sur le bord droit {L} : 𝜶_L * u(L) + 𝛃_L * u'(L) = g_L

Le signe devant le terme dérivé change selon la partie du bord considérée : la raison est que le vecteur normal à ]0, L[ au point 0 pointe
vers la direction négative (gauche), tandis qu'en L ce vecteur pointe vers les positifs. 
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

L = 1
x_0, x_L = 0, L # Domaine de définition

alpha_0 = 1
alpha_L = 1

beta_0 = 1/2
beta_L = 1

g_0 = 1
g_L = 1

#=============================================================================#
# Definition du maillage
#=============================================================================#

nodes = np.linspace(x_0, x_L, n) # discretisation de l'intervalle [x0,xL] en n noeuds

# Definition de la matrice element #
vector1 = np.arange(1, n) # Creer un vecteur ligne contenant les entiers de 1 a n-1
vector2 = np.arange(2, n + 1) # Creer un vecteur ligne contenant les entiers de 2 a n
element = np.array([vector1, vector2])

#=============================================================================#
# Assemblage de la matrice de rigidité
#=============================================================================#

K = np.zeros((n, n)) # Initialisation
for i in range(E):
    # Selection des noeuds bordant l'element i
    
    # Indices des noeuds bordant de l'element i
    N = element[:2, i] # selectionne les deux premieres lignes de la i-eme colonne de la matrice element
    
    # Coordonnee des noeuds bordant l'element
    x = nodes[N-1] # Recupere les coordonnees des deux noeuds definis par N dans la matrice noeud, ou chaque ligne represente les coordonnees d'un noeud
    xa, xb = x[0], x[1] # Coordonnees des deux noeuds bordant l'element
    
    # Longueur de l'element
    h = abs(xb - xa)
    
    # Assemblage de la matrice élémentaire
    K_el = (1/ h) * np.array([[1, -1],[-1, 1]])
    
    # Assemblage de la matrice globale    
    K[np.ix_(N-1, N-1)] += K_el

# Ajout de la contribution de Robin à la matrice de rigidité
K[0, 0] += alpha_0/ beta_0
K[-1, -1] += alpha_L/ beta_L

#=============================================================================#
# Assemblage du vecteur de charge
#=============================================================================#

b = np.zeros(n) # Initialisation
for i in range(E): # La boucle parcourt chaque élément
    # Séléction des noeuds bordant l'élément i
    
    # Indices des nœuds bordant de l'élément i
    N = element[:2, i] # séléctionne les deux premières lignes de la i-ème colonne de la matrice element
    
    # Coordonnée des noeuds bordant l'élément
    x = nodes[N-1]  # Récupération des coordonnées des nœuds bordant l'élément courant
    xa, xb = x[0], x[1]  # Coordonnées des nœuds bordant l'élément courant
    
    # Longueur de l'élément courant
    h = abs(xb - xa)

    # Assemblage du vecteur local
    b_el = (h / 2) * np.array([1, 1])
    
    # Assemblage du vecteur global
    b[N-1] += b_el
    
# Ajout de la contribution de Robin au vecteur de charge
b[0] += g_0/ beta_0
b[-1] += g_L/ beta_L

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
    return (2 * beta_L * g_0 + 2 * beta_0 * g_L + 2 * beta_0 * beta_L * L + 2 * alpha_L * g_0 * L + 
   alpha_L * beta_0 * L**2 - 2 * alpha_L * g_0 * x_range + 2 * alpha_0 * g_L * x_range + 
   2 * alpha_0 * beta_L * L * x_range + alpha_0 * alpha_L * L**2 * x_range - alpha_L * beta_0 * x_range**2 - 
   alpha_0 * beta_L * x_range**2 - 
   alpha_0 * alpha_L * L * x_range**2)/(2 * (alpha_L * beta_0 + alpha_0 * beta_L + 
     alpha_0 * alpha_L * L))
                                  
#=============================================================================#
# Tracer la solution numerique
#=============================================================================#

# Affichage de la solution MEF
plt.plot(nodes, u,
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