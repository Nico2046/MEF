#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 18:13:50 2025

@author: nicolas-fournie
"""

"""
Equation de Poisson 2D :
                         -Δu(x,y) = 1 sur Ω = [0,lx]x[0,ly] ;
                         
avec problème de Dirichlet non-uniforme non-homogène :
    
    - u(x,y) = u_1 sur Gamma_1 = {(x, y) in Ω | (x=0, y)}
    
    - u(x,y) = u_2 sur Gamma_2 = {(x, y) in Ω | (x=lx, y)}
    
    - u(x,y) = u_3 sur Gamma_3 = {(x, y) in Ω | (x, y=ly)}
    
    - u(x,y) = u_4 sur Gamma_4 = {(x, y) in Ω | (x, y=0)}
"""

import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt

#=============================================================================#
# Paramètre du problème de Dirichlet
#=============================================================================#
u_1 = 0
u_2 = 1
u_3 = 1
u_4 = 0
#=============================================================================#
# Définition du maillage
#=============================================================================#

###################
#  (n3) ---- (n4) #
#    |  \     |   #
#    |   \    |   #
#    |    \   |   #
#  (n1) ---- (n2) #
###################

Nx, Ny = 50, 50  # Nombre de divisions dans chaque direction
lx, ly = 1, 1 # Longueur des axes x, (resp. y)
NoN = (Nx + 1) * (Ny + 1) # Nombre de noeuds
NoT = 2 * Nx * Ny # Nombre d'éléments
NnpE = 3 # Nombre de noeud par élément
Dim = 2 # Dimension du maillage

# Génère une grille de coordonnée à partir des deux tableaux x et y.
# X et Y sont des matrices de taille (Ny+1)x(Nx+1) contenant les coordonnées x et y de chaque point du maillage.
#
# -  X contient les coordonnées x répétées en ligne
# 
# -  Y contient les coordonnées y répétées en colonne
x = np.linspace(0, lx, Nx + 1)
y = np.linspace(0, ly, Ny + 1)
X, Y = np.meshgrid(x, y)

# Coordonnées des noeuds
# nodes est une matrice de taille (NoN) x (Dim)
nodes = np.vstack([X.ravel(), Y.ravel()]).T    # ravel() : transforme une matrice en un tableau ligne en mettant bout à bout les lignes de la matrice
                                               # vstack() : empile les deux tableaux en une matrice (Dim x NoN), la 1ère ligne contient les coordonnées x
                                               #                                                                 la 2ème ligne contient les coordonnées y
                                               # T : prend la transposé, transormant la matrice en (NoN x Dim), ou chaque ligne représente la coordonnée (x,y) d'un noeud

# Création des éléments triangulaires
triangles = [] # stocke les indices des noeuds qui composent chaque élément triangulaire
for i in range(Ny):      # parcourt l'axe des y
    for j in range(Nx):  # parcourt l'axe des x
    
        # Numérotations des noeuds
        n1 = i * (Nx + 1) + j   # coin bas-gauche
        n2 = n1 + 1             # coin bas-droit
        n3 = n1 + (Nx + 1)      # coin haut-gauche
        n4 = n3 + 1             # coin haut-droit
        
        # Création des deux triangles composants la cellule carré
        triangles.append([n1, n2, n3])  # Premier triangle (bas-gauche)
        triangles.append([n2, n4, n3])  # Deuxième triangle (haut-droit)

# Triangles est une matrice de taille (NoE)x(NnpE) , NoE : nombre d'éléments, NnpE : nombre de noeuds par élément
triangles = np.array(triangles) # convertie la liste triangles en tableau

#=============================================================================#
# Assemblage de la matrice de rigidité K
#=============================================================================#

K = np.zeros((NoN, NoN)) # Initialisation
for i in range(NoT): # parcourt les éléments triangulaires
    #====================================
    # Selection des sommets composants le triangle i
    
    # Indices des sommets composants le triangle courant
    N = triangles[i, :3] # séléctionne les trois colonnes de la i-ème ligne de la matrice triangles
    
    # Coordonnee des sommets composants le triangle
    sommets = nodes[N]  # Matrice (3x2) dont les lignes sont les coordonées (x,y) des sommets
    N1 = sommets[0, :2] # séléctionne les deux colonnes de la première ligne de la matrice sommets
    N2 = sommets[1, :2] # séléctionne les deux colonnes de la deuxième ligne de la matrice sommets
    N3 = sommets[2, :2] # séléctionne les deux colonnes de la troisème ligne de la matrice sommets
    
    x1, y1 = N1[0], N1[-1] # coorodnnées du sommet N1
    x2, y2 = N2[0], N2[-1] # coorodnnées du sommet N2
    x3, y3 = N3[0], N3[-1] # coorodnnées du sommet N3
    
    # Aire du triangle courant
    Aire = abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))/ 2)
    
    #====================================
    # Coefficients des fonctions de forme
    #====================================
    a1 = x2 * y3 - x3 * y2
    a2 = x3 * y1 - x1 * y3
    a3 = x1 * y2 - x2 * y1
   
    b1 = y2 - y3
    b2 = y3 - y1
    b3 = y1 - y2
    
    c1 = x3 - x2
    c2 = x1 - x3
    c3 = x2 - x1
    #====================================
    # Assemblage de la matrice élémentaire
    K_el = (1/ (4 * Aire)) * np.array([[b1*b1 + c1*c1, b1*b2 + c1*c2, b1*b3 + c1*c3],
                                       [b2*b1 + c2*c1, b2*b2 + c2*c2, b2*b3 + c2*c3],
                                       [b3*b1 + c3*c1, b3*b2 + c3*c2, b3*b3 + c3*c3]])
    
    # Assemblage de la matrice globale    
    K[np.ix_(N, N)] += K_el
    
#=============================================================================#
# Second membre (terme de source)
#=============================================================================#

# Assemblage du vecteur de charge : contribution de la source f(x)
B = np.zeros(NoN) # Initialisation
for i in range(NoT):
    #====================================
    # Selection des sommets composant le triangle i
    
    # Indices des sommets composant le triangle
    N = triangles[i, :3] # séléctionne les trois colonnes de la i-ème ligne de la matrice triangles
    
    # Coordonnee des sommets composant le triangle
    sommets = nodes[N] # Matrice 3x2 dont les lignes sont les coordonées (x,y) des sommets
    N1 = sommets[0, :2] # séléctionne les deux colonnes de la première ligne de la matrice sommets
    N2 = sommets[1, :2] # séléctionne les deux colonnes de la deuxième ligne de la matrice sommets
    N3 = sommets[2, :2] # séléctionne les deux colonnes de la troisème ligne de la matrice sommets
    
    x1, y1 = N1[0], N1[-1] # coorodnnées du sommet N1
    x2, y2 = N2[0], N2[-1] # coorodnnées du sommet N2
    x3, y3 = N3[0], N3[-1] # coorodnnées du sommet N3
    
    # Aire du triangle courant
    Aire = abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))/ 2)
    
    #====================================
    # Coefficients des fonctions de forme
    #====================================
    a1 = x2 * y3 - x3 * y2
    a2 = x3 * y1 - x1 * y3
    a3 = x1 * y2 - x2 * y1
   
    b1 = y2 - y3
    b2 = y3 - y1
    b3 = y1 - y2
    
    c1 = x3 - x2
    c2 = x1 - x3
    c3 = x2 - x1
    #====================================
    # Coefficient du vecteur de charge
    B1 = (1/6) * (x3 * (y1 - y2) + x1 * (y2 - y3) + x2 * (-y1 + y3))
    B2 = B1
    B3 = B1
    #====================================
    # Assemblage du vecteur local
    B_el = np.array([B1, B2, B3])

    # Assemblage du vecteur global
    B[N] += B_el

#=============================================================================#
# Prise en compte des conditions aux limites
#=============================================================================#

# Identification des nœuds de bord et intérieurs.
# On considère comme nœuds de bord ceux dont l'une des coordonnées est égale à 0 ou 1

tol = 1e-10
interior_nodes = []

boundary_nodes_gamma_1 = []
boundary_nodes_gamma_2 = []
boundary_nodes_gamma_3 = []
boundary_nodes_gamma_4 = []

for i, (xi, yi) in enumerate(nodes):
    is_boundary = False  # On suppose d'abord que le nœud est intérieur
    if (xi < tol):
        boundary_nodes_gamma_1.append(i)
        is_boundary = True
    if (xi > lx - tol):
        boundary_nodes_gamma_2.append(i)
        is_boundary = True
    if (yi > ly - tol):
        boundary_nodes_gamma_3.append(i)
        is_boundary = True
    if (yi < tol):
        boundary_nodes_gamma_4.append(i)
        is_boundary = True
    if not is_boundary:
        interior_nodes.append(i)

#=============================
# Assemblage du vecteur l : contribution des effets de bords

l = np.zeros(NoN) # Initialisation
for i in range(NoN): # la boucle parcours les noeuds
    for j in boundary_nodes_gamma_1: # la boucle parcours les noeuds sur le bord gamma_1
        l[i] += u_1 * K[i, j]

for i in range(NoN): # la boucle parcours les noeuds
    for j in boundary_nodes_gamma_2: # la boucle parcours les noeuds sur le bord gamma_2
        l[i] += u_2 * K[i, j]

for i in range(NoN): # la boucle parcours les noeuds
    for j in boundary_nodes_gamma_3: # la boucle parcours les noeuds sur le bord gamma_2
        l[i] += u_3 * K[i, j]

for i in range(NoN): # la boucle parcours les noeuds
    for j in boundary_nodes_gamma_4: # la boucle parcours les noeuds sur le bord gamma_2
        l[i] += u_4 * K[i, j]
    
#=============================

L = B - l

#=============================================================================#
# Résolution du système linéaire
#=============================================================================#
K_int = K[np.ix_(interior_nodes, interior_nodes)]
L_int  = L[interior_nodes]

K_sparse = csr_matrix(K_int)

w_int = spsolve(K_int, L_int)

w = np.zeros(NoN)
w[interior_nodes] = w_int

# Définition du vecteur u_tilde 
u_tilde = np.zeros(NoN)
u_tilde[boundary_nodes_gamma_1] = u_1
u_tilde[boundary_nodes_gamma_2] = u_2
u_tilde[boundary_nodes_gamma_3] = u_3
u_tilde[boundary_nodes_gamma_4] = u_4

u = w + u_tilde

# =============================================================================
# Affichage de la solution
# =============================================================================
plt.figure(figsize=(6,5))

# Tracé par éléments finis avec une interpolation sur les triangles
plt.tricontourf(nodes[:,0], nodes[:,1], triangles, u,
                cmap='inferno', # plasma, inferno, magma
                levels=100)
plt.colorbar(label='u(x,y)')

#plt.triplot(nodes[:,0], nodes[:,1], triangles, color='k', linewidth=0.1)

plt.title("Solution FEM")
plt.xlabel("x")
plt.ylabel("y")
plt.show()