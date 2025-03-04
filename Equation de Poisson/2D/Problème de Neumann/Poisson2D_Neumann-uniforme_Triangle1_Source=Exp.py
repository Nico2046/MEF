#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 15:44:55 2025

@author: nicolas-fournie
"""

"""
Equation de Laplace 2D :
                              -Δu(x,y) = 1 sur Ω = [0,lx]x[0,ly] ;
                              
avec problème de Neumann uniforme : ∂u/∂n(x,y) = g sur ∂Ω , ou n est le vecteur normal au bord.

La relation de compatibilité est : g = -A/4 , ou A est l'aire du domaine Ω.

Pour un problème de Neumann pur, la solution de l'équation de Poisson est définie à une constante additive près. Il y a donc une infinité de solutions
qui diffèrent seulement par une constante. Pour rendre le problème bien posé et obtenir une solution unique, il faut ajouter une contrainte
supplémentaire. Une méthode courante consiste à fixer la moyenne de la solution sur le domaine, par exemple :
    
                     (1/ nombre de noeuds) * Sum_{i=1}^{nombre de noeuds} u_i = 0 ;
                     
cette contrainte élimine l'ambiguïté due à l'infinité de solutions définies à une constante près.

Dans la pratique, pour imposer cette condition dans la résolution du système linéaire, on remplace l'une des équations (souvent la première)
par l'équation de normalisation. Cela se traduit par :
    
    - K[0, :] = np.ones(nombre_nodes) / nombre_nodes : modifie la première ligne de la matrice K, de sorte que l'équation correspondante soit celle
                                                       définie plus haut.
                                                       
    - B[0] = 0 : fixe le côté droit de l'équation de normalisation à zéro, imposant ainsi que la moyenne de u soit nulle
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix

#=============================================================================#
# Paramètre du problème
#=============================================================================#
g = -1/4
#=============================================================================#
# Définition de la fonction source f(x,y)
#=============================================================================#
def f(x, y):
    Q = 1
    k = 1/5
    sigma = 1/4
    return (Q/ k) * np.exp(- (x**2 + y**2)/ (2 * sigma**2))

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

# Assemblage du vecteur de charge B
B = np.zeros(NoN) # Initialisation
for i in range(NoT):
    #====================================
    # Selection des sommets composant le triangle i
    
    # Indices des sommets composant le triangle
    N = triangles[i, :3] # séléctionne les trois colonnes de la i-ème ligne de la matrice triangles
    
    # Coordonnee des sommets composant le triangle
    sommets = nodes[N] # Matrice 3x2 dont les lignes sont les coordonées (x,y) des sommets
    
    x1, y1 = sommets[0] # coorodnnées du sommet N1
    x2, y2 = sommets[1] # coorodnnées du sommet N2
    x3, y3 = sommets[2] # coorodnnées du sommet N3
    
    # Aire du triangle courant
    Aire = abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))/ 2)
    
    #====================================
    # Assemblage du vecteur élémentaire
    #====================================
    """
    La méthode d'intégration de la source est la quadrature de Gauss à 3 points sur un triangle
    """
    # Coordonnée des points d'intégrations
    P1Tx = x1 + (x2 - x1) * (1/6) + (x3 - x1) * (2/3)
    P1Ty = y1 + (y2 - y1) * (1/6) + (y3 - y1) * (2/3)
    
    P2Tx = x1 + (x2 - x1) * (2/3) + (x3 - x1) * (1/6)
    P2Ty = y1 + (y2 - y1) * (2/3) + (y3 - y1) * (1/6)
    
    P3Tx = x1 + (x2 - x1) * (1/6) + (x3 - x1) * (1/6)
    P3Ty = y1 + (y2 - y1) * (1/6) + (y3 - y1) * (1/6)
    
    # Assemblage du vecteur élémentaire
    B_el = (f(P1Tx, P1Ty) + f(P2Tx, P2Ty) + f(P3Tx, P3Ty)) * (Aire / 3) * np.ones(3)

    # Assemblage du vecteur global
    B[N] += B_el

#=============================================================================#
# Prise en compte des conditions aux limites (CL) par substitution
#=============================================================================#

# Identification des nœuds de bord et intérieurs.
# On considère comme nœuds de bord ceux dont l'une des coordonnées est égale à 0 ou 1

# Identification des nœuds de bord et intérieurs
tol = 1e-10
boundary_nodes = [i for i, (xi, yi) in enumerate(nodes) if xi < tol or xi > lx - tol or yi < tol or yi > ly - tol]


# Intégration de g sur le bord : la contribution de Neumann se répartit uniformément sur tous les noeuds composants les bords
for i in boundary_nodes:
    xi, yi = nodes[i]
    if xi == 0 or xi == lx:
        B[i] += g * (1 / Ny)
    if yi == 0 or yi == ly:
        B[i] += g * (1 / Nx)

# Normalisation pour un problème de Neumann bien posé
K[0, :] = np.ones(NoN) / NoN
B[0] = 1 # ici on impose que la moyenne de u est 1

K_sparse = csr_matrix(K)
u = spsolve(K_sparse, B)

# =============================================================================
# Affichage de la solution
# =============================================================================
plt.figure(figsize=(6,5))

# Tracé par éléments finis avec une interpolation sur les triangles
plt.tricontourf(nodes[:,0], nodes[:,1], triangles, u,
                cmap='inferno', # splasma, inferno, magma
                levels=100)
plt.colorbar(label='u(x,y)')

#plt.triplot(nodes[:,0], nodes[:,1], triangles, color='k', linewidth=0.1)

plt.title("Solution FEM")
plt.xlabel("x")
plt.ylabel("y")
plt.show()