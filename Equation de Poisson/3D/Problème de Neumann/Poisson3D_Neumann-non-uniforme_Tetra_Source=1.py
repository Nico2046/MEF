#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 15:04:28 2025

@author: nicolas-fournie
"""

import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix
import pyvista as pv

"""
Equation de Poisson 3D : -Δu(x,y,z) = 1 sur Ω = [0,lx]x[0,ly]x[0,lz] ;
avec problème de Neumann non-uniforme :
    
    - ∂u/∂n = g_1 sur Gamma_1 = {(x, y, z) in Ω | (x, y=ly, z)}
    
    - ∂u/∂n = g_2 sur Gamma_2 = {(x, y, z) in Ω | (x, y=0, z)}
    
    - ∂u/∂n = g_3 sur Gamma_3 = {(x, y, z) in Ω | (x=0, y, z)}
    
    - ∂u/∂n = g_4 sur Gamma_4 = {(x, y, z) in Ω | (x=lx, y, z)}
    
    - ∂u/∂n = g_5 sur Gamma_5 = {(x, y, z) in Ω | (x, y, z=lz)}
    
    - ∂u/∂n = g_6 sur Gamma_6 = {(x, y, z) in Ω | (x, y, z=0)}

n vecteur normal au bord.

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

#=============================================================================#
# Définition du maillage
#=============================================================================#

Nx, Ny, Nz = 20, 20, 20  # Nombre de divisions dans chaque direction
lx, ly, lz = 1, 1, 1
x = np.linspace(0, lx, Nx + 1)
y = np.linspace(0, ly, Ny + 1)
z = np.linspace(0, lz, Nz + 1)

X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# Coordonnées des noeuds
nodes = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T  # ravel() : transforme une matrice en un tableau ligne en mettant bout à bout les lignes de la matrice
                                             # vstack() : empile les deux tableaux en une matrice 2 x N, la 1ère ligne contient les coordonnées x
                                             #                                                           la 2ème ligne contient les coordonnées y

                                             # T : prend la transposé, transormant la matrice en N x 2, ou chaque ligne représente la coordonnée (x,y) d'un noeud
                                             
# Création des éléments tétraèdriques
tetrahedra = [] # stocke les indices des noeuds qui composent chaque élément tétraèdrique

# Parcours de chaque cellule cubique du maillage
for i in range(Nx):
    for j in range(Ny):
        for k in range(Nz):
            # Calcul des indices des 8 noeuds du cube
            n0 = i     * (Ny+1) * (Nz+1) + j     * (Nz+1) + k
            n1 = (i+1) * (Ny+1) * (Nz+1) + j     * (Nz+1) + k
            n2 = (i+1) * (Ny+1) * (Nz+1) + (j+1) * (Nz+1) + k
            n3 = i     * (Ny+1) * (Nz+1) + (j+1) * (Nz+1) + k
            n4 = i     * (Ny+1) * (Nz+1) + j     * (Nz+1) + (k+1)
            n5 = (i+1) * (Ny+1) * (Nz+1) + j     * (Nz+1) + (k+1)
            n6 = (i+1) * (Ny+1) * (Nz+1) + (j+1) * (Nz+1) + (k+1)
            n7 = i     * (Ny+1) * (Nz+1) + (j+1) * (Nz+1) + (k+1)

            # Décomposition en 6 tétraèdres utilisant la diagonale reliant n0 et n6
            tetrahedra.append([n0, n1, n3, n6])
            tetrahedra.append([n1, n2, n3, n6])
            tetrahedra.append([n0, n1, n4, n6])
            tetrahedra.append([n1, n4, n5, n6])
            tetrahedra.append([n0, n3, n4, n6])
            tetrahedra.append([n3, n4, n7, n6])

tetrahedra = np.array(tetrahedra)

number_tetrahedra = tetrahedra.shape[0] # element.shape retourne un tuple reprrsentant les dimensions du tableau
                                      # shape[0] donne la taille de la première dimension, qui correspond au nombre de ligne

NoN = (Nx + 1) * (Ny + 1) * (Nz + 1) # Number of Nodes

#=============================================================================#
# Paramètre du problème de Neumann
#=============================================================================#

# relation de compatibilité du problème de Neumann avec la source
g_1 = -1
g_2 = 0
g_3 = 1
g_4 = 0
g_5 = 1
g_6 = -lz - (lz/ ly) * (g_1 + g_2) - (lz/ lx) * (g_3 + g_4) - g_5

#=============================================================================#
# Assemblage de la matrice de rigidité K
#=============================================================================#
K = np.zeros((NoN, NoN)) # Initialisation
for i in range(number_tetrahedra): # parcourt les éléments tétraèdriques
    #====================================
    # Selection des sommets composants le tétraèdre i
    
    # Indices des sommets composants le tétraèdre courant
    N = tetrahedra[i, :4] # séléctionne les trois colonnes de la i-ème ligne de la matrice tetrahedra
    
    # Coordonnee des sommets composants le tétraèdre
    sommets = nodes[N]  # Matrice (4x3) dont les lignes sont les coordonées (x,y,z) des sommets
    
    x1, y1, z1 = sommets[0] # coorodnnées du sommet N1
    x2, y2, z2 = sommets[1] # coorodnnées du sommet N2
    x3, y3, z3 = sommets[2] # coorodnnées du sommet N3
    x4, y4, z4 = sommets[3] # coorodnnées du sommet N4
    
    # Volume du tétraèdre courant
    Vol = abs( ( (1/6) * (-x2 * y3 * z1 + x2 * y4 * z1 + x1 * y3 * z2 - x1 * y4 * z2 + x2 * y1 * z3 - 
   x1 * y2 * z3 + x1 * y4 * z3 - x2 * y4 * z3 + 
   x4 * (-y2 * z1 + y3 * z1 + y1 * z2 - y3 * z2 - y1 * z3 + y2 * z3) + (-x2 * y1 + 
      x1 * y2 - x1 * y3 + x2 * y3) * z4 + 
   x3 * (-y4 * z1 - y1 * z2 + y4 * z2 + y2 * (z1 - z4) + y1 * z4))
           ) )
    
    #====================================
    # Assemblage de la matrice élémentaire
    DerivPhi1x = (1/ (6 * Vol)) * (y4 * (-z2 + z3) + y3 * (z2 - z4) + y2 * (-z3 + z4))
    DerivPhi1y = (1/ (6 * Vol)) * (x4 * (z2 - z3) + x2 * (z3 - z4) + x3 * (-z2 + z4))
    DerivPhi1z = (1/ (6 * Vol)) * (x4 * (-y2 + y3) + x3 * (y2 - y4) + x2 * (-y3 + y4))
    
    DerivPhi2x = (1/ (6 * Vol)) * (y4 * (z1 - z3) + y1 * (z3 - z4) + y3 * (-z1 + z4))
    DerivPhi2y = (1/ (6 * Vol)) * (x4 * (-z1 + z3) + x3 * (z1 - z4) + x1 * (-z3 + z4))
    DerivPhi2z = (1/ (6 * Vol)) * (x4 * (y1 - y3) + x1 * (y3 - y4) + x3 * (-y1 + y4))
    
    DerivPhi3x = (1/ (6 * Vol)) * (y4 * (-z1 + z2) + y2 * (z1 - z4) + y1 * (-z2 + z4))
    DerivPhi3y = (1/ (6 * Vol)) * (x4 * (z1 - z2) + x1 * (z2 - z4) + x2 * (-z1 + z4))
    DerivPhi3z = (1/ (6 * Vol)) * (x4 * (-y1 + y2) + x2 * (y1 - y4) + x1 * (-y2 + y4))
    
    DerivPhi4x = (1/ (6 * Vol)) * (y3 * (z1 - z2) + y1 * (z2 - z3) + y2 * (-z1 + z3))
    DerivPhi4y = (1/ (6 * Vol)) * (x3 * (-z1 + z2) + x2 * (z1 - z3) + x1 * (-z2 + z3))
    DerivPhi4z = (1/ (6 * Vol)) * (x3 * (y1 - y2) + x1 * (y2 - y3) + x2 * (-y1 + y3))
    
    K11 = Vol * ( DerivPhi1x * DerivPhi1x + DerivPhi1y * DerivPhi1y + DerivPhi1z * DerivPhi1z )
    K12 = Vol * ( DerivPhi1x * DerivPhi2x + DerivPhi1y * DerivPhi2y + DerivPhi1z * DerivPhi2z )
    K13 = Vol * ( DerivPhi1x * DerivPhi3x + DerivPhi1y * DerivPhi3y + DerivPhi1z * DerivPhi3z )
    K14 = Vol * ( DerivPhi1x * DerivPhi4x + DerivPhi1y * DerivPhi4y + DerivPhi1z * DerivPhi4z )
    
    K22 = Vol * ( DerivPhi2x * DerivPhi2x + DerivPhi2y * DerivPhi2y + DerivPhi2z * DerivPhi2z )
    K23 = Vol * ( DerivPhi2x * DerivPhi3x + DerivPhi2y * DerivPhi3y + DerivPhi2z * DerivPhi3z )
    K24 = Vol * ( DerivPhi2x * DerivPhi4x + DerivPhi2y * DerivPhi4y + DerivPhi2z * DerivPhi4z )
    
    K33 = Vol * ( DerivPhi3x * DerivPhi3x + DerivPhi3y * DerivPhi3y + DerivPhi3z * DerivPhi3z )
    K34 = Vol * ( DerivPhi3x * DerivPhi4x + DerivPhi3y * DerivPhi4y + DerivPhi3z * DerivPhi4z )
    
    K44 = Vol * ( DerivPhi4x * DerivPhi4x + DerivPhi4y * DerivPhi4y + DerivPhi4z * DerivPhi4z )
    
    K_el = np.array(([K11, K12, K13, K14],
                     [K12, K22, K23, K24],
                     [K13, K23, K33, K34],
                     [K14, K24, K34, K44]))
    
    # Assemblage de la matrice globale    
    K[np.ix_(N, N)] += K_el

#=============================================================================#
# Second membre
#=============================================================================#
# Assemblage du vecteur de charge B
B = np.zeros(NoN) # Initialisation
for i in range(number_tetrahedra): # parcourt les éléments tétraèdriques
    #====================================
    # Selection des sommets composants le tétraèdre i
    
    # Indices des sommets composants le tétraèdre courant
    N = tetrahedra[i, :4] # séléctionne les trois colonnes de la i-ème ligne de la matrice tetrahedra
    
    # Coordonnee des sommets composants le tétraèdre
    sommets = nodes[N]  # Matrice (4x3) dont les lignes sont les coordonées (x,y,z) des sommets
    
    x1, y1, z1 = sommets[0] # coorodnnées du sommet N1
    x2, y2, z2 = sommets[1] # coorodnnées du sommet N2
    x3, y3, z3 = sommets[2] # coorodnnées du sommet N3
    x4, y4, z4 = sommets[3] # coorodnnées du sommet N4
    
    # Volume du tétraèdre courant
    Vol = abs( ( (1/6) * (-x2 * y3 * z1 + x2 * y4 * z1 + x1 * y3 * z2 - x1 * y4 * z2 + x2 * y1 * z3 - 
   x1 * y2 * z3 + x1 * y4 * z3 - x2 * y4 * z3 + 
   x4 * (-y2 * z1 + y3 * z1 + y1 * z2 - y3 * z2 - y1 * z3 + y2 * z3) + (-x2 * y1 + 
      x1 * y2 - x1 * y3 + x2 * y3) * z4 + 
   x3 * (-y4 * z1 - y1 * z2 + y4 * z2 + y2 * (z1 - z4) + y1 * z4))
           ) )
    
    #====================================
    # Assemblage du vecteur élémentaire
    B_el = (Vol/ 4) * np.ones(4)
    
    # Assemblage du vecteur global
    B[N] += B_el

#=============================================================================#
# Prise en compte des conditions aux limites
#=============================================================================#
tol = 1e-10
interior_nodes = []

boundary_nodes_gamma_1 = []
boundary_nodes_gamma_2 = []
boundary_nodes_gamma_3 = []
boundary_nodes_gamma_4 = []
boundary_nodes_gamma_5 = []
boundary_nodes_gamma_6 = []

for i, (xi, yi, zi) in enumerate(nodes):
    is_boundary = False  # On suppose d'abord que le nœud est intérieur
    if (yi > ly - tol):
        boundary_nodes_gamma_1.append(i)
        is_boundary = True
    if (yi < tol):
        boundary_nodes_gamma_2.append(i)
        is_boundary = True
    if (xi < tol):
        boundary_nodes_gamma_3.append(i)
        is_boundary = True
    if (xi > lx - tol):
        boundary_nodes_gamma_4.append(i)
        is_boundary = True
    if (zi > lz - tol):
        boundary_nodes_gamma_5.append(i)
        is_boundary = True
    if (zi < tol):
        boundary_nodes_gamma_6.append(i)
        is_boundary = True
    if not is_boundary:
        interior_nodes.append(i)

# Prise en compte des conditions de Neumann :
    # La contribution de Neumann se répartit uniformément sur les domaines de bords
B[boundary_nodes_gamma_1] += g_1/ (Nx * Nz)
B[boundary_nodes_gamma_2] += g_2/ (Nx * Nz)
B[boundary_nodes_gamma_3] += g_3/ (Ny * Nz)
B[boundary_nodes_gamma_4] += g_4/ (Ny * Nz)
B[boundary_nodes_gamma_5] += g_5/ (Nx * Ny)
B[boundary_nodes_gamma_6] += g_6/ (Nx * Ny)

#=============================================================================#
# Résolution du système linéaire
#=============================================================================#

# Normalisation pour un problème de Neumann bien posé
K[0, :] = np.ones(NoN) / NoN
B[0] = 0 # ici on impose que la moyenne de u est 1

K_sparse = csr_matrix(K)
u = spsolve(K_sparse, B)

# =============================================================================
# Affichage de la solution
# =============================================================================

# Construction du tableau de connectivité pour Pyvista
cells = []
for tet in tetrahedra:
    cells.append(4)  # Nombre de nœuds par tétraèdre
    cells.extend(tet.tolist())
cells = np.array(cells)

# Définition du type de cellule : VTK_TETRA a le code 10
celltypes = np.full(tetrahedra.shape[0], 10)

# Création du maillage non structuré Pyvista
grid = pv.UnstructuredGrid(cells, celltypes, nodes)

# Ajout de la solution u en tant que donnée scalaire associée aux nœuds
grid["u"] = u

# Création du plotter
plotter = pv.Plotter()

# Génération des isosurfaces
n_contours = 500  # Nombre d'isosurfaces : 100 pour un continuum
contours = grid.contour(n_contours, scalars="u")

plotter.add_mesh(
     contours,
     cmap="inferno",
     opacity=0.7,  # Transparence pour mieux voir les structures internes 0.6
     specular=0.1,
     specular_power=10,
     ambient=0.8,
     diffuse=0.9
)

# # Génération des coupes orthogonales
# slices = grid.slice_orthogonal()

# plotter.add_mesh(
#     slices,
#     cmap="inferno",  # Palette différente pour distinguer des isosurfaces
#     opacity=0.7,
#     show_edges=False
# )

# Ajout de la barre des couleurs
#plotter.add_scalar_bar(title="Solution u")

# Affichage
plotter.set_background("gray")
plotter.show()
