#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 17:38:37 2025

@author: nicolas-fournie
"""

"""
Equation de Poisson 3D : -Δu(x,y,z) = f(x,y,z) sur Ω = [0, lx]x[0, ly]x[0, lz] ;
avec problème de Dirichlet homogene : u = 0 sur ∂Ω
"""

import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix
import pyvista as pv

#=============================================================================#
# Définition de la fonction source f(x,y,z)
#=============================================================================#

# fonction sinus
def f(x, y, z):
    n = 2
    k = n * np.pi
    rho = 10
    return rho * np.sin(k * x) * np.sin(k * y) * np.sin(k * z)

#=============================================================================#
# Définition du maillage
#=============================================================================#

Nx, Ny, Nz = 20, 20, 20  # Nombre de divisions dans chaque direction
lx, ly, lz = 1, 1, 1 # Longueur des axes x, y et z

x = np.linspace(0, lx, Nx + 1)
y = np.linspace(0, ly, Ny + 1)
z = np.linspace(0, lz, Nz + 1)

X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# Coordonnées des noeuds
nodes = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
                                             
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

number_nodes = (Nx + 1) * (Ny + 1) * (Nz + 1)

#=============================================================================#
# Assemblage de la matrice de rigidité K
#=============================================================================#
K = np.zeros((number_nodes, number_nodes)) # Initialisation
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
# Assemblage du vecteur de charge B
#=============================================================================#

B = np.zeros(number_nodes) # Initialisation

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
    #====================================
    
    # Coordonées des points d'intégrations en coordonées cartésiennes sur le tétraèdre de référence
    P1 = np.array([1/4 + np.sqrt(5)/10, 1/4 - np.sqrt(5)/10, 1/4])
    P2 = np.array([1/4 - np.sqrt(5)/10, 1/4 + np.sqrt(5)/10, 1/4])
    P3 = np.array([1/4 - np.sqrt(5)/10, 1/4, 1/4 + np.sqrt(5)/10])
    P4 = np.array([1/4, 1/4, 1/4 + np.sqrt(5)/10, 1/4 - np.sqrt(5)/10])
    
    # Coordonnée des points d'intégrations en coordonnées cartésiennes sur le tétraèdre courant (quelconque)
    P1Tx = x1 * ( 1 - P1[0] - P1[1] - P1[2] ) + x2 * P1[0] + x3 * P1[1] + x4 * P1[2]
    P1Ty = y1 * ( 1 - P1[0] - P1[1] - P1[2] ) + y2 * P1[0] + y3 * P1[1] + y4 * P1[2]
    P1Tz = z1 * ( 1 - P1[0] - P1[1] - P1[2] ) + z2 * P1[0] + z3 * P1[1] + z4 * P1[2]
    
    P2Tx = x1 * ( 1 - P2[0] - P2[1] - P2[2] ) + x2 * P2[0] + x3 * P2[1] + x4 * P2[2]
    P2Ty = y1 * ( 1 - P2[0] - P2[1] - P2[2] ) + y2 * P2[0] + y3 * P2[1] + y4 * P2[2]
    P2Tz = z1 * ( 1 - P2[0] - P2[1] - P2[2] ) + z2 * P2[0] + z3 * P2[1] + z4 * P2[2]
    
    P3Tx = x1 * ( 1 - P3[0] - P3[1] - P3[2] ) + x2 * P3[0] + x3 * P3[1] + x4 * P3[2]
    P3Ty = y1 * ( 1 - P3[0] - P3[1] - P3[2] ) + y2 * P3[0] + y3 * P3[1] + y4 * P3[2]
    P3Tz = z1 * ( 1 - P3[0] - P3[1] - P3[2] ) + z2 * P3[0] + z3 * P3[1] + z4 * P3[2]
    
    P4Tx = x1 * ( 1 - P4[0] - P4[1] - P4[2] ) + x2 * P4[0] + x3 * P4[1] + x4 * P4[2]
    P4Ty = y1 * ( 1 - P4[0] - P4[1] - P4[2] ) + y2 * P4[0] + y3 * P4[1] + y4 * P4[2]
    P4Tz = z1 * ( 1 - P4[0] - P4[1] - P4[2] ) + z2 * P4[0] + z3 * P4[1] + z4 * P4[2]
    
    # Assemblage du vecteur élémentaire
    B_el = (f(P1Tx, P1Ty, P1Tz) + f(P2Tx, P2Ty, P2Tz) + f(P3Tx, P3Ty, P3Tz) + f(P4Tx, P4Ty, P4Tz)) * (Vol/ 4) * np.ones(4)
    
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


#=============================================================================#
# Résolution du système linéaire
#=============================================================================#
K_int = K[np.ix_(interior_nodes, interior_nodes)]
B_int  = B[interior_nodes]

K_sparse = csr_matrix(K_int)
u_int = spsolve(K_sparse, B_int)

# Ajout des conditions de Dirichlet homogène
u = np.zeros(number_nodes)
u[interior_nodes] = u_int

# =============================================================================
# Affichage de la solution
# =============================================================================

# Construction du tableau de connectivité pour Pyvista
# Pour chaque tétraèdre, on crée une séquence [4, n0, n1, n2, n3]
cells = []
for tet in tetrahedra:
    cells.append(4)         # nombre de noeuds pour un tétraèdre
    cells.extend(tet.tolist())
cells = np.array(cells)

# Définition du type de cellule : VTK_TETRA a le code 10
celltypes = np.full(tetrahedra.shape[0], 10)

# Création du maillage non structuré Pyvista
grid = pv.UnstructuredGrid(cells, celltypes, nodes)

# Ajout de la solution u en tant que donnée scalaire associée aux noeuds
grid["u"] = u

# Affichage du maillage et de la solution
plotter = pv.Plotter()

plotter.add_mesh(
    grid,
    scalars="u",
    cmap="inferno",
    opacity=0.4,
    show_edges=False,
    line_width=0.5,
    specular=0.6,
    specular_power=10,
    ambient=0.3,
    diffuse=1)


#plotter.add_scalar_bar(title="Solution u")
plotter.show()