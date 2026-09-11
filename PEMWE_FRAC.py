#!/usr/bin/env python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
mod13.py: Phase-Field Fracture Simulation for PEM Water Electrolyzers

This script solves a coupled electro-chemo-thermo-mechanical problem using 
FEniCSx (DOLFINx) to simulate fracture evolution, fluid transport, and potential 
distributions within a Proton Exchange Membrane Water Electrolyzer (PEMWE).

Authors:
    Sidharth PC <sidharthpccalicut@gmail.com>
    Alberto Antonini (Co-author)

Supervision:
    Prof. Dr.-Ing. Fadi Aldakheel

Affiliation:
    Institute of Mechanics and Computational Mechanics (IBNM)
    Gottfried Wilhelm Leibniz Universität Hannover (LUH)
    Appelstraße 11, 30167 Hannover, Germany

Dependencies:
    - DOLFINx (dolfinx)
    - UFL (Unified Form Language)
    - PETSc / petsc4py
    - MPI (mpi4py)
"""

# Import the fundamental libraries
from dolfinx import plot, nls, io, geometry
from dolfinx.fem import Constant, dirichletbc, Function, FunctionSpace, form, locate_dofs_topological, Expression, assemble_scalar ,locate_dofs_geometrical
from dolfinx.io import XDMFFile
from dolfinx import default_real_type, fem
from dolfinx.fem.petsc import NonlinearProblem, LinearProblem
from dolfinx.mesh import create_rectangle ,CellType, locate_entities_boundary, locate_entities ,meshtags
from dolfinx.log import set_log_level, LogLevel
# from ufl import  FiniteElement, VectorElement, TensorElement, MixedElement, TestFunctions, TrialFunction, Measure, FacetNormal, EnrichedElement
from ufl import  div, grad, dot, inner, derivative, split, Identity, dx, SpatialCoordinate, exp, sin, sym, variable,diff, as_matrix, as_vector
from ufl import conditional, lt, max_value
from petsc4py import PETSc
from petsc4py.PETSc import ScalarType
from mpi4py import MPI
import time
import numpy as np
# import meshio
import gmsh
# import pyvista
import basix
import os
import ufl
from basix.ufl import element, mixed_element

proc = MPI.COMM_WORLD.rank
# from ufl import  FiniteElement, VectorElement, TensorElement, MixedElement
from ufl import TestFunctions, TrialFunction, Measure, FacetNormal, TestFunction


# import os
# os.chdir("/mnt/f/PartSat/Mod 57/Displacement/7.5/")
# print("Working directory:", os.getcwd())

# ---------------------------
# Geometry parameters
# ---------------------------
x0, y0 = 0.0, 0.0
Ly = 0.1          # Total height
Lx1 = 0.01        # ACL Width
Lx2 = 0.180       # Membrane Width
Lx3 = 0.01        # CCL Width

x1 = x0 
x2 = x1 + Lx1
x3 = x2 + Lx2
x4 = x3 + Lx3

# ---------------------------
# Calibrated Sizing Parameters for ~50,000 Elements
# ---------------------------
lc_notch = 0.0001       # Fine size at notch tips (kept for ACL left edge consistency)
lc_cl = 0.0001          # Coarse size at the right edge of ACL
lc_interface = 0.0005   # Interface fine size (matches ACL right edge)
lc2 = 0.0090            # Coarse size at the exact center of Membrane (V-profile peak)
lc_cl2 = 0.0025         # Coarse size for CCL outer boundary

# ---------------------------
# Define Flat Base Geometry
# ---------------------------
n_parts = 7
h = Ly / (n_parts - 1)
dy = [h/2] + [h]*(n_parts-2) + [h/2]


with XDMFFile(MPI.COMM_WORLD, "fibre_2D_127micron.xdmf", "r") as xdmf:
    mesh = xdmf.read_mesh(name="Grid")
    ct = xdmf.read_meshtags(mesh, name="Grid")

# 1. Define dimensions first
tdim = mesh.topology.dim
fdim = mesh.topology.dim - 1

# 2. Build topology connectivity BEFORE reading facet tags or locating entities
mesh.topology.create_entities(fdim)
mesh.topology.create_connectivity(fdim, tdim)
mesh.topology.create_connectivity(tdim, fdim)

# 3. Now safely read facet meshtags
with XDMFFile(MPI.COMM_WORLD, "fibre_2D_127micron_mt.xdmf", "r") as xdmf:
    ft = xdmf.read_meshtags(mesh, name="Grid")

import dolfinx.mesh as dmesh

# -------------------------------------------------------------
# Dynamic Geometrical Selection for Unstable Boolean Boundaries
# -------------------------------------------------------------
# Tolerance buffer for geometry mismatch
tol = 1e-5

# 1. Right Boundary: Located exactly at x = x4
right_facets = dmesh.locate_entities_boundary(
    mesh, fdim, lambda x: np.isclose(x[0], x4, atol=tol)
)

# 2. Bottom Boundary: Located exactly at y = y0
bottom_facets = dmesh.locate_entities_boundary(
    mesh, fdim, lambda x: np.isclose(x[1], y0, atol=tol)
)

# 3. Top Boundary: Located exactly at y = Ly
top_facets = dmesh.locate_entities_boundary(
    mesh, fdim, lambda x: np.isclose(x[1], Ly, atol=tol)
)

# 4. Left Boundaries (Divided into Compression segments vs Flux segments)
# Instead of searching line numbers, we replicate the "i % 2 == 0" logic based on coordinates
left_all_facets = dmesh.locate_entities_boundary(
    mesh, fdim, lambda x: np.isclose(x[0], x0, atol=tol)
)

left_facets = []
left_flux_facets = []

for facet in left_all_facets:
    # Get the midpoint coordinates of the facet to check its position along y
    facet_coords = mesh.geometry.x[mesh.topology.connectivity(fdim, 0).links(facet)]
    y_mid = np.mean(facet_coords[:, 1])
    
    # Trace through your original parts height matrix 'dy'
    y_check = y0
    for i, d in enumerate(dy):
        if (y_check - tol) <= y_mid <= (y_check + d + tol):
            if i % 2 == 0:
                left_flux_facets.append(facet)  # Maps to old Left_Odd (10)
            else:
                left_facets.append(facet)       # Maps to old Left_Even (11)
            break
        y_check += d

# Convert to structured numpy arrays required by dolfinx
left_facets = np.array(left_facets, dtype=np.int32)
left_flux_facets = np.array(left_flux_facets, dtype=np.int32)

# --- Keep the rest of your mapping tags matching exactly ---
marked_facets = np.hstack([bottom_facets, top_facets, right_facets, left_facets, left_flux_facets]).astype(np.int32)
marked_values = np.hstack([
    np.full_like(bottom_facets, 1), 
    np.full_like(top_facets, 2), 
    np.full_like(right_facets, 3), 
    np.full_like(left_facets, 4),                     
    np.full_like(left_flux_facets, 5)
]).astype(np.int32)

sorted_facets = np.argsort(marked_facets)
facet_tag = meshtags(mesh, fdim, marked_facets[sorted_facets], marked_values[sorted_facets])

ds = Measure("ds", domain=mesh, subdomain_data=facet_tag)
dx = Measure("dx", domain=mesh)

# ### 2. Define elements and the time for the mixed problem 

P1 = element("CG", mesh.basix_cell(), 1, dtype=default_real_type)
P2 = element("CG", mesh.basix_cell(), 2, shape=(mesh.geometry.dim,), dtype=default_real_type)

funch = fem.functionspace(mesh, ("DG",0))           # Functionspace for the history variable H

mixed_element1 = mixed_element([P1, P1])
V_pot = fem.functionspace(mesh, mixed_element1)

V_dam = fem.functionspace(mesh, P1)

mixed_element2 = mixed_element([P1, P1])
V_tsp = fem.functionspace(mesh, mixed_element2)
# V_tsp = fem.functionspace(mesh, P2)

V_mech = fem.functionspace(mesh, P2)

u1_m = Function(V_pot)
u2_m = Function(V_tsp)
u3_m = Function(V_dam)
u4_m = Function(V_mech)

dc_H = Function(funch)
temp_dc = Function(funch)

# Vpe, Vpe_nodes = V_pot.sub(0).collapse()[0], V_pot.sub(0).collapse()[1] 
# Vpp, Vpp_nodes = V_pot.sub(1).collapse()[0], V_pot.sub(1).collapse()[1]
# VuS, VuS_nodes = V_tsp.sub(0).collapse()[0], V_tsp.sub(1).collapse()[1]

Vpe, Vpe_nodes = V_pot.sub(0).collapse()[0], V_pot.sub(0).collapse()[1] 
Vpp, Vpp_nodes = V_pot.sub(1).collapse()[0], V_pot.sub(1).collapse()[1]
Vpl, Vpl_nodes = V_tsp.sub(0).collapse()[0], V_tsp.sub(0).collapse()[1]
Vpg, Vpg_nodes = V_tsp.sub(1).collapse()[0], V_tsp.sub(1).collapse()[1]
# VuS, VuS_nodes = V_tsp.sub(2).collapse()[0], V_tsp.sub(2).collapse()[1]

pe_m, pp_m = split(u1_m)  
pl_m, pg_m = split(u2_m) 
# pl_m, pg_m, _ = ufl.split(u2_m)
# pl_m = u2_m.sub(0)
# pg_m=  u2_m.sub(1)
# uS_m = u2_m.sub(2)

uS_m = u4_m
d_m = u3_m

test_pe, test_pp  = TestFunctions(V_pot)
test_pl, test_pg = TestFunctions(V_tsp)
test_uS = TestFunction(V_mech)
test_d  = TestFunction(V_dam)

#time for simulation
t = 0
dt = 0.01
t_max = 1.1
t_num = t_max/dt
print(int(t_num))

# ### 3. Boundary conditions

#Assign the dirichlet boundary conditions
from dolfinx.fem import set_bc
from petsc4py.PETSc import ScalarType

def update_variable_linear(t,initial_value,final_value,time_to_start,time_to_constant):
    if t < time_to_start.value:
        return initial_value.value
    elif t < time_to_constant.value:
        return initial_value.value + (final_value.value - initial_value.value) * ((t - time_to_start.value) / (time_to_constant.value - time_to_start.value))
    else:
        return final_value.value

def update_variable_nonlinear(t, initial_value, final_value, time_to_start, time_to_constant, power):
    t_val = float(t) # Ensure t is treated as a float
    t_start = time_to_start.value
    t_end = time_to_constant.value
    v_start = initial_value.value
    v_end = final_value.value

    if t_val < t_start:
        return v_start
    elif t_val < t_end:
        # Calculate the normalized progress (0.0 to 1.0)
        progress = (t_val - t_start) / (t_end - t_start)
        
        # Apply power to slow down the start and speed up the end
        curved_progress = progress ** power
        
        return v_start + (v_end - v_start) * curved_progress
    else:
        return v_end
    
initial_value_current = Constant(mesh, 0.0001)
final_value_current = Constant(mesh, 0.02)
current_value = Constant(mesh, ScalarType(0.0001))

initial_value_electro_switch = Constant(mesh, 0.0)
final_value_electro_switch = Constant(mesh, 1.0)
electro_switch = Constant(mesh, 0.0)

initial_value_disp = Constant(mesh, 0.0)
final_value_disp = Constant(mesh, x4 * 0.1)        # 10% of total thickness of the cell
disp_value = Constant(mesh, 0.0)

initial_value_tx1 = Constant(mesh, 0.01) 
final_value_tx1 = Constant(mesh, 0.01+139) 
tx1_value = Constant(mesh, 0.01) 

time_to_start_current = Constant(mesh, 1.0)
time_to_constant_current = Constant(mesh, 1.5)

time_to_start_disp = Constant(mesh, 0.03)
time_to_constant_disp = Constant(mesh, 1.0)

time_to_start_tx1 = Constant(mesh, 0.03)
time_to_constant_tx1 = Constant(mesh, 1.0)
# #-----------------------------------------------------------------------------------------------------------------------------------------

pe_dof_right     =  locate_dofs_topological(V_pot.sub(0),fdim,right_facets)
pl_dof_right     =  locate_dofs_topological((V_tsp.sub(0)),fdim,right_facets)
pl_dof_left_flux =  locate_dofs_topological((V_tsp.sub(0)),fdim,left_flux_facets)
pg_dof_left_flux =  locate_dofs_topological((V_tsp.sub(1)),fdim,left_flux_facets)
pg_dof_right     =  locate_dofs_topological((V_tsp.sub(1)),fdim,right_facets)
u_dof_top        =  locate_dofs_topological(V_mech.sub(1),fdim, top_facets)
u_dof_bottom     =  locate_dofs_topological(V_mech.sub(1),fdim, bottom_facets)
u_dof_left_disp  =  locate_dofs_topological(V_mech.sub(0),fdim, left_facets)
u_dof_right_x      =  locate_dofs_topological(V_mech.sub(0),fdim, right_facets)
u_dof_right_y      =  locate_dofs_topological(V_mech.sub(1),fdim, right_facets)

bc_pe_right    = dirichletbc(ScalarType(0),pe_dof_right, V_pot.sub(0))
bc_pl_right    = dirichletbc(ScalarType(0.1060),pl_dof_right, V_tsp.sub(0))
bc_pg_right    = dirichletbc(ScalarType(0.1),pg_dof_right, V_tsp.sub(1))
bc_pl_left     = dirichletbc(ScalarType(0.1),pl_dof_left_flux, V_tsp.sub(0))
bc_pg_left     = dirichletbc(ScalarType(0.1013),pg_dof_left_flux, V_tsp.sub(1)) 
bc_uD_top      = dirichletbc(ScalarType(0),u_dof_top,V_mech.sub(1))
bc_uD_right_x  = dirichletbc(ScalarType(0),u_dof_right_x,V_mech.sub(0))
bc_uD_right_y  = dirichletbc(ScalarType(0),u_dof_right_y,V_mech.sub(1))
bc_uD_bottom   = dirichletbc(ScalarType(0),u_dof_bottom,V_mech.sub(1))
bc_uD_left_disp= dirichletbc(disp_value,u_dof_left_disp,V_mech.sub(0))        

bc_pot = [
    bc_pe_right,
    # bc_pe_right2,
]

bc_tsp = [
    bc_pl_left,
    bc_pl_right,
    bc_pg_left,
    bc_pg_right,
]

bc_mech = [
    bc_uD_right_x,
    bc_uD_right_y,
]

def e_rev(T, dh, ds):
    return (dh - T.value*ds)/(2*96485)

teta = Constant(mesh, 353.15) #K
R_joule = Constant(mesh, 8.3144) #J/mol K
dh_a = 285.83e3 # J mol-1
dh_c = 0.0      # J mol-1
ds_a = 163.3    # J mol-1 K-1
ds_c = -0.104   # J mol-1 K-1
Eact_a = 4e4    # J mol-1
Eact_c = 2e4    # J mol-1
E_sigma = 8e3   # J mol-1
frac_cl = 0.3
lambda_m = 16

U0_a = e_rev(teta, dh_a, ds_a)
U0_c = e_rev(teta, dh_c, ds_c)

# fv = (20 * 18.02) / (lambda_m * 18.02 + (1100/1.98))
# kappa_pa = frac_cl**1.5 * 110.0 * (0.4101 - 0.06)**1.5 * exp(((1/353.15)-(1/teta.value)) * E_sigma/R_joule.value) * 1e-3
# kappa_pm = 110.0 * (fv - 0.06)**1.5 * exp(((1/353.15)-(1/teta.value)) * E_sigma/R_joule.value) * 1e-3

kappa_pa = frac_cl**1.5 * (0.5139 * lambda_m - 0.326) * exp(((1/303)-(1/teta.value)) * 1268) * 1e-3
kappa_pm = (0.5139 * lambda_m - 0.326) * exp(((1/303)-(1/teta.value)) * 1268) * 1e-3
Ja0_a = 2.1e-2 * exp(((1/353.15)-(1/teta.value)) * Eact_a/R_joule.value) * 1e-3
Ja0_c = 2.1e2 * exp(((1/353.15)-(1/teta.value)) * Eact_c/R_joule.value) * 1e-3
# Ja0_a = 7e-6
# Ja0_c = 7e-2
# kappa_eptl = 1.25
# # kappa_ecl  = 2.16
# kappa_ecl  = 22.2e-3

# print(fv)
print(U0_a)
print(U0_c)
print(kappa_pa)
print(kappa_pm)
print(Ja0_a)
print(Ja0_c)

Q = fem.functionspace(mesh, ("DG", 0))

cells_D2 = ct.find(5)   # aCL (Titanium)
cells_D3 = ct.find(6)   # Membrane (Insulator)
cells_D4 = ct.find(7)   # cCL (Titanium)

# def set_subdomain_values(func, d2_val, d3_val, d4_val):
#     indices = np.concatenate([cells_D2, cells_D3, cells_D4])
#     values = np.repeat([d2_val, d3_val, d4_val], [len(cells_D2), len(cells_D3), len(cells_D4)])
#     func.x.array[indices] = values

def set_subdomain_values(func, d2_val, d3_val, d4_val):
    # 1. Gather the global mapping sizes
    index_map = mesh.topology.index_map(mesh.topology.dim)
    num_cells = index_map.size_local + index_map.num_ghosts
    local_values = np.zeros(num_cells, dtype=np.float64)
    
    # 2. Populate the entire array context safely
    local_values[cells_D2] = d2_val
    local_values[cells_D3] = d3_val
    local_values[cells_D4] = d4_val
    
    # 3. FIXED: Assign the full length to match the total local + ghost array footprint
    func.x.array[:] = local_values
    assert len(func.x.array) == num_cells
    # 4. Synchronize the boundary values across all active parallel CPUs
    func.x.scatter_forward()

# Create all functions at once
kappa_eb, kappa_p, lame1, lame2, reaction_marker, U0, Ja0_field, a2, a_ox_red, KS0, o2_marker, h2_marker, mem_marker, cov_exp, nS_0S, theta, frac_ion, Mg, sigma_st, Gc = [fem.Function(Q) for _ in range(20)]
# Assign values
#                                           aPTL        aCL         MEM         cCL         cPTL
# set_subdomain_values(kappa_e,           kappa_eptl, kappa_ecl,  1e-10,      kappa_ecl   , kappa_eptl) # PTLs are good conductors of electrons, but MEM is a bad conductor
# set_subdomain_values(kappa_e,           1.250,      350e-3,     350e-10,    350e-3      , 1.250     )
set_subdomain_values(kappa_eb,                      1.2745,     4.4e-10,    1.2745                  ) # Bulk electronic conductivity ; back calculated from Debe et al.
set_subdomain_values(kappa_p,                       kappa_pa,   kappa_pm,   kappa_pa                ) # PTLs are bad conductors of protons, but CLs and MEM are good conductors
set_subdomain_values(lame1,                         143.65,     94.83,      143.65                  ) 
set_subdomain_values(lame2,                         95.77,      92.58,      95.77                   ) 
set_subdomain_values(reaction_marker,               1.0,        0.0,        1.0                     )    
set_subdomain_values(U0,                            U0_a,       0.0,        U0_c                    )
set_subdomain_values(Ja0_field,                     Ja0_a,      0.0,        Ja0_c                   )
set_subdomain_values(a2,                            1.0,        0.0,        1.0                     )
set_subdomain_values(a_ox_red,                      0.5,        0.0,        0.5                     )
set_subdomain_values(KS0,                           4e-8,       1e-12,      4e-8                    )
set_subdomain_values(o2_marker,                     1.0,        0.0,        0.0                     )
set_subdomain_values(h2_marker,                     0.0,        0.0,        1.0                     )
set_subdomain_values(mem_marker,                    0.0,        1.0,        0.0                     )
set_subdomain_values(cov_exp,                       2.0,        0.0,        0.0                     ) # Coverage exponent for CLs
set_subdomain_values(nS_0S,                         0.65,       0.61,       0.65                    ) # Solidity
set_subdomain_values(theta,                         1.3962,     1.3962,     1.658                   ) # Contact angle
set_subdomain_values(frac_ion,                      frac_cl,    1.0,        frac_cl                 ) # Ionomer fraction
set_subdomain_values(Mg,                            32.0e-3,    32.0e-3,    2.016e-3                ) # Ionomer fraction
set_subdomain_values(sigma_st,                      1.0,        20.0,       10.0                    ) # Critical Stress for damage , Taken from Fadi Electro-chemo blah blah
set_subdomain_values(Gc,                            0.002,      0.01,        1.0                    ) # Critical Stress for damage , Taken from Fadi Electro-chemo blah blah

etaFR = Constant(mesh, 3.55e-10) #N*s/mm2
etaGR = Constant(mesh, 1.881e-11) #N*s/mm2
# # fluid volume fraction @ zero strain
nF_0S = fem.Function(Q)
nF_0S.x.array[:] = 1.0 - nS_0S.x.array[:]
F = Constant(mesh, 96485.0) #C/mol
# z_fc = Constant(mesh, -1.0) # valence of negative charges
z_proton = Constant(mesh, 1.0) # proton valence
surf_h2o = Constant(mesh, 63.5e-6) #N/mm
# MmH = Constant(mesh, 1.008e-3) #kg/mol molar weight proton
R = Constant(mesh, 8314.4) #(N*mm)/(mol*K)                                                    8314,4 (N*mm)/(mol*K)
Mo2 = Constant(mesh, 32.0e-3)                   # kg/mol oxygen
Mh2 = Constant(mesh, 2.016e-3)
Mwater = Constant(mesh, 18.0e-3)                # kg/mol water
Mproton = Constant(mesh, 1.008e-3)
rho_water = Constant(mesh, 1e-6)        # kg/mm3
Ja0_scale = Constant(mesh, 2.5)        # kg/mm3

eta_res = Constant(mesh, 1.0e-6)                                           # >>> Added residual stiffness for degradation function <<<
ls_pfm = Constant(mesh, lc_notch * 4.0)  
kappa_s = lame1+(2/3)*lame2

# ### 4. Define the stress tensor and the deformation dependent permeability


def grad_2D_vector(u):
    return as_vector((u.dx(0), u.dx(1), 0))
    
def grad_2D(u):
    return as_matrix([[u[0].dx(0), u[0].dx(1), 0],
                      [u[1].dx(0), u[1].dx(1), 0],
                      [0, 0, 0]])

def eps_s(u,dim):
    #  strain tensor
    eps_s_ = ScalarType(0.5) * (grad(u) + grad(u).T)
    return eps_s_

def eps_sd(u,dim):
    # Deviatoric strain tensor
    eps_s = variable(ScalarType(0.5) * (grad(u) + grad(u).T))
    eps_sd_ = (eps_s - ScalarType(1/3)*ufl.tr(eps_s)*Identity(dim))
    return variable(eps_sd_)

def tr_eps(u):
    eps_s = ScalarType(0.5) * (grad(u) + grad(u).T)
    tr_eps_ = (ufl.tr(eps_s))
    return tr_eps_

def ptr_es(u):
    return ufl.max_value(tr_eps(u), ScalarType(0))

def ntr_es(u):
    return ufl.min_value(tr_eps(u), ScalarType(0))

import math
from ufl import exp
from ufl import conditional, gt

# #fluid volume fractuion
def nF(uS,d,dim):
    from ufl import div, variable, det
    nF_raw =  (ScalarType(1.0) - (ScalarType(1.0)-d) * nS_0S *(ScalarType(1.0)-div(uS))) #small strains
    nF_ = conditional(lt(nF_raw, 1e-6), 1e-2, nF_raw)
    return nF_

# deformation dependend permeability
def KS(uS, d, dim):
    from ufl import div, variable
    perm_power = ScalarType(1.0)
    KS_ = KS0 * (nF(uS,d,dim)/nF_0S)**perm_power
    # print(f'KS(uS) = {KS_}')
    return KS_

def rho_g(pg):
    p_ref = ScalarType(0.0)     # When using absolute pressure
    # p_ref = ScalarType(0.1)       # When using excess pressure                
    rho_g_ = (pg+p_ref)/(R*teta)*Mg
    return rho_g_


def Ja(phi_e,phi_p,sl):
    overpot = phi_e - phi_p - U0
    limit = ScalarType(0.8)
    safe_overpot = ufl.max_value(-limit, ufl.min_value(limit, overpot))
    # safe_overpot = overpot
    const_term = F/(R_joule*teta)
    term_oer = ScalarType(2) * a_ox_red
    term_her = ScalarType(2) * (a_ox_red - ScalarType(1))
    Ja_ = reaction_marker * sl**cov_exp * Ja0_scale * Ja0_field * ( exp(term_oer*safe_overpot*const_term) - exp(term_her*safe_overpot*const_term))         # A/m2
    return Ja_

# def sigma_plain(u,dim):
#     sig_d_plus_ = ScalarType(2)*lame2*eps_s(u,dim) + lame1*tr_eps(u,dim)*Identity(dim)             # # kappa_s and lame_2 function of pc
#     return sig_d_plus_   

eps_sat = ScalarType(1e-6)
def WL_vec(u,dim,s,p):
    WL_term = -KS(u,dim)/etaFR*(grad(p))/(nF(u,tdim)*(s+eps_sat))
    return WL_term

def WG_vec(u,dim,s,pg):
    WG_term = -KS(u,dim)/etaGR*(grad(pg))/(nF(u,dim)*(ScalarType(1)-(s+eps_sat)))
    return WG_term

def g_deg(d):                                                                                                               # >>> Added degradation function <<<
    g_deg_ = ( ((ScalarType(1) - eta_res.value ) * (ScalarType(1) - d)**ScalarType(2) ) + eta_res.value )                                                                               # >>> Added degradation function <<<
    return g_deg_  

def sigma_se(u,d,dim):                                                                                                      # >>> Addded Effective solid stress tensor <<<
    eps = eps_sd(u, dim)   # Strain tensor
    sigma = g_deg(d) * (kappa_s * ptr_es(u) * Identity(dim) + ScalarType(2) * lame2 * eps) \
            + kappa_s * ntr_es(u) * Identity(dim)
    return sigma

def smooth_H(g, width):
    # smooth approximation of Heaviside(g); width controls sharpness
    return 0.5 * (1.0 + ufl.tanh(g / width))

dp_blend_width = fem.Constant(mesh, ScalarType(0.05)) * kappa_s   # TUNE: pick from the g1/g2 range diagnostic, not blindly

def sigma_dp(u,d,dim):
    def sigma_dp(u, d, dim):
    B_dp = fem.Constant(mesh, -0.26)
    C_dp = 1.0 / (18.0 * B_dp**2 * kappa_s + 2.0 * lame2)
    stab_dp = 1e-12
    eps_dp = ufl.sym(ufl.grad(u))
    I = ufl.Identity(dim)
    I1 = ufl.tr(eps_dp)
    eps_dev = eps_dp - (1.0/3.0) * I1 * I
    J2 = 0.5 * ufl.inner(eps_dev, eps_dev)
    sqrt_J2 = ufl.sqrt(J2 + stab_dp)

    # --- switching quantities, both now stress-scaled so they're consistent with `width` ---
    g1 = kappa_s * (I1 + 6.0 * B_dp * sqrt_J2)                      # FIXED: scaled by kappa_s (strain -> stress units)
    g2 = 2.0 * lame2 * sqrt_J2 - 3.0 * B_dp * kappa_s * I1          # already stress-scale, unchanged

    width = dp_blend_width
    H1 = 0.5 * (1.0 + ufl.tanh(g1 / width))   # ~1 in region 1, ~0 otherwise
    H2 = 0.5 * (1.0 + ufl.tanh(g2 / width))   # ~1 in region 2, ~0 in region 3

    psi_s_r1 = 0.0
    psi_d_r1 = 0.5 * kappa_s * I1**2 + 2.0 * lame2 * J2
    psi_s_r2 = C_dp * kappa_s * lame2 * (I1 + 6.0 * B_dp * sqrt_J2)**2
    psi_d_r2 = C_dp * (-3.0 * B_dp * kappa_s * I1 + 2.0 * lame2 * sqrt_J2)**2
    psi_s_r3 = 0.5 * kappa_s * I1**2 + 2.0 * lame2 * J2
    psi_d_r3 = 0.0

    # same nesting as before, but blended instead of switched
    psi_s = H1 * psi_s_r1 + (1.0 - H1) * (H2 * psi_s_r2 + (1.0 - H2) * psi_s_r3)
    psi_d = H1 * psi_d_r1 + (1.0 - H1) * (H2 * psi_d_r2 + (1.0 - H2) * psi_d_r3)

    eps_var = ufl.variable(eps_dp)
    psi_s_var = ufl.replace(psi_s, {eps_dp: eps_var})
    psi_d_var = ufl.replace(psi_d, {eps_dp: eps_var})

    return g_deg(d) * ufl.diff(psi_d_var, eps_var) + ufl.diff(psi_s_var, eps_var)

def psi_d_dp(u, dim):
    B_dp = fem.Constant(mesh, -0.26)
    C_dp = 1.0 / (18.0 * B_dp**2 * kappa_s + 2.0 * lame2)
    stab_dp = 1e-12
    eps_dp = ufl.sym(ufl.grad(u))
    I = ufl.Identity(dim)
    I1 = ufl.tr(eps_dp)
    eps_dev = eps_dp - (1.0/3.0) * I1 * I
    J2 = 0.5 * ufl.inner(eps_dev, eps_dev)
    sqrt_J2 = ufl.sqrt(J2 + stab_dp)

    # --- same switching quantities as sigma_dp, same fix applied ---
    g1 = kappa_s * (I1 + 6.0 * B_dp * sqrt_J2)                      # FIXED: scaled by kappa_s, matches sigma_dp
    g2 = 2.0 * lame2 * sqrt_J2 - 3.0 * B_dp * kappa_s * I1          # already stress-scale, unchanged

    # --- same shared constant as sigma_dp — do NOT redefine locally ---
    width = dp_blend_width
    H1 = 0.5 * (1.0 + ufl.tanh(g1 / width))   # ~1 in region 1, ~0 otherwise
    H2 = 0.5 * (1.0 + ufl.tanh(g2 / width))   # ~1 in region 2, ~0 in region 3

    psi_d_r1 = 0.5 * kappa_s * I1**2 + 2.0 * lame2 * J2
    psi_d_r2 = C_dp * (-3.0 * B_dp * kappa_s * I1 + 2.0 * lame2 * sqrt_J2)**2
    psi_d_r3 = 0.0

    # blended instead of nested conditional
    psi_d = H1 * psi_d_r1 + (1.0 - H1) * (H2 * psi_d_r2 + (1.0 - H2) * psi_d_r3)

    driving_force = (ScalarType(2.0) * psi_d) / (Gc / ls_pfm)
    thresholded_driving_force = driving_force - 1.0

    return ufl.max_value(0.0, thresholded_driving_force)

def sigma_plain(u,dim):
    sig_d_plus = ScalarType(2)*lame2*eps_sd(u,dim) + (kappa_s*ptr_es(u)) *Identity(tdim)             # # kappa_s and lame_2 as value
    return sig_d_plus

def principal_stresses(u,d,dim):
    s = sigma_se(u,d,dim)
    mean_stress = 0.5 * ufl.tr(s)
    diff = 0.5 * (s[0, 0] - s[1, 1])
    radius = ufl.sqrt(diff**2 + s[0, 1]**2)
    sigma_1 = mean_stress + radius
    sigma_2 = mean_stress - radius
    return ufl.max_value(0,sigma_1), ufl.max_value(0,sigma_2)

def principal_stress_undegraded(u,dim):
    s = sigma_plain(u,dim)
    mean_stress = 0.5 * ufl.tr(s)
    diff = 0.5 * (s[0, 0] - s[1, 1])
    radius = ufl.sqrt(diff**2 + s[0, 1]**2)
    sigma_1 = mean_stress + radius
    sigma_2 = mean_stress - radius
    return ufl.max_value(0,sigma_1), ufl.max_value(0,sigma_2)

def dc_energy_elastic(u,dim):
    E_crit = lame2 * (ScalarType(3) * lame1 + ScalarType(2) * lame2) / (lame1 + lame2)
    pot_crit = sigma_st ** ScalarType(2) / (ScalarType(2) * E_crit)
    interm1_num = ScalarType(0.5) * kappa_s * (ptr_es(u))**ScalarType(2) + lame2 * ufl.inner(eps_sd(u,dim),eps_sd(u,dim))
    interm1 = (interm1_num / pot_crit) - ScalarType(1)
    return ScalarType(0.5)*(interm1 + abs(interm1))

# def dc_tilda(u,dim):
#     # principal_stress_values = principal_stresses(u,d,dim)
#     # principal_stress_values = principal_stress_undegraded(u,dim)
#     # interm = sum((s**ScalarType(2)/sigma_st**ScalarType(2)) for s in principal_stress_values)-ScalarType(1)           # sigma_st as function
#     # return ScalarType(0.5)*(interm + abs(interm))
#     return dc_energy_elastic(u,dim)

# def dc_tilda(u, dim):
#     p1, p2 = principal_stress_undegraded(u, dim)
#     E = lame2 * (3 * lame1 + 2 * lame2) / (lame1 + lame2)
#     nu = lame1 / (2 * (lame1 + lame2))
#     E_eff = E / (1 - nu**2)
#     # energy_tensile = (p1**2 + p2**2) / (2 * E_eff)
#     nu_eff = nu / (1.0 - nu)
#     psi_tensile = (1.0 / (2.0 * E_eff)) * (p1**2 + p2**2 - 2.0 * nu_eff * p1 * p2)
#     # psi_crit = (sigma_st**2) / (2.0 * E_eff)
#     # driving_force = (psi_tensile / psi_crit) - 1.0
#     driving_force = (psi_tensile / (sigma_st/ScalarType(2))) 
#     # return 0.5 * (driving_force + abs(driving_force))
#     trace_eps = ufl.tr(eps_s(u, dim))
#     bulge_only_filter = ufl.conditional(ufl.gt(trace_eps, 1e-6), 1.0, 0.0)
#     return driving_force * bulge_only_filter

def eps_s(u, dim):
    return ufl.sym(ufl.grad(u))

def sigma_undegraded(u, dim):
    # 1. Get the strain tensor
    eps = eps_s(u, dim)
    
    # 2. Apply Hooke's law in 2D/3D using UFL operators
    # lame1 = lambda, lame2 = mu
    return 2.0 * lame2 * eps + lame1 * ufl.tr(eps) * ufl.Identity(dim)

def dc_tilda(u, dim):
    # 1. Compute undegraded stress tensor directly
    sigma = sigma_undegraded(u, dim)
    tr_s = ufl.tr(sigma)
    det_s = ufl.det(sigma)
    
    # 2. Extract principal stresses (2D Eigenvalues of the stress tensor)
    term1 = tr_s / 2.0
    term2 = ufl.sqrt(ufl.max_value(0.0, (tr_s / 2.0)**2 - det_s))
    
    val1 = term1 + term2
    val2 = term1 - term2
    
    # 3. Isolate ONLY the positive (tensile) principal stresses
    # Under the punch tips, val1 and val2 are heavily negative, yielding 0.0.
    # In the valleys between fibers, local vertical tension makes val1 positive.
    p1 = ufl.max_value(0.0, val1)
    p2 = ufl.max_value(0.0, val2)
    
    # 4. Extract proper effective material constants
    E = lame2 * (3.0 * lame1 + 2.0 * lame2) / (lame1 + lame2)
    nu = lame1 / (2.0 * (lame1 + lame2))
    E_eff = E / (1.0 - nu**2)
    nu_eff = nu / (1.0 - nu)
    
    # 5. Tensile Stress Energy Density
    # Since p1 and p2 are strictly positive, compression energy is perfectly 0.0.
    psi_tensile = (1.0 / (2.0 * E_eff)) * (p1**2 + p2**2 - 2.0 * nu_eff * p1 * p2)
    
    # 6. Normalize precisely to your paper's definition (Equation 23)
    # H = 2 * psi_plus / (Gc / ls)
    driving_force = (ScalarType(2.0) * psi_tensile) / (Gc / ls_pfm)
    
    # 7. Physical Threshold Check (-1.0 from Equation 23)
    # If the energy hasn't surpassed the threshold, do not accumulate damage.
    thresholded_driving_force = driving_force - 1.0
    
    # Max value with 0 removes negative states, ensuring damage only grows
    return ufl.max_value(0.0, thresholded_driving_force)


def invert_J_vectorized(J_target, is_hydrophilic):
    J_clamped = np.clip(J_target, 0.0, 0.559)
    x = np.zeros_like(J_clamped)
    for _ in range(8): 
        f = 1.417*x - 2.12*x**2 + 1.262*x**3 - J_clamped
        df = 1.417 - 4.24*x + 3.786*x**2
        mask = np.abs(df) > 1e-6
        x[mask] = x[mask] - f[mask] / df[mask]
    x_final = np.clip(x, 0.0, 1.0)
    return np.where(is_hydrophilic, 1.0 - x_final, x_final)

import numpy as np
import dolfinx.fem as fem

def update_saturation_single_angle(p_l_sub, p_g_sub, current_nF_array, s_l_func):
    Q_dg = s_l_func.function_space
    pl_collapsed = p_l_sub.collapse()
    pg_collapsed = p_g_sub.collapse()
    pl_standalone = pl_collapsed[0] if isinstance(pl_collapsed, tuple) else pl_collapsed
    pg_standalone = pg_collapsed[0] if isinstance(pg_collapsed, tuple) else pg_collapsed
    pl_dg = fem.Function(Q_dg)
    pg_dg = fem.Function(Q_dg)
    
    pl_dg.interpolate(pl_standalone)
    pg_dg.interpolate(pg_standalone)

    pc_array    = pg_dg.x.array - pl_dg.x.array
    theta_array = theta.x.array
    eps_array   = current_nF_array
    K_array     = KS0.x.array
    gamma_val   = surf_h2o.value
    ct_array = ct.values

    cos_theta = np.cos(theta.x.array)
    sqrt_EK = np.sqrt(eps_array / K_array)
    scaling_factor = gamma_val * cos_theta * sqrt_EK

    sl_new = np.ones_like(pc_array)

    # Define Domain Masks
    is_hydrophilic = theta_array < (np.pi / 2)
    is_membrane = (ct_array == 6)

    j_target = np.abs(pc_array / (scaling_factor + 1e-12))
    # mask_active = (pc_array > 0)
    # mask_active = ~is_membrane
    
    # if np.any(mask_active):
    #         results = invert_J_vectorized(j_target[mask_active], is_hydrophilic[mask_active])
    #         sl_new[mask_active] = results

    sl_new = invert_J_vectorized(j_target, is_hydrophilic)
    sl_new[is_membrane] = 0.999
    # s_l_func.x.array[:] = sl_new
    s_l_func.x.array[:] = np.clip(sl_new, 0.01, 1.0)

# ### 5. Assign the initial conditions

#Define the functions at time t=0s:
u1_n = Function(V_pot)
u2_n = Function(V_tsp)
u3_n = Function(V_dam)
u4_n = Function(V_mech)
# pe_init, pp_init  = split(u1_n) 
pl_init, pg_init = split(u2_n) 
# pl_init = u2_n.sub(0)
# pg_init = u2_n.sub(1)
# uS_init = u2_n.sub(2)

# uS_init = u2_n
d_init = u3_n
uS_init = u4_n

membrane_interior_marker = Function(V_tsp.sub(0).collapse()[0])
def find_interior(x):
    in_mem = (x[0] > x2 + 0.00062+1e-6) & (x[0] < x3 - 0.00062-1e-6)
    return in_mem.astype(ScalarType)
membrane_interior_marker.interpolate(find_interior)

def membrane_pl_distribution(x):
    pl_anode = 0.1
    pl_cathode = 0.1060
    L_mem = x3 - x2
    return pl_anode + (pl_cathode - pl_anode) * (x[0] - x2) / L_mem
pl_pin = Function(V_tsp.sub(0).collapse()[0]) 
pl_pin.interpolate(membrane_pl_distribution)

def membrane_pg_distribution(x):
    pg_anode = 0.1013
    pg_cathode = 0.1
    L_mem = x3 - x2
    return pg_anode + (pg_cathode - pg_anode) * (x[0] - x2) / L_mem
pg_pin = Function(V_tsp.sub(1).collapse()[0]) 
pg_pin.interpolate(membrane_pg_distribution)

def general_linear_init(x, v_left, v_right):
    values = np.zeros(x.shape[1], dtype=default_real_type)
    mask_acl = x[0] < x2
    values[mask_acl] = v_left
    mask_ccl = x[0] > x3
    values[mask_ccl] = v_right
    mask_mem = np.logical_and(x[0] >= x2, x[0] <= x3)
    x_rel = (x[0][mask_mem] - x2) / (x3 - x2)
    values[mask_mem] = v_left + (v_right - v_left) * x_rel
    return values

#Initial condidtion:
u1_m.sub(0).interpolate(lambda x: general_linear_init(x, 1.5, 0.02))
u1_m.sub(1).interpolate(lambda x: general_linear_init(x, 0.2, 0.001))
u2_n.sub(0).interpolate(lambda x: general_linear_init(x, 0.1, 0.1060))
u2_n.sub(1).interpolate(lambda x: general_linear_init(x, 0.1013, 0.1))

# uS_init = Function(VuS)
uS_init.x.array[:] = ScalarType(1e-10)               
u4_n.x.array[:] = uS_init.x.array

Q = fem.functionspace(mesh, ("DG", 0))
s_l = Function(Q)
sl_calculated = Function(Q)
s_l.name = "Saturation"
# ================================================================================
### POTENTIAL SOLVER ONLY
# s_l.x.array[:] = 1.0                               
# ================================================================================    
### BOTH POTENTIAL AND TRANSPORT SOLVER                                    
s_l.x.array[:] = 0.9
update_saturation_single_angle(u2_n.sub(0), u2_n.sub(1), nF_0S.x.array, s_l)  
# ================================================================================            
# u1_m.x.array[:] = u1_n.x.array[:] # Sync current state
u2_m.x.array[:] = u2_n.x.array[:] # Sync current state
sl_m = s_l
sl_n = s_l
sg_m = ScalarType(1.0) - sl_m
sg_n = ScalarType(1.0) - sl_n
eps_diff = 1e-5
sl_safe = ufl.max_value(ufl.min_value(sl_m, ScalarType(1.0)), ScalarType(0.0))
k_rel_w = (sl_safe * sl_safe * sl_safe) + eps_diff
k_rel_g = (1 - sl_safe)*(1 - sl_safe)*(1 - sl_safe) + eps_diff

u1_n.x.scatter_forward()
u2_n.x.scatter_forward()
u3_n.x.scatter_forward()
u4_n.x.scatter_forward()
u1_m.x.scatter_forward()
u2_m.x.scatter_forward()
print('Min value of sl_n =',np.min(sl_n.x.array), flush=True)
print("u2_m has NaN:", np.isnan(u2_m.x.array).any(), flush=True)
print("sl_n range:", sl_n.x.array.min(), sl_n.x.array.max(),flush=True)

trac1 = Constant(mesh, ScalarType([tx1_value,0]))

xdmf = XDMFFile(mesh.comm, "debugging_initial_ECHM.xdmf", "w")
s_l.name = 's_l'
xdmf.write_mesh(mesh)
xdmf.write_function(s_l,t) 
xdmf.close()

pe_n, pp_n  = split(u1_n) 
# pe_m, pp_m  = split(u1_m)   # fake
# pl_n, pg_n, uS_n  = split(u2_n) 
pl_n, pg_n  = split(u2_n) 

# pl_n = u2_n.sub(0)
# pg_n = u2_n.sub(1)
# uS_n = u2_n.sub(2)

# uS_m = ufl.split(u2_m)[2]
# pl_m, pg_m, uS_m = ufl.split(u2_m)


uS_n = u4_n
uS_m = u4_m
d_n  = u3_n
d_m = u3_m
du_pot = TrialFunction(V_pot)
du_tsp = TrialFunction(V_tsp)
du_dam = TrialFunction(V_dam)
du_mech = TrialFunction(V_mech)
n_out = FacetNormal(mesh)

m_value = 0.95

# =========================================================================
# INITIALIZATION (Run ONCE before the while loop)
# =========================================================================
V_press = fem.functionspace(mesh, ("CG", 1))
pressure_profile = fem.Function(V_press)
       
dof_map = V_press.dofmap
index_map = dof_map.index_map
local_size = index_map.size_local * dof_map.index_map_bs
dof_coords = V_press.tabulate_dof_coordinates()[:index_map.size_local]

values = np.zeros(local_size)
bs = dof_map.index_map_bs

# Fiber geometry setup
centers = [0.025, 0.050, 0.075]
weights = [1.5,   1.5,   1.5]
radii   = [0.015, 0.005, 0.015]

# Tuning parameter for the exterior fiber bulge shape:
# Lower numbers (< 1.0) make it bulge out wider in the middle.
# Higher numbers (> 1.0) make it sharper. Try 0.5 to 0.7 for a fat bulge.
bulge_power = 0.8

# Correctly map the y-coordinate using two separate mathematical profiles
for local_dof, coord in enumerate(dof_coords):
    y_val = coord[1]  
    
    for i, (c, w, r) in enumerate(zip(centers, weights, radii)):
        dist = np.abs(y_val - c)
        
        if dist < r:
            if i == 1:
                # ---------------------------------------------------------
                # MIDDLE FIBER (i=1): Standard Semicircle Profile
                # ---------------------------------------------------------
                hump = np.sqrt(r**2 - dist**2) / r
            else:
                # ---------------------------------------------------------
                # EXTERIOR FIBERS (i=0 and i=2): Custom Bulging Curve
                # ---------------------------------------------------------
                # This creates a smooth cosine-bell curve raised to a power
                # to control the center-bulge width independently.
                hump = np.cos((np.pi * dist) / (2.0 * r)) ** bulge_power
            
            # Map the resulting shape factor safely into the DOF array
            for b in range(bs):
                values[local_dof * bs + b] = hump * w

# Assign the mapped array into the FEniCSx function memory container
pressure_profile.x.array[:local_size] = values
pressure_profile.x.scatter_forward()

# Compile into the traction vector form for your GU equation
traction_vector = ufl.as_vector((trac1[0] * pressure_profile, 0.0))

# ### 6. Define the weak formulation and the solver

G_pot = GEP + GPP 
G_tsp = GPL + GPG 
G_dam = GD
G_mech = GU

J_pot = derivative(G_pot, u1_m, du_pot)
J_tsp = derivative(G_tsp, u2_m, du_tsp)
J_dam = derivative(G_dam, u3_m, du_dam)
J_mech = derivative(G_mech, u4_m, du_mech)

#Define the newton solver for the nonlinear problem
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver

problem_pot = NonlinearProblem(G_pot, u1_m, bcs=bc_pot, J=J_pot)
solver_pot = NewtonSolver( MPI.COMM_WORLD, problem_pot) #domain.comm
solver_pot.convergence_criterion = "incremental"
solver_pot.rtol = 1e-6 #relative tol
solver_pot.atol = 1e-6 #absolut tol
# solver.relaxation_parameter = 1
ksp = solver_pot.krylov_solver
opts = PETSc.Options()
prefix = ksp.getOptionsPrefix()
opts[f"{prefix}ksp_type"] = "gmres"
opts[f"{prefix}pc_type"] = "hypre"
# opts[f"{prefix}pc_factor_mat_solver_type"] = "mumps"
opts[f"{prefix}ksp_error_if_not_converged"] = True
ksp.setFromOptions()
solver_pot.max_it = 50
solver_pot.error_on_nonconvergence = True
solver_pot.report = True
solver_pot.relaxation_parameter = 0.8

problem_tsp = NonlinearProblem(G_tsp, u2_m, bcs=bc_tsp, J=J_tsp)
solver_tsp = NewtonSolver( MPI.COMM_WORLD, problem_tsp) #domain.comm
solver_tsp.convergence_criterion = "incremental"
solver_tsp.rtol = 1e-5 #relative tol
solver_tsp.atol = 1e-5 #absolut tol
# solver.relaxation_parameter = 1
ksp = solver_tsp.krylov_solver
opts = PETSc.Options()
prefix = ksp.getOptionsPrefix()
opts[f"{prefix}ksp_type"] = "gmres"
opts[f"{prefix}pc_type"] = "hypre"
opts[f"{prefix}pc_factor_mat_solver_type"] = "mumps"
opts[f"{prefix}ksp_error_if_not_converged"] = True
solver_tsp.error_on_nonconvergence = False
ksp.setFromOptions()
solver_tsp.max_it = 50
solver_tsp.error_on_nonconvergence = True
solver_tsp.report = True
solver_tsp.relaxation_parameter = 0.7

problem_mech = NonlinearProblem(G_mech, u4_m, bcs=bc_mech, J=J_mech)
solver_mech = NewtonSolver( MPI.COMM_WORLD, problem_mech) #domain.comm
solver_mech.convergence_criterion = "incremental"
solver_mech.rtol = 1e-4 #relative tol
solver_mech.atol = 1e-4 #absolut tol
# solver.relaxation_parameter = 1
ksp = solver_mech.krylov_solver
opts = PETSc.Options()
prefix = ksp.getOptionsPrefix()
opts[f"{prefix}ksp_type"] = "gmres"
opts[f"{prefix}pc_type"] = "hypre"
# opts[f"{prefix}pc_factor_mat_solver_type"] = "mumps"
opts[f"{prefix}ksp_error_if_not_converged"] = True
solver_mech.error_on_nonconvergence = False
ksp.setFromOptions()
solver_mech.max_it = 50
solver_mech.error_on_nonconvergence = True
solver_mech.report = True
solver_mech.relaxation_parameter = 0.8

problem_dam = NonlinearProblem(G_dam, u3_m, bcs=[], J=J_dam)
solver_dam = NewtonSolver( MPI.COMM_WORLD, problem_dam) #domain.comm
solver_dam.convergence_criterion = "incremental"
solver_dam.rtol = 1e-4 #relative tol
solver_dam.atol = 1e-4 #absolut tol
# solver.relaxation_parameter = 1
ksp = solver_dam.krylov_solver
opts = PETSc.Options()
prefix = ksp.getOptionsPrefix()
opts[f"{prefix}ksp_type"] = "gmres"
opts[f"{prefix}pc_type"] = "hypre"
# opts[f"{prefix}pc_factor_mat_solver_type"] = "mumps"
opts[f"{prefix}ksp_error_if_not_converged"] = True
ksp.setFromOptions()
solver_dam.max_it = 50
solver_dam.error_on_nonconvergence = True
solver_dam.report = True
solver_dam.relaxation_parameter = 0.8

# problem_dam = LinearProblem(
#     a_dam,
#     L_dam,
#     u=u3_m,
#     bcs=[],
#     petsc_options={
#         "ksp_type": "cg",
#         "pc_type": "hypre",
#         "pc_hypre_type": "boomeramg",
#         "ksp_rtol": 1e-8,
#         "ksp_atol": 1e-10,
#         "ksp_max_it": 200
#     }
# )

# from dolfinx.io import VTXWriter
from pathlib import Path
# folder = Path("3domain_3Field")
# folder.mkdir(exist_ok=True, parents=True)

Vpe_sol, up_to_pe_sol = V_pot.sub(0).collapse() 
pe_sol = Function(Vpe_sol) 
Vpp_sol, up_to_pp_sol = V_pot.sub(1).collapse() 
pp_sol = Function(Vpp_sol) 
# Vd_sol, up_to_d_sol = V_pot.sub(2).collapse() 
# d_sol = Function(Vd_sol) 
Vpl_sol, up_to_pl_sol = V_tsp.sub(0).collapse() 
pl_sol = Function(Vpl_sol) 
Vpg_sol, up_to_pg_sol = V_tsp.sub(1).collapse() 
pg_sol = Function(Vpg_sol) 
# VuS_sol, up_to_uS_sol = V_tsp.sub(2).collapse() 
# u_sol = Function(VuS_sol)  
# VuS_sol, up_to_uS_sol = V_tsp.collapse()
u_sol = Function(V_mech)  
d_sol = Function(V_dam)  

pe_sol.name  = "Electric Potential [V]"
pp_sol.name  = "Protonic Potential [V]"
d_sol.name   = "Phase feld"
pl_sol.name  = "Liquid Pressure [MPa]"
pg_sol.name  = "Gas Pressure [MPa]"
u_sol.name   = "Displacement [mm]"

xdmf_pe= io.XDMFFile(mesh.comm, "./results/pe.xdmf", "w")
xdmf_pp= io.XDMFFile(mesh.comm, "./results/pp.xdmf", "w")
xdmf_pl= io.XDMFFile(mesh.comm, "./results/pl.xdmf", "w")
xdmf_pg= io.XDMFFile(mesh.comm, "./results/pg.xdmf", "w")
xdmf_d= io.XDMFFile(mesh.comm, "./results/d.xdmf", "w")
xdmf_u= io.XDMFFile(mesh.comm, "./results/u.xdmf", "w")

xdmf_pe.write_mesh(mesh)
xdmf_pp.write_mesh(mesh)
xdmf_pl.write_mesh(mesh)
xdmf_pg.write_mesh(mesh)
xdmf_d.write_mesh(mesh)
xdmf_u.write_mesh(mesh)

# with XDMFFile(mesh.comm, "d_live.xdmf", "w") as xdmf_d_live:
#     xdmf_d_live.write_mesh(mesh)

# ### 8. Solve the nonlinear consolidation problem 
def solve_named(solver, u, name):
    try:
        return solver.solve(u)
    except RuntimeError as e:
        if MPI.COMM_WORLD.rank == 0:
            print(f"[{name}] Newton solver failed to converge: {e}")
        raise

# import matplotlib.pyplot as plt
import numpy as np

t = 0
# Adaptive time stepping parameters
dt_min = 1e-6          # Minimum allowed time step
dt_max = 0.01           # Maximum allowed time step
target_delta_d = 0.05    # We want d to change by ~0.1 per step
dt = 0.01              # Initial dt
dt_dam.value = ScalarType(dt)

results_V = []
results_I = []

output_folder = "vtk_output"
if MPI.COMM_WORLD.rank == 0:
    os.makedirs(output_folder, exist_ok=True)
file_counter = 0
n_it_tsp = 1
# -------------------------------------------------------------------------
# FIXED POINT ITERATION PARAMETERS
# -------------------------------------------------------------------------
max_fp_iter = 15      
fp_tol = 1e-3             
relaxation = 0.7         

if MPI.COMM_WORLD.rank == 0:
    print(f"{'Time':<15} | {'J (A/cm2)':<15} | {'Voltage (Ve)':<15} | {'Voltage (Vp)':<15} | {'Nit_pot':<10} | {'Nit_dam':<10} | {'Nit_tsp':<10} | {'Electro_switch':<15} | {'Saturation min':<15}" )
    print("-" * 105)

while t < t_max:
    
    # --- FIX 1: Extract d from its dedicated standalone function u3_n ---
    d_prev_array = u3_n.x.array.copy()
    H_prev = dc_H.x.array.copy()

    t += dt
    disp_value.value = np.array(update_variable_linear(t, initial_value_disp, final_value_disp, time_to_start_disp, time_to_constant_disp), dtype=np.float64)
    current_value.value = np.array(update_variable_nonlinear(t, initial_value_current, final_value_current, time_to_start_current, time_to_constant_current, 2.0), dtype=np.float64)
    electro_switch.value = np.array(update_variable_linear(t, initial_value_electro_switch, final_value_electro_switch, time_to_start_current, time_to_start_current), dtype=np.float64)
    trac1.value[0]          =  update_variable_linear(t,initial_value_tx1,final_value_tx1,time_to_start_tx1,time_to_constant_tx1)
    
    pe_n, pp_n   = split(u1_n) 
    pl_n, pg_n  = split(u2_n) 
    # pl_n = u2_n.sub(0)
    # pg_n = u2_n.sub(1)
    # uS_n = u2_n.sub(2)
    uS_n  = u4_n
    d_n   = u3_n  # Correctly referencing standalone damage
    
    try:
    
        u1_backup = u1_n.x.array.copy()
        u2_backup = u2_n.x.array.copy()
        u3_backup = u3_n.x.array.copy()
        u4_backup = u4_n.x.array.copy()
        dc_H_backup = dc_H.x.array.copy()
        sl_backup = sl_n.x.array.copy()

        # 1. Solve Potentials
        n_it_pot, converged = solve_named(solver_pot,  u1_m, "POT")

        u1_n.x.array[:] = u1_m.x.array
        u1_n.x.scatter_forward()

        n_it_mech, converged = solve_named(solver_mech, u4_m, "MECH")
        u4_n.x.array[:] = u4_m.x.array
        u4_n.x.scatter_forward()

        # ===========================================================================================================================================================
        # TRANSPORT SOLVER BLOCK
        # ===========================================================================================================================================================
        for iter_fp in range(max_fp_iter):
            sl_prev_iter = sl_m.x.array.copy()

            try:
                n_it_tsp, converged = solver_tsp.solve(u2_m)
            except RuntimeError:
                print("Transport Solver failed to converge.")
                break
            nF_current_expr = nF(uS_n, d_m, tdim) # Note: updated to d_m (current iteration damage)
            nF_dg = fem.Function(Q)       
            nF_dg.interpolate(fem.Expression(nF_current_expr, Q.element.interpolation_points()))

            update_saturation_single_angle(u2_m.sub(0), u2_m.sub(1), nF_dg.x.array, sl_calculated)
            new_sl_values = (1.0 - relaxation) * sl_prev_iter + relaxation * sl_calculated.x.array

            sl_m.x.array[:] = new_sl_values
            sl_m.x.scatter_forward()

            local_diff_sq = np.linalg.norm(sl_m.x.array - sl_prev_iter)**2
            diff = np.sqrt(MPI.COMM_WORLD.allreduce(local_diff_sq, op=MPI.SUM))

            if diff < fp_tol:
                break
        else:
            print("   Warning: Picard iteration for saturation did not converge.")
    
        # =====================================================================
        # UPDATE HISTORY FIELD
        # =====================================================================
        dc_tilda_expr = Expression(psi_d_dp(uS_n, tdim), funch.element.interpolation_points())
        temp_dc.interpolate(dc_tilda_expr) 
        dc_H.x.array[:] = np.maximum(dc_H.x.array, temp_dc.x.array)
        dc_H.x.scatter_forward()

        # =====================================================================
        # DAMAGE SOLVE
        # =====================================================================
        
        # local_dc_max = np.max(dc_H.x.array)
        # local_disp_norm = np.linalg.norm(u2_m.x.array)
        # local_disp_max = np.max(np.abs(u2_m.x.array))
        # local_traction = trac1.value[0]
        # dc_max = mesh.comm.allreduce(local_dc_max,op=MPI.MAX)
        # disp_norm = mesh.comm.allreduce(local_disp_norm,op=MPI.MAX)
        # disp_max = mesh.comm.allreduce(local_disp_max,op=MPI.MAX)
        # traction = mesh.comm.allreduce(local_traction,op=MPI.MAX)
        # if MPI.COMM_WORLD.rank == 0:
        #     print(
        #         f"dt={dt:.2e}, "
        #     )
        # print(
        #     f"dmax(old)={np.max(u3_n.x.array):.6f}"
        # )
        # n_it_dam, converged_dam = solver_dam.solve(u3_m)
        dt_dam.value = ScalarType(dt)
        # problem_dam.solve()
        # n_it_dam = 0
        n_it_dam, converged_dam = solve_named(solver_dam,  u3_m, "DAM")
        u3_m.x.array[:] = np.clip(u3_m.x.array, 1e-5, (1.0-1e-5))
        d_m = u3_m
        
        # if MPI.COMM_WORLD.rank == 0:
        #     print(f"Damage Newton iterations = {n_it_dam}")

        # if MPI.COMM_WORLD.rank == 0:
        #     print(f"dmax(new)={np.max(u3_m.x.array):.6f}")

        # --- FIX 2: Calculate delta_d using the dedicated u3_m array ---
        d_current_array = u3_m.x.array
        # if t > 1.0 : 
        #     d_current_array = d_n
        #     u3_m.x.array[:] = d_n
        
        local_delta = np.max(np.abs(u3_m.x.array - d_prev_array))
        
        delta_d_max = mesh.comm.allreduce(local_delta,op=MPI.MAX)
        # if MPI.COMM_WORLD.rank == 0:
        #     print(f"delta_d = {delta_d_max:.6e}")

        # =====================================================================
        # ADAPTIVE TIME STEPPING
        # =====================================================================
  
        if delta_d_max > 1.0 * target_delta_d:

            if dt > dt_min * 1.1: 
                t -= dt  # Roll back
                dt = max(dt / 2.0, dt_min)
                dt_dam.value = ScalarType(dt)
                
                u1_m.x.array[:] = u1_backup
                u2_m.x.array[:] = u2_backup
                u3_m.x.array[:] = u3_backup
                u4_m.x.array[:] = u4_backup
                
                dc_H.x.array[:] = dc_H_backup
                sl_m.x.array[:] = sl_backup
                
                u1_m.x.scatter_forward()
                u2_m.x.scatter_forward()
                u3_m.x.scatter_forward()
                u4_m.x.scatter_forward()
                dc_H.x.scatter_forward()
                sl_m.x.scatter_forward() 
                
                if MPI.COMM_WORLD.rank == 0:
                    print(f"REJECTED: delta_d {delta_d_max:.4f} > {target_delta_d}. New dt: {dt:.2e}", flush=True)
                continue 
            else:
                if MPI.COMM_WORLD.rank == 0:
                    print(f"FORCED ACCEPT: delta_d {delta_d_max:.4f} at dt_min floor.")
        
        elif delta_d_max < target_delta_d * 0.8:
            dt = min(dt * 1.2, dt_max)
            dt_dam.value = ScalarType(dt)

    except RuntimeError:
        if dt > dt_min * 1.1:
            t -= dt
            dt = max(dt / 5.0, dt_min)
            dt_dam.value = ScalarType(dt)
            u1_m.x.array[:] = u1_backup
            u2_m.x.array[:] = u2_backup
            u3_m.x.array[:] = u3_backup
            u4_m.x.array[:] = u4_backup
            u1_m.x.scatter_forward()
            u2_m.x.scatter_forward()
            u3_m.x.scatter_forward()
            u4_m.x.scatter_forward()

            if MPI.COMM_WORLD.rank == 0:
                print(f"REJECTED: Solver failed. New dt: {dt:.2e}")
            continue
        else:
            if MPI.COMM_WORLD.rank == 0:
                print("FATAL: Solver failed at dt_min. Simulation terminated.")
            break
    
    u1_n.x.array[:] = u1_m.x.array
    u2_n.x.array[:] = u2_m.x.array
    u3_n.x.array[:] = u3_m.x.array # Keep solution updated
    u4_n.x.array[:] = u4_m.x.array

    u1_n.x.scatter_forward()
    u3_n.x.scatter_forward()
    u2_n.x.scatter_forward()
    u4_n.x.scatter_forward()

    
    # Process potential outputs safely via sub() mapping
    Ve_vals = u1_m.sub(0).collapse().x.array
    Vp_vals = u1_m.sub(1).collapse().x.array
    local_ve = np.max(Ve_vals) if len(Ve_vals) > 0 else -np.inf
    local_vp = np.max(Vp_vals) if len(Vp_vals) > 0 else -np.inf
    cell_ve = mesh.comm.allreduce(local_ve, op=MPI.MAX)
    cell_vp = mesh.comm.allreduce(local_vp, op=MPI.MAX)

    current_print_val = float(current_value.value) * 100

    if 0.2 < cell_ve < 4.0:
        results_V.append(cell_ve)
        results_I.append(current_value.value*100)
        if MPI.COMM_WORLD.rank == 0:
            print(f"{t:<15.4f} | {current_print_val:<15.4f} | {cell_ve:<15.4f} | {cell_vp:<15.4f} | {n_it_pot:<10.0f} | {n_it_dam:<10.0f} | {n_it_tsp:<10.0f} | {float(electro_switch.value):<15.4f} | {float(electro_switch.value):<15.4f} ", flush=True)
    else:
        if MPI.COMM_WORLD.rank == 0:
            print(f"{t:<15.4f} | {current_print_val:<15.4f} | {cell_ve:<15.4f} | {cell_vp:<15.4f} | Unphysical")

    # # Catalyst activity ratio
    # kappa_used = (kappa_eb_scaled + d_n * (kappa_eb_residual - kappa_eb_scaled))
    # a2_eff = fem.assemble_scalar(fem.form(o2_marker * a2_degraded_old * ufl.dx))
    # a2_orig = fem.assemble_scalar(fem.form(o2_marker * a2 * ufl.dx))
    # a2_eff = mesh.comm.allreduce(a2_eff, op=MPI.SUM)
    # a2_orig = mesh.comm.allreduce(a2_orig, op=MPI.SUM)
    # # Conductivity ratio
    # kappa_eff = fem.assemble_scalar(fem.form(o2_marker * kappa_used * ufl.dx))
    # kappa_orig = fem.assemble_scalar(fem.form(o2_marker * kappa_eb * ufl.dx))
    # kappa_eff = mesh.comm.allreduce(kappa_eff, op=MPI.SUM)
    # kappa_orig = mesh.comm.allreduce(kappa_orig, op=MPI.SUM)

    # crack_indicator = ufl.conditional(ufl.gt(d_n, 0.5), 1.0, 0.0)
    # crack_area_local = fem.assemble_scalar(fem.form(o2_marker * crack_indicator * ufl.dx))
    # acl_area_local = fem.assemble_scalar(fem.form(o2_marker * ufl.dx))
    # crack_area = mesh.comm.allreduce(crack_area_local, op=MPI.SUM)
    # acl_area = mesh.comm.allreduce(acl_area_local, op=MPI.SUM)
    # factor_acl = 1.0 - crack_area/acl_area

    # a2_global_factor.value = ScalarType(factor_acl**10)
    
    # # a2_degraded = a2 * (ScalarType(1.0) - (crack_area / acl_area))**3
    # if mesh.comm.rank == 0:
    #     print(f"acl area      = {acl_area:.4f}")
    #     print(f"a2 ratio      = {a2_eff/a2_orig:.4f}")
    #     print(f"area ratio    = {factor_acl:.4f}")
    #     print(f"kappa ratio   = {kappa_eff/kappa_orig:.4f}")

    # Update solutions ghost cells
    u1_m.x.scatter_forward()
    u2_m.x.scatter_forward()
    u3_m.x.scatter_forward() # Added forward scatter for damage
    u4_m.x.scatter_forward()

    u1_n.x.array[:] = u1_m.x.array
    u2_n.x.array[:] = u2_m.x.array
    u3_n.x.array[:] = u3_m.x.array # Sync historical tracking array
    u4_n.x.array[:] = u4_m.x.array
    sl_n.x.array[:] = sl_m.x.array

    sg_m = ScalarType(1.0) - sl_m
    sg_n = ScalarType(1.0) - sl_n

    # --- FIX 3: Post-processing splits matched to updated formats ---
    pe_h, pp_h = u1_m.split()  # V_pot splits into 2 components now
    pl_h, pg_h = u2_m.split()
    d_h = u3_m                 # d is standalone
    uS_h = u4_m
    
    pe_sol.interpolate(pe_h)
    pp_sol.interpolate(pp_h)
    pl_sol.interpolate(pl_h)
    pg_sol.interpolate(pg_h)
    u_sol.interpolate(uS_h)
    d_sol.interpolate(d_h)

    V_vis = fem.functionspace(mesh, ("CG", 1, (mesh.geometry.dim,)))
    u_vis = Function(V_vis)
    u_vis.name = "u"
    u_vis.interpolate(uS_h)

    # File Writing
    xdmf_pe.write_function(pe_sol, t)
    xdmf_pp.write_function(pp_sol, t)
    xdmf_pl.write_function(pp_sol, t)
    xdmf_pg.write_function(pp_sol, t)
    xdmf_d.write_function(d_sol, t)
    xdmf_u.write_function(u_vis, t)
    

    if MPI.COMM_WORLD.rank == 0:
            with open("output.txt", "a") as f:
                f.write(f"Time step: {t} finished\n")
                f.write(f"{t:<15.4f} | {current_print_val:<15.4f} | {cell_ve:<15.4f} | {cell_vp:<15.4f} | {n_it_pot:<5.0f} | {float(electro_switch.value):<15.4f} | {float(electro_switch.value):<15.4f} ")
    file_counter += 1
print("\nSimulation completed!")
xdmf_pe.close()
xdmf_pp.close()
xdmf_pl.close()
xdmf_pg.close()
xdmf_d.close()
xdmf_u.close()
