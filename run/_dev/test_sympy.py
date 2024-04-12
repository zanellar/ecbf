import sympy as sp
from sympy.utilities.lambdify import lambdify

num_states = 4

q = sp.Matrix(sp.symbols(f'q:{num_states//2}'))
p = sp.Matrix(sp.symbols(f'p:{num_states//2}'))

print(q,p)
  
# Define the Hamiltonian
_H = sp.cos(q[0]) + sp.cos(q[1])

H = lambdify([q, p], _H, modules=['sympy', 'numpy'])

print(H(q,p))