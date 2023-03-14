from jax import random
import jax.numpy as jnp
import numpy as np
from jax.experimental import sparse
from scipy.linalg import circulant
import timeit
import jax
from scipy.sparse import csr_matrix, kron, identity
import scipy


def solve_cg(A, B):

    x, e = jax.scipy.sparse.linalg.cg(
        sparse.BCOO.fromdense(A.toarray()), jnp.array(B), tol=1e-9)
    # x, e = scipy.sparse.linalg.cg(A, B, tol=1e-9)
    return np.array(x)


n = 5000
kernel = np.zeros((n, 1))
kernel[-1] = 1
kernel[0] = -2
kernel[1] = 1
D2 = csr_matrix(circulant(kernel))
v = np.ones((n, 1))
# D2 = sparse.BCOO.fromdense(circulant(kernel))
# v = jnp.array(np.ones((n, 1)))

#


start = timeit.default_timer()
x = solve_cg(D2, v)

print(timeit.default_timer()-start)
print(x.shape)
