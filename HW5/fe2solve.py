import numpy
from matplotlib import pyplot

def fsolve_newton(F, J, u0, rtol=1e-10, maxit=50, verbose=False):
    u = u0.copy()
    Fu = F(u)
    norm0 = numpy.linalg.norm(Fu)
    for i in range(maxit):
        du = sp.linalg.spsolve(J(u), -Fu)
        u += du
        Fu = F(u)
        norm = numpy.linalg.norm(Fu)
        if verbose:
            print('Newton {:d} anorm {:6.2e} rnorm {:6.2e}'.
                  format(i+1, norm, norm/norm0))
        if norm < rtol * norm0:
            break
    return u, i

def fe2_geom(fe, mesh):
    x, Erestrict = mesh.Erestrict(fe.p)
    nelem = len(Erestrict)
    Q = len(fe.w)
    B, D = fe.B, fe.D
    W = numpy.empty((nelem, Q))
    dXdx = numpy.empty((nelem, Q, 2, 2))
    xq = numpy.empty((nelem, Q, 2))
    for e, E in enumerate(Erestrict):
        xE = x[E,:]
        xq[e] = B @ xE
        dxdX = D @ xE # 2x2 matrices at each quadrature point
        det = numpy.linalg.det(dxdX)
        W[e] = w * det # Quadrature weight on physical element
        dXdx[e] = numpy.linalg.inv(dxdX)
    return xq, W, dXdx
    
