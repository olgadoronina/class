import numpy
from matplotlib import pyplot

def tri_quad4():
    q = numpy.array([[ -0.10810301816807,   -0.78379396366386  ],
                     [ -0.78379396366386,   -0.10810301816807  ],
                     [ -0.10810301816807,   -0.10810301816807  ],
                     [-0.816847572980458,   0.633695145960917  ],
                     [ 0.633695145960917,  -0.816847572980458  ],
                     [-0.816847572980458,  -0.816847572980458  ]])
    w = numpy.array([ 0.446763179356023,
                      0.446763179356023,
                      0.446763179356023,
                      0.219903487310644,
                      0.219903487310644,
                      0.219903487310644])
    return q, w
q, w = tri_quad4()

class fe2tri:
    def __init__(self, p):
        x1 = numpy.array([[-1, 1], [-1, -1], [1, -1]])
        x2 = numpy.array([[-1, 0], [0, -1], [0, 0]])
        if p == 1:
            x = x1
        elif p == 2:
            x = numpy.vstack([x1, x2])
        self.p = p
        self.xref = x
        self.q, self.w = tri_quad4() # Could use fewer points for p==1
        V, _ = self.prime(x)
        Vinv = numpy.linalg.inv(V)
        Bprime, Dprime = self.prime(q)
        self.B = Bprime @ Vinv
        self.D = Dprime @ Vinv

    def prime(self, x):
        V = numpy.ones((len(x), len(self.xref)))
        dV = numpy.zeros((len(x), 2, len(self.xref)))
        V[:,1] = x[:,0]
        V[:,2] = x[:,1]
        # dV[:,2*i] is derivative in x direction, dV[:,2*i+1] is in y-direction
        dV[:,0,1] = 1
        dV[:,1,2] = 1
        if self.p > 1:
            V[:,3] = x[:,0]**2
            V[:,4] = x[:,0]*x[:,1]
            V[:,5] = x[:,1]**2
            dV[:,0,3] = 2*x[:,0]
            dV[:,0,4] = x[:,1]
            dV[:,1,4] = x[:,0]
            dV[:,1,5] = 2*x[:,1]
        return V, dV
    
    def meshref(self):
        # Mesh for plotting on reference element
        x1 = numpy.linspace(-1, 1)
        xx, yy = numpy.meshgrid(x1, x1)
        for i,y in enumerate(yy):
            xx[i] = numpy.linspace(-1, -y[0])
        return numpy.vstack([xx.flatten(), yy.flatten()]).T
    
    def plot(self):
        pyplot.plot(self.xref[:,0], self.xref[:,1], 'o')
        pyplot.plot(self.q[:,0], self.q[:,1], 's')
        pyplot.triplot([-1, -1, 1], [1, -1, -1])
        
        X = self.meshref()
        Vinv = numpy.linalg.inv(self.prime(self.xref)[0])
        Bprime = self.prime(X)[0]
        B = Bprime @ Vinv
        pyplot.figure()
        for i in range(6):
            from matplotlib import cm
            pyplot.subplot(2, 3, i+1)
            pyplot.tricontourf(X[:,0], X[:,1], B[:,i], 30, cmap=cm.seismic, vmin=-1, vmax=1)
