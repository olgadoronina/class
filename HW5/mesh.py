import numpy

class Mesh:
    def __init__(self, lcar=.5, shape='circle', reshape_boundary=False):
        import pygmsh
        geom = pygmsh.built_in.Geometry()
        if shape == 'circle':
            geom.add_circle((0,0,0), 1, lcar)
        elif shape == 'rectangle':
            geom.add_rectangle(-1, 1, -.5, .5, 0, lcar)
        elif shape == 'eyes':
            holes = [geom.add_circle((c,0,0), .25, .25*lcar, make_surface=False)
                 for c in (-.5, .5)]
            geom.add_circle((0,0,0), 1, lcar, holes=holes)
        else:
            raise RuntimeError('Shape not recognized:', shape)
        points, elements, _, _, _ = pygmsh.generate_mesh(geom, verbose=False, dim=2)
        vtx = points[:,:2]
        tri = elements['triangle']
        # Gmsh doesn't guarantee consistent orientation so fix up any inverted elements
        orient = numpy.cross(vtx[tri[:,1]] - vtx[tri[:,0]],
                             vtx[tri[:,2]] - vtx[tri[:,1]]) < 0
        tri[orient] = tri[orient][:,[0,2,1]]
        # Create edges
        edges = tri[:,[0,1,1,2,2,0]].reshape((-1,2))
        edges.sort(axis=1)
        ind = numpy.lexsort((edges[:,1], edges[:,0]))
        edge2vertex, starts, perm, counts = numpy.unique(edges[ind], axis=0,
                        return_index=True, return_inverse=True, return_counts=True)
        cell2edge = numpy.empty(len(edges), dtype=int)
        cell2edge[ind] = perm
        cell2edge = cell2edge.reshape((-1, 3))
        edgenumbers, edgecount = numpy.unique(cell2edge.flatten(), return_counts=True)
        edgecenter = .5*(vtx[edge2vertex[:,0]] + vtx[edge2vertex[:,1]])
        
        centroids = (vtx[tri[:,0]] + vtx[tri[:,1]] + vtx[tri[:,2]]) / 3
        h = numpy.min(numpy.linalg.norm(numpy.kron([1,1,1], centroids).reshape((-1,2))
                                        - edgecenter[cell2edge.flatten()], axis=1))
        
        # Classify boundaries
        bedges = edgenumbers[edgecount == 1]
        if shape == 'eyes':
            def distance(c, r):
                return numpy.abs(numpy.linalg.norm(edgecenter[bedges] - c, axis=1) - r)
            mouter = distance((0,0), 1)
            mleft = distance((-.5,0), .25)
            mright = distance((.5,0), .25)
            boundary = dict(outer=bedges[mouter <= numpy.minimum(mleft, mright)],
                           left=bedges[mleft <= numpy.minimum(mouter, mright)],
                           right=bedges[mright <= numpy.minimum(mleft, mouter)])
        else:
            boundary = dict(outer=bedges)
        
        self.vtx = vtx
        self.tri = tri
        self.edge2vertex = edge2vertex
        self.cell2edge = cell2edge
        self.edgecenter = edgecenter
        self.boundary = boundary
        self.shape = shape
        self.nvtx = len(vtx)
        self.nface = len(edge2vertex)
        self.h = h
        if reshape_boundary:
            self.reshape_boundary()
    
    def reshape_boundary(self):
        def project_to_circle(label, c, r):
            edges = self.boundary[label]
            x = self.edgecenter[edges]
            self.edgecenter[edges] = c + r*(x-c) / numpy.linalg.norm(x-c, axis=1)[:,None]
        if self.shape == 'circle':
            project_to_circle('outer', (0,0), 1)
        elif self.shape == 'eyes':
            project_to_circle('outer', (0,0), 1)
            project_to_circle('left', (-.5,0), .25)
            project_to_circle('right', (.5,0), .25)
            
    def tri2(self):
        _, Erestrict = self.Erestrict(2)
        return Erestrict[:,[0,3,5, 1,4,3, 2,5,4, 3,4,5]].reshape(-1,3)
    
    def Erestrict(self, p):
        if p == 1:
            return self.vtx, self.tri
        elif p == 2:
            x = numpy.vstack([self.vtx, self.edgecenter])
            Erestrict = numpy.hstack([self.tri, self.nvtx+self.cell2edge])
            return x, Erestrict
        raise RuntimeError('Not implemented for order', p)
    
    def Frestrict(self, p):
        if p == 1:
            return self.edge2vertex
        elif p == 2:
            return numpy.hstack([self.edge2vertex,
                                 self.nvtx + numpy.arange(self.nface)[:,None]])
        raise RuntimeError('Not implemented for order', p)
    
    def plotmesh(self):
        pyplot.triplot(self.vtx[:,0], self.vtx[:,1], triangles=self.tri)
        x, _ = self.Erestrict(2)
        Frestrict = self.Frestrict(2)
        for label, faces in self.boundary.items():
            xF = x[Frestrict[faces,2]]
            pyplot.plot(xF[:,0], xF[:,1], 's', label=label)
            xFv = x[Frestrict[faces,:2].flatten()]
            pyplot.plot(xFv[:,0], xFv[:,1], '.k')
        pyplot.legend()
