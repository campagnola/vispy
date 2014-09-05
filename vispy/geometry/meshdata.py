# -*- coding: utf-8 -*-
# Copyright (c) 2014, Vispy Development Team.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.

import numpy as np

from ..ext.six.moves import xrange


class MeshData(object):
    """
    Class for storing and operating on 3D mesh data.

    Parameters
    ----------
    vertices : ndarray, shape (Nv, 3)
        Vertex coordinates. If faces is not specified, then this will
        instead be interpreted as (Nf, 3, 3) array of coordinates.
    faces : ndarray, shape (Nf, 3)
        Indices into the vertex array. Note that if this argument is None,
        then _all_ other arguments are assumed to be pre-indexed by face.
    edges : ndarray, shape (Ne, 2)
        Indices into the vertex array.
    vertex_colors : ndarray, shape (Nv, 4)
        Vertex colors. If faces is not specified, this will be
        interpreted as (Nf, 3, 4) array of colors.
    face_colors : ndarray, shape (Nf, 4)
        Face colors.

    Notes
    -----
    All arguments are optional.

    The object may contain:

    - list of vertex locations
    - list of edges
    - list of triangles
    - colors per vertex, edge, or tri
    - normals per vertex or tri

    This class handles conversion between the standard
    [list of vertices, list of faces] format (suitable for use with
    glDrawElements) and 'indexed' [list of vertices] format (suitable
    for use with glDrawArrays). It will automatically compute face normal
    vectors as well as averaged vertex normal vectors.

    The class attempts to be as efficient as possible in caching conversion
    results and avoiding unnecessary conversions.
    
    
    Instantiation Examples
    ----------------------
    
    Vertex array with three (x,y,z) points per face::
    
        MeshData(vertices=(Nf, 3, 3))
    
    Vertex array with accompanying face index::
    
        MeshData(vertices=(Nv, 3), faces=(Nf, 3))
    
    Vertex array with accompanying face and edge indices::
        
        MeshData(vertices=(Nv, 3), faces=(Nf, 3), edges=(Ne, 2))

            
    Usage Examples
    --------------
    
    With vertices indexed by faces::
    
        verts = meshdata.vertexes(indexed='faces')
        edges = meshdata.edges(indexed='faces')
        program['a_position'] = VertexBuffer(verts)
        program.draw(GL_TRIANGLES)
        program.draw(GL_LINES, index=IndexBuffer(edges))

    With vertices indexed by edges::
    
        verts = meshdata.vertexes(indexed='edges')
        program['a_position'] = VertexBuffer(verts)
        program.draw(GL_LINES)
    
    With unindexed vertices::
    
        verts = meshdata.vertexes()
        faces = meshdata.faces()
        edges = meshdata.edges()
        program['a_position'] = VertexBuffer(verts)
        program.draw(GL_TRIANGLES, index=IndexBuffer(faces))
        program.draw(GL_LINES, index=IndexBuffer(edges))
    
    With per-vertex normals (smooth)::
    
        verts = meshdata.vertexes()
        faces = meshdata.faces()
        norms = meshdata.vertex_normals()
        program['a_position'] = VertexBuffer(verts)
        program['a_normal'] = VertexBuffer(norms)
        program.draw(GL_TRIANGLES, index=IndexBuffer(faces))
    
    With per-face normals (faceted)::
    
        verts = meshdata.vertexes(indexed='faces')
        norms = meshdata.face_normals(indexed='faces')
        program['a_position'] = VertexBuffer(verts)
        program['a_normal'] = VertexBuffer(norms)
        program.draw(GL_TRIANGLES)
    
    """

    def __init__(self, vertices=None, faces=None, edges=None,
                 vertex_colors=None, face_colors=None):
        self._vertices = None  # (Nv,3) array of vertex coordinates
        self._vertices_indexed_by_faces = None  # (Nf, 3, 3) vertex coordinates
        self._vertices_indexed_by_edges = None  # (Ne, 2, 3) vertex coordinates

        # mappings between vertices, faces, and edges
        self._faces = None  # Nx3 indices into self._vertices, 3 verts/face
        self._edges = None  # Nx2 indices into self._vertices, 2 verts/edge
        self._edges_indexed_by_faces = None  # (Ne, 2) indices into
        # self.vertices(indexed='faces'), 2 verts/edge
        
        # inverse mappings
        self._vertex_faces = None  # maps vertex ID to a list of face IDs
        self._vertex_edges = None  # maps vertex ID to a list of edge IDs

        # Per-vertex data
        self._vertex_normals = None                # (Nv, 3) normals
        self._vertex_normals_indexed_by_faces = None  # (Nf, 3, 3) normals
        self._vertex_colors = None                 # (Nv, 3) colors
        self._vertex_colors_indexed_by_faces = None   # (Nf, 3, 4) colors
        self._vertex_colors_indexed_by_edges = None   # (Nf, 2, 4) colors

        # Per-face data
        self._face_normals = None                # (Nf, 3) face normals
        self._face_normals_indexed_by_faces = None  # (Nf, 3, 3) face normals
        self._face_colors = None                 # (Nf, 4) face colors
        self._face_colors_indexed_by_faces = None   # (Nf, 3, 4) face colors
        self._face_colors_indexed_by_edges = None   # (Ne, 2, 4) face colors

        # Per-edge data
        self._edge_colors = None                # (Ne, 4) edge colors
        self._edge_colors_indexed_by_edges = None  # (Ne, 2, 4) edge colors
        # default color to use if no face/edge/vertex colors are given
        # self._meshColor = (1, 1, 1, 0.1)

        if faces is None:
            index = 'faces'
        else:
            self.set_faces(faces)
            index = None
        
        if vertices is not None:
            self.set_vertices(vertices, indexed=index)
        if edges is not None:
            self.set_edges(edges, indexed=index)
        if vertex_colors is not None:
            self.set_vertex_colors(vertex_colors, indexed=index)
        if face_colors is not None:
            self.set_face_colors(face_colors, indexed=index)

    def faces(self, indexed=None):
        """Array of vertex indices with shape (Nfaces, 3). The values are
        indexes into the array returned by ``self.vertexes()``.
        """
        return self._faces
        
    def edges(self, indexed=None):
        """Array of vertex indices with shape (Nedges, 2). If edges were not
        provided at instantiation or by calling set_edges(), then they will be
        computed by generating three edges per face and removing duplicates.
        
        Parameters
        ----------
        indexed : None or 'faces'
            * If None, then the array will have shape (Nedges, 2) and its 
              values are meant to index the vertex array generated by
              ``self.vertexes()``.
            * If 'faces', then the array will have shape (Nfaces, 3, 2) and its 
              values are meant to index the vertex array generated by
              ``self.vertexes(indexed='faces')``.
        
        """
        if indexed is None:
            if self._edges is None:
                self._compute_edges(indexed=None)
            return self._edges
        elif indexed == 'faces':
            if self._edges_indexed_by_faces is None:
                self._compute_edges(indexed='faces')
            return self._edges_indexed_by_faces
        else:
            raise Exception("Invalid indexing mode. Accepts: None, 'faces'")

    def set_faces(self, faces):
        """Set the (Nf, 3) array of faces. Each rown in the array contains
        three indices into the vertex array, specifying the three corners
        of a triangular face."""
        self._faces = faces
        self._edges = None
        self._edges_indexed_by_faces = None
        self._vertex_faces = None
        self._vertices_indexed_by_faces = None
        self._vertices_indexed_by_edges = None
        self.reset_normals()
        self._vertex_colors_indexed_by_faces = None
        self._face_colors_indexed_by_faces = None

    def set_edges(self, edges, indexed=None):
        """Set the array of edge indexes.
        
        Parameters
        ----------
        edges : array
            The array of edge indexes. Must have shape (Nedges, 2).
        indexed : None or 'faces'
            Specifies whether the array values are indexes into the non-indexed
            vertex array returned by ``self.vertexes()``, or the face-indexed 
            vertex array returned by ``self.vertexes(indexed='faces')``.
        """
        if indexed is None:
            self._edges = edges
            self._edges_indexed_by_faces = None
        elif indexed == 'faces':
            self._edges = None
            self._edges_indexed_by_faces = edges
        else:
            raise Exception("Invalid indexing mode. Accepts: None, 'faces'")

    def vertices(self, indexed=None):
        """Return an array (N,3) of the positions of vertices in the mesh.
        By default, each unique vertex appears only once in the array.
        If indexed is 'faces', then the array will instead contain three
        vertices per face in the mesh (and a single vertex may appear more
        than once in the array)."""
        if indexed is None:
            if (self._vertices is None and
                    self._vertices_indexed_by_faces is not None):
                self._compute_unindexed_vertices()
            return self._vertices
        elif indexed == 'faces':
            if (self._vertices_indexed_by_faces is None and
                    self._vertices is not None):
                self._vertices_indexed_by_faces = self._vertices[self.faces()]
            return self._vertices_indexed_by_faces
        elif indexed == 'edges':
            if (self._vertices_indexed_by_edges is None and
                    self._vertices is not None):
                self._vertices_indexed_by_edges = self._vertices[self.edges()]
            return self._vertices_indexed_by_edges
        else:
            raise Exception("Invalid indexing mode. Accepts: None, 'faces'")

    def set_vertices(self, verts=None, indexed=None, reset_normals=True):
        """
        Set the array (Nv, 3) of vertex coordinates.
        If indexed=='faces', then the data must have shape (Nf, 3, 3) and is
        assumed to be already indexed as a list of faces.
        This will cause any pre-existing normal vectors to be cleared
        unless reset_normals=False.
        """
        if indexed is None:
            if verts is not None:
                self._vertices = verts
            self._vertices_indexed_by_faces = None
        elif indexed == 'faces':
            self._vertices = None
            if verts is not None:
                self._vertices_indexed_by_faces = verts
        else:
            raise Exception("Invalid indexing mode. Accepts: None, 'faces'")

        if reset_normals:
            self.reset_normals()

    def reset_normals(self):
        self._vertex_normals = None
        self._vertex_normals_indexed_by_faces = None
        self._face_normals = None
        self._face_normals_indexed_by_faces = None

    def has_face_indexed_data(self):
        """Return True if this object already has vertex positions indexed
        by face"""
        return self._vertices_indexed_by_faces is not None

    def has_edge_indexed_data(self):
        return self._vertices_indexed_by_edges is not None

    def has_vertex_color(self):
        """Return True if this data set has vertex color information"""
        for v in (self._vertex_colors, self._vertex_colors_indexed_by_faces,
                  self._vertex_colors_indexed_by_edges):
            if v is not None:
                return True
        return False

    def has_face_color(self):
        """Return True if this data set has face color information"""
        for v in (self._face_colors, self._face_colors_indexed_by_faces,
                  self._face_colors_indexed_by_edges):
            if v is not None:
                return True
        return False

    def face_normals(self, indexed=None):
        """
        Return an array (Nf, 3) of normal vectors for each face.
        If indexed='faces', then instead return an indexed array
        (Nf, 3, 3)  (this is just the same array with each vector
        copied three times).
        """
        if self._face_normals is None:
            v = self.vertices(indexed='faces')
            self._face_normals = np.cross(v[:, 1] - v[:, 0],
                                          v[:, 2] - v[:, 0])

        if indexed is None:
            return self._face_normals
        elif indexed == 'faces':
            if self._face_normals_indexed_by_faces is None:
                norms = np.empty((self._face_normals.shape[0], 3, 3),
                                 dtype=np.float32)
                norms[:] = self._face_normals[:, np.newaxis, :]
                self._face_normals_indexed_by_faces = norms
            return self._face_normals_indexed_by_faces
        else:
            raise Exception("Invalid indexing mode. Accepts: None, 'faces'")

    def vertex_normals(self, indexed=None):
        """
        Return an array of normal vectors.
        By default, the array will be (N, 3) with one entry per unique
        vertex in the mesh. If indexed is 'faces', then the array will
        contain three normal vectors per face (and some vertices may be
        repeated).
        """
        if self._vertex_normals is None:
            faceNorms = self.face_normals()
            vertFaces = self.vertex_faces()
            self._vertex_normals = np.empty(self._vertices.shape,
                                            dtype=np.float32)
            for vindex in xrange(self._vertices.shape[0]):
                faces = vertFaces[vindex]
                if len(faces) == 0:
                    self._vertex_normals[vindex] = (0, 0, 0)
                    continue
                norms = faceNorms[faces]  # get all face normals
                norm = norms.sum(axis=0)  # sum normals
                norm /= (norm**2).sum()**0.5  # and re-normalize
                self._vertex_normals[vindex] = norm

        if indexed is None:
            return self._vertex_normals
        elif indexed == 'faces':
            return self._vertex_normals[self.faces()]
        else:
            raise Exception("Invalid indexing mode. Accepts: None, 'faces'")

    def vertex_colors(self, indexed=None):
        """
        Return an array (Nv, 4) of vertex colors.
        If indexed=='faces', then instead return an indexed array
        (Nf, 3, 4).
        """
        if indexed is None:
            return self._vertex_colors
        elif indexed == 'faces':
            if self._vertex_colors_indexed_by_faces is None:
                self._vertex_colors_indexed_by_faces = \
                    self._vertex_colors[self.faces()]
            return self._vertex_colors_indexed_by_faces
        else:
            raise Exception("Invalid indexing mode. Accepts: None, 'faces'")

    def set_vertex_colors(self, colors, indexed=None):
        """
        Set the vertex color array (Nv, 4).
        If indexed=='faces', then the array will be interpreted
        as indexed and should have shape (Nf, 3, 4)
        """
        if indexed is None:
            self._vertex_colors = colors
            self._vertex_colors_indexed_by_faces = None
        elif indexed == 'faces':
            self._vertex_colors = None
            self._vertex_colors_indexed_by_faces = colors
        else:
            raise Exception("Invalid indexing mode. Accepts: None, 'faces'")

    def face_colors(self, indexed=None):
        """
        Return an array (Nf, 4) of face colors.
        If indexed=='faces', then instead return an indexed array
        (Nf, 3, 4)  (note this is just the same array with each color
        repeated three times).
        """
        if indexed is None:
            return self._face_colors
        elif indexed == 'faces':
            if (self._face_colors_indexed_by_faces is None and
                    self._face_colors is not None):
                Nf = self._face_colors.shape[0]
                self._face_colors_indexed_by_faces = \
                    np.empty((Nf, 3, 4), dtype=self._face_colors.dtype)
                self._face_colors_indexed_by_faces[:] = \
                    self._face_colors.reshape(Nf, 1, 4)
            return self._face_colors_indexed_by_faces
        else:
            raise Exception("Invalid indexing mode. Accepts: None, 'faces'")

    def set_face_colors(self, colors, indexed=None):
        """
        Set the face color array (Nf, 4).
        If indexed=='faces', then the array will be interpreted
        as indexed and should have shape (Nf, 3, 4)
        """
        if indexed is None:
            self._face_colors = colors
            self._face_colors_indexed_by_faces = None
        elif indexed == 'faces':
            self._face_colors = None
            self._face_colors_indexed_by_faces = colors
        else:
            raise Exception("Invalid indexing mode. Accepts: None, 'faces'")

    def face_count(self):
        """
        Return the number of faces in the mesh.
        """
        if self._faces is not None:
            return self._faces.shape[0]
        elif self._vertices_indexed_by_faces is not None:
            return self._vertices_indexed_by_faces.shape[0]

    def edge_colors(self):
        return self._edge_colors

    def _compute_unindexed_vertices(self):
        # Given (Nv, 3, 3) array of vertices-indexed-by-face, convert
        # backward to unindexed vertices
        # This is done by collapsing into a list of 'unique' vertices
        # (difference < 1e-14)

        # I think generally this should be discouraged..
        faces = self._vertices_indexed_by_faces
        verts = {}  # used to remember the index of each vertex position
        self._faces = np.empty(faces.shape[:2], dtype=np.uint)
        self._vertices = []
        self._vertex_faces = []
        self._face_normals = None
        self._vertex_normals = None
        for i in xrange(faces.shape[0]):
            face = faces[i]
            for j in range(face.shape[0]):
                pt = face[j]
                # quantize to ensure nearly-identical points will be merged
                pt2 = tuple([round(x*1e14) for x in pt])
                index = verts.get(pt2, None)
                if index is None:
                    self._vertices.append(pt)
                    self._vertex_faces.append([])
                    index = len(self._vertices)-1
                    verts[pt2] = index
                # track which vertices belong to which faces
                self._vertex_faces[index].append(i)
                self._faces[i, j] = index
        self._vertices = np.array(self._vertices, dtype=np.float32)

    def vertex_faces(self):
        """
        List mapping each vertex index to a list of face indices that use it.
        """
        if self._vertex_faces is None:
            self._vertex_faces = [[] for i in xrange(len(self.vertices()))]
            for i in xrange(self._faces.shape[0]):
                face = self._faces[i]
                for ind in face:
                    self._vertex_faces[ind].append(i)
        return self._vertex_faces

    def _compute_edges(self, indexed=None):
        if indexed is None:
            # return array that indexes into self.vertexes()
            # with shape (Ne, 2)
            if self._faces is None:
                # no face index exists; need to generate from pre-indexed
                # vertices
                self._compute_unindexed_vertices()
            
            # generate self._edges from self._faces
            ne = 3 * self._faces.shape[0]
            dtype=self._faces.dtype
            edges = np.empty((ne, 2), dtype=dtype)
            edges[0::3] = self._faces[:, :2]
            edges[1::3] = self._faces[:, 1:3]
            edges[2::3, 0] = self._faces[:, 2]
            edges[2::3, 1] = self._faces[:, 0]
            
            # sort per-edge and remove duplicate entries
            edges.sort(axis=1)
            edges = edges.view(dtype=[('i', dtype, 2)])
            self._edges = np.ascontiguousarray(
                np.unique(edges)[:, np.newaxis]).view(dtype)
            
        elif indexed == 'faces':
            # return array that indexes into self.vertexes(indexed='faces')
            # with shape (Nf, 3, 2)
            
            # Generate face-indexed vertices if these do not exist yet
            verts = self.vertices(indexed='faces')
            
            
            if self._edges is None
                # generate self._edges_indexed_by_faces from 
                # self._vertexes_indexed_by_faces
                ne = 3 * verts.shape[0]
                edges = np.empty((ne, 2), dtype=np.uint)
                edges[:, 0] = np.arange(ne)
                edges[0::3, 1] = edges[1::3, 0]
                edges[1::3, 1] = edges[2::3, 0]
                edges[2::3, 1] = edges[0::3, 0]
                self._edges_indexed_by_faces = edges
                
            else:
                # generate self._edges_indexed_by_faces from self._edges
                
                
        else:
            raise Exception("Invalid indexing mode. Accepts: None, 'faces'")

    def save(self):
        """Serialize this mesh to a string appropriate for disk storage"""
        import pickle
        if self._faces is not None:
            names = ['_vertices', '_faces']
        else:
            names = ['_vertices_indexed_by_faces']

        if self._vertex_colors is not None:
            names.append('_vertex_colors')
        elif self._vertex_colors_indexed_by_faces is not None:
            names.append('_vertex_colors_indexed_by_faces')

        if self._face_colors is not None:
            names.append('_face_colors')
        elif self._face_colors_indexed_by_faces is not None:
            names.append('_face_colors_indexed_by_faces')

        state = dict([(n, getattr(self, n)) for n in names])
        return pickle.dumps(state)

    def restore(self, state):
        """Restore the state of a mesh previously saved using save()"""
        import pickle
        state = pickle.loads(state)
        for k in state:
            if isinstance(state[k], list):
                state[k] = np.array(state[k])
            setattr(self, k, state[k])
