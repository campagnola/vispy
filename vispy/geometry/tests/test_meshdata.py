# -*- coding: utf-8 -*-
# Copyright (c) 2014, Vispy Development Team.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.

import numpy as np
from numpy.testing import assert_array_equal, assert_allclose

from vispy.geometry.meshdata import MeshData


def assert_array_set_equal(a, b):
    """Check the set of rows in a is the same as in b"""
    a = a.reshape((a.size/a.shape[-1], a.shape[-1]))
    a = [tuple(x) for x in a]
    b = b.reshape((b.size/b.shape[-1], b.shape[-1]))
    b = [tuple(x) for x in b]
    assert set(a) == set(b)

def test_meshdata():
    """ Test MeshData class
    """
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 1]],
                               dtype=np.float)
    vert_copy = vertices.copy()
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint)
    face_copy = faces.copy()
    sq2 = 2. ** -0.5
    vert_normals = np.array([[sq2, 0, sq2], [0, 0, 1], [sq2, 0, sq2], [1, 0, 0]],
                              dtype=np.float)
    face_normals = np.array([[0, 0, 1], [1, 0, 0]],
                              dtype=np.float)
    edges = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [2, 3]],
                            dtype=np.uint)

    #
    # Test (vertices, faces)
    #
    mesh = MeshData(vertices=vertices, faces=faces)
    
    # test vertices are returned untouched
    assert mesh.vertices() is vertices
    assert_array_equal(vert_copy, mesh.vertices())
    
    # test faces are returned untouched
    assert mesh.faces() is faces
    assert_array_equal(face_copy, mesh.faces())
    
    # test conversion to face-indexed vertices
    assert not mesh.has_face_indexed_data()
    assert_array_equal(mesh.vertices(indexed='faces'), vertices[faces])
    assert mesh.has_face_indexed_data()
    
    # test normal calculation
    assert_allclose(vert_normals, mesh.vertex_normals())
    assert_allclose(vert_normals[faces], mesh.vertex_normals(indexed='faces'))
    assert_allclose(face_normals, mesh.face_normals())
    
    # test edge calculation
    assert_array_set_equal(edges, mesh.edges())
    face_edges = mesh.vertices(indexed='edges')
    assert_array_set_equal(vertices[edges], face_edges)


    #
    # Test (vertices)
    #
    face_vertices = vertices[faces]
    fv_copy = face_vertices.copy()
    mesh = MeshData(vertices=face_vertices)
    assert mesh.has_face_indexed_data()

    # test indexed vertices are returned untouched
    assert mesh.vertices(indexed='faces') is face_vertices
    assert_array_equal(fv_copy, mesh.vertices(indexed='faces'))
    
    # Test back-conversion to unindexed vertices
    assert_array_equal(mesh.vertices(), vertices)
    
    
    #
    # Test edges
    #
    face_vertices_flat = face_vertices.reshape((6, 3))
    edges2 = np.array([[0, 1], [0, 2], [2, 3]],
                            dtype=np.uint)
    mesh = MeshData(vertices=face_vertices, edges=edges2)
    assert mesh.edges(indexed='faces') is edges2
    assert_array_set_equal(face_vertices_flat[edges2], 
        mesh.vertices()[mesh.edges()])
    
    mesh = MeshData(vertices=vertices, faces=faces, edges=edges2)
    assert mesh.edges() is edges
    assert_array_set_equal(vertices[edges2], 
        mesh.vertices(indexed='faces')[mesh.edges(indexed='faces')])
    
    """
    Edges: what could we reasonably want to do here? 
    
    * Auto-compute edges from a mesh and draw indexed lines:
      - mesh.edges() returns an INDEX array into mesh.vertexes().
        - If vertex buffer is (Nv, 3), then edge index must be (Ne, 2)
            => edges(indexed='faces') returns (Ne, 2)
        - If vertex buffer is (Nf, 3, 3), then edge index must again be (Ne, 2)
            => edges() returns (Ne, 2)
    * Auto-compute edges and draw unindexed lines:
        - mesh.vertexes(indexed='edges') returns (Ne, 2, 3)
    * User specifies indexed edges:
        MeshData(vertexes, indexed_edges=...)
    * User specifies unindexed edges:
        MeshData(edges=...)
    
    """