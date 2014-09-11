# -*- coding: utf-8 -*-
# Copyright (c) 2014, Vispy Development Team.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
"""
The vispy.scene namespace provides functionality for higher level
visuals as well as scenegraph and related classes.

What is a scenegraph?
=====================

In short, a scenegraph is a data structure used for organizing many graphical 
objects in a single scene. Scenegraphs are a de-facto standard for implementing 
graphics across a wide variety of contexts, including 2D and 3D
illustration software, user interface libraries, games, and the SVG standard. 

All scenegraphs share one central idea in common: a scene is defined
by a _graph_ of objects; that is, individual objects in the scene are 
organized as a network of nodes connected by edges. In this graph, each node
(called an Visual in vispy) represents some part of the scene--a circle, an
image, a block of text, a tree, etc. 

[figure]

In vispy, Visuals are arranged in a hierarchical structure such that each may 
have parent and child entities. Children inherit certain properties from their 
parents, most notably the coordinate system transformations such as scaling, 
rotation, translation, and others. So for example, moving one Visual in a scene
will cause all of its children (and grandchildren) to move together, while the
rest of the scene remains in place. 

[figure]

There are many articles on this topic:
*
*
*


Vispy's building blocks: Visuals, Cameras, and Widgets
======================================================

Perhaps the most central feature vispy provides is its library of Visuals. By
combining many different types of Visual together, it is possible to create
complex and beautiful scenes with very little effort.

To begin, we will need a ``SceneCanvas``, which is a special subclass of 
``Canvas`` that implements much of the infrastructure needed to operate a
scenegraph::

    from vispy.scene import SceneCanvas
    canvas = SceneCanvas()
    
This canvas comes with its own root Visual ``canvas.scene``. Now to build up a 
scenegraph, all we need to do is create Visuals and make them children (or
grandchildren) of ``canvas.scene``::

    from vispy.scene import visuals
    ellipse = visuals.Ellipse(pos=(100, 100), radius=(100, 50))
    ellipse.parent = canvas.scene

[figure]

Every time the canvas redraws, it traverses the enire graph of Visuals 
connected to ``canvas.scene``, calling ``visual.draw()`` for each in order. 




Coordinate systems in OpenGL and vispy
======================================

* 






How the vispy scenegraph is organized
=====================================









Terminology
-----------

* **entity** - an object that lives in the scenegraph. It can have zero or
  more children and zero or more parents (although one is recommended).
  It also has a transform that maps the local coordinate frame to the
  coordinate frame of the parent.

* **scene** - a complete connected graph of entities.

* **subscene** - the entities that are children of a viewbox. Any viewboxes
  inside this subscene are part of the subscene, but not their children.
  The SubScene class is the toplevel entity for any subscene. Each
  subscene has its own camera, lights, aspect ratio, etc.

* **visual** - an entity that has a visual representation. It can be made
  visible/invisible and also has certain bounds.

* **widget** - an entity of a certain size that provides interaction. It
  is made to live in a 2D scene with a pixel camera.

* **viewbox** - an entity that provides a rectangular window to which a
  subscene is rendered. Clipping is performed in one of several ways.

* **camera** - an entity that specifies how the subscene of a viewbox is
  rendered to the pixel grid. It determines position and orientation
  (through its transform) an projection (through a special
  transformation property). Some cameras also provide interaction (e.g.
  zooming). Although there can be multiple cameras in a subscene, each
  subscene has one active camera.

* **viewport** - as in glViewPort, a sub pixel grid in a framebuffer.

* **drawing system** - a part of the viewbox that takes care of rendering
  a subscene to the pixel grid of that viewbox.

"""

__all__ = ['SceneCanvas', 'Entity']

from .entity import Entity  # noqa
from .canvas import SceneCanvas  # noqa
from . import visuals  # noqa
from . import widgets  # noqa
from . import cameras  # noqa
from .visuals import *  # noqa
from .cameras import *  # noqa
from .transforms import *  # noqa
from .widgets import *  # noqa
