from Octree import *
from Point import Point3D

from random import random

print("TESTME")
ot = Octree(Point3D(0, 0, 0), Point3D(1, 1, 1), max_levels=2)


for i in range(10000):
    ot.addPoint(Point3D(random(), random(), random()))

def printInfo(node, ctr=0):
    print(f"Node: {ctr}")
    print(f"Number of points in subtree: {node.numPoints} (has_children={node.has_children})")
    print(f"Number of children: {len(node.children)}, numper of points in cell: {len(node.points)}")
    print(f"Bounds: lower={node.lower_bound} upper={node.upper_bound}")
    print(f"Center: {node.center}, feature point: {node.feature_point}")
    print(f"Max levels {node.max_levels}, max size {node.max_size}")
    print('='*10)
    ctr = ctr+1
    for node in node.children:
        ctr = printInfo(node,ctr)
    return ctr


printInfo(ot)

boundary_cells = ot.boundary_cells
print(f"Size of boundary_cells set: {len(boundary_cells)}")
fps = [ c.feature_point for c in boundary_cells ]
cts = [ c.center for c in boundary_cells ]
print(f"Feature points: {fps}")

print(f"Center points: {cts}")


#Test
for i in range(1000):
    point = Point3D(random(),random(),random())
    assert ot.check_point_contained(point)
    cell = ot.find_cell(point)
    #print(
    #    f"Test point: {point} " \
    #    f"Point in cell: {cell} (bounds: {cell.lower_bound}, {cell.upper_bound}) " \
    #    f"with center: {cell.center} and feature point: {cell.feature_point}"
    #)
    assert cell and cell.is_boundary_cell()