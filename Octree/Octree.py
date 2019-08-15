from collections import namedtuple
#TODO: Remove Point3D class
#TODO: make it work for 2D
from Point import Point3D
from typing import List
import numpy as np

def index2mask(idx: int):
    return (idx&1, (idx>>1)&1, (idx>>2)&1)

def mask2index(idx: tuple):
    x, y, z = [int(bool(idx[i])) for i in range(3)]
    return (z<<2)+(y<<1)+x

def invMask(mask: tuple):
    return tuple([mask[i]^1 for i in range(3)])

class Octree():
    """Representation of a Node in the Octree."""
    def __init__(self, lower_bound: Point3D, upper_bound: Point3D, max_levels=-1, max_size=10):
        self.has_children = False
        self.children: List[Octree] = []
        self.points: List[Point3D] = []
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.max_levels = max_levels
        self.max_size = max_size
        self.center = (lower_bound + upper_bound) / 2
        self.numPoints = 0
        self.__feature_point = None

    @property
    def feature_point(self):
        if len(self.points):
            self.__feature_point = np.mean(self.points)
        return self.__feature_point

    def extend(self):
        if (self.max_levels == 0):
            return False
        else:
            self.has_children = True
            self.children = list([self._generateChildTree(idx) for idx in range(8)])
            for point in self.points:
                self._addPointToChild(point)
            self.points = []
            return True

    def _generateChildTree(self, idx):
        mask = index2mask(idx)
        imask = invMask(mask)
        new_lower_bound = self.lower_bound*imask + self.center*mask
        new_upper_bound = self.center*imask + self.upper_bound*mask
        return Octree(new_lower_bound, new_upper_bound, self.max_levels - 1, self.max_size)

    def addPoint(self, pt: Point3D):
        if self.has_children:
            self._addPointToChild(pt)
        elif len(self.points) >= self.max_size:
            if self.extend():
                self._addPointToChild(pt)
            else:
                self.points.append(pt)
        else:
            self.points.append(pt)
        self.numPoints += 1
       
    def _addPointToChild(self, pt: Point3D):
        self.children[self._get_index(pt)].addPoint(pt)

    def _get_index(self, pt):
        x = int(pt.x >= self.center.x)
        y = int(pt.y >= self.center.y)
        z = int(pt.z >= self.center.z)
        return (mask2index((x,y,z)))

    def is_boundary_cell(self):
        return (len(self.points)>0)

    @property
    def boundary_cells(self):
        """List of boundary cells (e.g. non-empty)"""
        _bdr = [ self ] if self.is_boundary_cell() else []
        for _cell in self.children:
            _bdr += _cell.boundary_cells
        return _bdr

    def check_point_contained(self, point):
        """Check if point is contained in cell"""
        x,y,z = point.x,point.y,point.z
        lb, ub = self.lower_bound, self.upper_bound
        return (lb.x <= x) and (x <= ub.x ) and \
               (lb.y <= y ) and (y <= ub.y) and \
               (lb.z <= z) and (z<=ub.z)

    def find_cell(self, point):
        """Find cell cotaining point"""
        #print(self.center)
        if self.has_children:
            index = self._get_index(point)
            #print(f"Index for deeper search: {index}")
            return self.children[index].find_cell(point)
        else:
            #This fails only if point is outside of octree
            assert self.check_point_contained(point)
            return self
