from collections import namedtuple
#TODO: Remove Point3D class
#TODO: make it work for 2D
#from Point import Point3D
from typing import List, Optional
import numpy as np
from collections import Counter

#TODO: remove the hard-coded limitation of 3 dimensions
def index2mask(idx: int):
    return (idx&1, (idx>>1)&1, (idx>>2)&1)

def mask2index(idx: tuple):
    x, y, z = [int(bool(idx[i])) for i in range(3)]
    return (z<<2)+(y<<1)+x

def invMask(mask: tuple):
    return tuple([mask[i]^1 for i in range(3)])

class Octree():
    """Representation of a Node in the Octree."""
    def __init__(self, lower_bound: np.array, upper_bound: np.array, 
        max_levels=-1, max_size=10,
        initial_points: np.array = np.empty((0,3))):
        """Create a node in the octree.
        
        Specify upper and lower bounds, optionally specify maximum levels to split
        maximum number of points to contain, and an intial set of points"""
        if lower_bound.shape != upper_bound.shape:
            raise RuntimeError("Lower and Upper bound do not have the same shape")
        dim = lower_bound.shape[0]
        if dim<1:
            raise RuntimeError(f"Error, dimension of point space cannot be <1")
        if dim != initial_points.shape[1]:
            raise RuntimeError(f"Dimension mismatch: lower_bound={dim} initial_points={initial_points.shape}")
        if dim>3:
            raise NotImplementedError("Sorry, dim>3 for points not yet supported")
        self.has_children = False
        self.children: List[Octree] = []
        self.points: np.array = np.empty((0,dim))
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.max_levels = max_levels
        self.max_size = max_size
        self.center = (lower_bound + upper_bound) / 2
        self.numPoints = 0
        self.cardinality = 2**dim
        assert(self.cardinality == 8)#To remove this constrain
        self.dimensions = lower_bound.shape[0]
        self.__feature_point: Optional[np.ndarray] = None
        for p in initial_points:
            self.addPoint(p)

    @property
    def feature_point(self):
        if len(self.points):
            self.__feature_point = np.mean(self.points)
        return self.__feature_point

    def extend(self) -> bool:
        if (self.max_levels == 0):
            return False
        else:
            self.has_children = True
            self.children = list([self._generateChildTree(idx) for idx in range(self.cardinality)])
            for point in self.points:
                self._addPointToChild(point)
            self.points = np.empty((0,self.dimensions))
            return True

    def _generateChildTree(self, idx):
        mask = index2mask(idx)
        imask = invMask(mask)
        new_lower_bound = self.lower_bound*imask + self.center*mask
        new_upper_bound = self.center*imask + self.upper_bound*mask
        return Octree(new_lower_bound, new_upper_bound, self.max_levels - 1, self.max_size)

    def addPoint(self, pt: np.ndarray):
        if self.has_children:
            self._addPointToChild(pt)
        elif len(self.points) >= self.max_size:
            if self.extend():
                self._addPointToChild(pt)
            else:
                self.points = np.vstack((self.points,pt))
        else:
            self.points = np.vstack((self.points,pt))
        self.numPoints += 1
       
    def _addPointToChild(self, pt: np.ndarray):
        self.children[self._get_index(pt)].addPoint(pt)

    def _get_index(self, pt):
        #TODO: remove hard-coded use of 3 dimensions
        x = int(pt[0] >= self.center[0])
        y = int(pt[1] >= self.center[1])
        z = int(pt[2] >= self.center[2])
        return (mask2index((x,y,z)))

    def is_boundary_cell(self) -> bool:
        return self.points.shape[0]>0

    @property
    def boundary_cells(self) -> List[Octree]:
        """List of boundary cells (e.g. non-empty)"""
        _bdr = [ self ] if self.is_boundary_cell() else []
        for _cell in self.children:
            _bdr += _cell.boundary_cells
        return _bdr

    def check_point_contained(self, point: np.array) -> bool:
        """Check if point is contained in cell"""
        return np.all(self.lower_bound<point) and np.all(point<self.upper_bound)        

    def find_cell(self, point: np.ndarray) -> Octree:
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

    def sample(self, k: int = 1 ) -> np.ndarray:
        """Sample k points from octree

        Sample points from octree with probabilities proportation
        to the boundary cells points density

        :return: np.ndarray.shape[k,self.points.shape[1]]
        """
        # Step 1. Sort by number of points in (boundary) cell
        _bcd = { bc: len(bc.points) for bc in self.boundary_cells}
        _sorted = sorted(_bcd.items(), key=lambda e: e[1])
        # Step 2. Select k points at random. First select at 
        # random the boundary cell with a probability proportional
        # to the number of points in it
        _bcs, _density = zip(*_sorted) #Unzip
        # Build a segment from 0 to total cumulative sum
        # ex.: density = [10,5,2] -> [10,15,17], so arandom number 
        #      of 13 falls in the second binm: index=1
        #Now 
        _ss = np.cumsum(_density) 
        #k random numbers [0,tot+1)
        _kps = np.random.randint(0,_ss[-1]+1,size=k)
        #This is the list of indexes where the random points would
        #fall e.g. such that _kps <= _ss[_indexes]
        _indexes = np.searchsorted(_ss, _kps)
        #Get the corresponding 
        _selected = np.array(_bcs)[_indexes]
        #Deal with the case when the same boundary cell is selected more than once
        #Needed because we want to get random numbers with replacement with 
        #associated points
        _dd = Counter(_selected) # { obj : cardinality }
        #For each selected boundary cell, extract one random 
        return np.vstack([ np.random.choice(bc.points.shape[0], card, replace=False) for bc,card in _dd.items() ])
