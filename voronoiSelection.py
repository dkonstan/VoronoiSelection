import numpy as np
import numba
import MDAnalysis as mda
from freud.locality import Voronoi
from freud.box import Box


def wrapSlab(top, traj, out=None, start=0, stop=None, step=1):
    """
    this is a universal wrapping function for vacuum-containing systems
    it always works and it applies different wrapping routines for different cases

    top: str
        pdb or MD topology
    traj: str
        MD trajectory
    out: str (default = None)
        output trajectory (should be .dcd)
    start: int (default = 0)
        starting frame
    stop: int (default = None)
        stop frame
    step: int (default = 1)
        how many frames to skip
    """

    mda.warnings.simplefilter("ignore")
    system = mda.Universe(top, traj)
    stem = traj.split(".")[0]
    if out is None:
        out = f"{stem}_wrapped_onepiece.dcd"

    def wrapSlabEven(system, pos):
        # for when system split roughly evenly [--------         --------] and C.O.G is in the vacuum
        newPos = pos.copy()
        originalMinPos = np.min(newPos[:, 2])
        newPos[:, 2] += system.dimensions[2] / 2
        displacement = newPos[:, 2] - originalMinPos
        np.copyto(newPos[:, 2], newPos[:, 2] - system.dimensions[2], where=displacement > system.dimensions[2])

        return newPos

    def wrapSlabABitHanging(system, pos):
        # for when system split unevenly [--             -------------] and C.O.G. is in the larger piece
        newPos = pos.copy()
        minZ = np.min(newPos[:, 2])
        pos[:, 2] -= minZ
        system.atoms.positions = newPos

        midlineZ = system.atoms.centroid()[2]
        midlineShouldBe = system.dimensions[2] / 2

        newPos = system.atoms.positions.copy()
        newPos[:, 2] -= midlineZ - midlineShouldBe

        np.copyto(newPos[:, 2], newPos[:, 2] + system.dimensions[2], where=newPos[:, 2] < 0)
        np.copyto(newPos[:, 2], newPos[:, 2] - system.dimensions[2], where=newPos[:, 2] > system.dimensions[2])

        return newPos

    @numba.njit()
    def isZeroOnlyInMiddle(arr):
        if arr[0] == 0 or arr[-1] == 0:  # if edges are zero, false
            return False
        else:
            zerosInARow = 0
            sentinel = 0
            zerosInARow = []
            for i in range(1, len(arr) - 1):  # if middle has zero, true
                if sentinel == 0 and arr[i] == 0:
                    zerosInARow.append(1)
                    sentinel = 1
                elif sentinel == 1 and arr[i] == 0:
                    zerosInARow[-1] += 1
                elif sentinel == 1 and arr[i] != 0:
                    sentinel = 0

            if len(zerosInARow) > 0 and max(zerosInARow) > 1:
                return True
            else:
                return False  # if middle has no zero, false

    with mda.Writer(out, n_atoms=system.atoms.n_atoms) as dcd:
        for cnt, ts in enumerate(system.trajectory[start:stop:step]):
            print(f"wrapping frame {cnt + 1} of {len(system.trajectory[start:stop:step])}...", flush=True)
            pos = system.atoms.positions.copy()
            posZ = pos[:, 2]

            nBins = int(np.max(posZ) - np.min(posZ))
            binEdges = [np.min(posZ) + 1.0 * i for i in range(nBins)]
            cnt, vals = np.histogram(posZ, binEdges)  # ~1 Å resolution

            if isZeroOnlyInMiddle(cnt) is True:
                centroidZ = np.mean(posZ)
                idx = np.digitize(centroidZ, binEdges)
                if cnt[idx] == 0:
                    print("running slab even!")
                    newPos = wrapSlabEven(system, pos)

                else:
                    print("running slab hanging!")
                    newPos = wrapSlabABitHanging(system, pos)
            else:
                print("doing nothing!")
                newPos = system.atoms.positions.copy()

            system.atoms.positions = newPos
            dcd.write(system.atoms)


def prepareVirtualPointsAboveSurface(system, meanDistance=1.0):
    """
    This routine adds two layers of virtual points to the END of an existing atomGroup
    The points are placed above the highest Z coordinate of the system,
    so these points can be used to get the Voronoi surface by making their indices
    the startingPointsIdx argument in the VoronoiSelection class

    system: MDAnalysis.Universe
        original universe to which you want to add virtual points
    meanDistance: float
        appproximate spacing between points in Å, must be at least 1.0

    Returns
    -------
    allPoints: ndarray
        points with the virtual points at the end
    bottomLayerIdx: ndarray
        the indices corresponding to the bottom layer of the virtual grid
    topLayerIdx: ndarray
        the indices corresponding to the top layer of the virtual grid
    """

    print("warning: is your trajectory wrapped into one piece? That is required!", flush=True)

    if meanDistance < 1.0:
        raise ValueError("meanDistance must be at least 1.0 Å - there is no need for a tighter grid")

    atomPositions = system.atoms.positions
    highestZ = np.max(atomPositions[:, 2])

    minX = np.min(atomPositions[:, 0])
    maxX = np.max(atomPositions[:, 0])
    minY = np.min(atomPositions[:, 1])
    maxY = np.max(atomPositions[:, 1])

    nPointsX = int((maxX - minX) // meanDistance)
    nPointsY = int((maxY - minY) // meanDistance)

    gridX = np.linspace(minX, maxX, nPointsX)
    gridY = np.linspace(minY, maxY, nPointsY)

    grid1, grid2 = np.meshgrid(gridX, gridY)
    grid1 = np.ravel(grid1)
    grid2 = np.ravel(grid2)
    gridA = np.empty((nPointsX * nPointsY, 2))

    # need to displace each point from strict grid or else voro++ crashes
    gridA[:, 0] = grid1 + (np.random.rand(nPointsX * nPointsY) - 0.5) * 0.2 * meanDistance
    gridA[:, 1] = grid2 + (np.random.rand(nPointsX * nPointsY) - 0.5) * 0.2 * meanDistance

    gridA = np.hstack((gridA, np.reshape(np.ones(nPointsX * nPointsY) * (highestZ + 2.0), (nPointsX * nPointsY, 1))))

    gridB = np.empty((nPointsX * nPointsY, 2))
    gridB[:, 0] = grid1 + (np.random.rand(nPointsX * nPointsY) - 0.5) * 0.2 * meanDistance
    gridB[:, 1] = grid2 + (np.random.rand(nPointsX * nPointsY) - 0.5) * 0.2 * meanDistance

    # need a second grid above the first to prevent wrap-around interactions
    gridB = np.hstack((gridB, np.reshape(np.ones(nPointsX * nPointsY) * (highestZ + 3.0), (nPointsX * nPointsY, 1))))

    virtualPos = np.array(gridA)
    virtualPos2 = np.array(gridB)

    # added this 2023/04/11 because voro++ still occasionally crashes
    virtualPos[:, 2] += (np.random.rand(len(virtualPos)) - 0.5) * 0.2 * meanDistance
    virtualPos2[:, 2] += (np.random.rand(len(virtualPos2)) - 0.5) * 0.2 * meanDistance

    # add virtual points to the end so all atom indices stay valid later on
    allPoints = np.vstack((system.atoms.positions, virtualPos, virtualPos2))
    bottomLayerIdx = np.arange(len(system.atoms), len(system.atoms) + (nPointsX * nPointsY))
    topLayerIdx = np.arange(len(system.atoms) + (nPointsX * nPointsY), len(allPoints))

    return allPoints, bottomLayerIdx, topLayerIdx


class VoronoiSelection(object):
    """
    This class automates the generation of Voronoi tessellation selections of molecular
    or other systems of points. The main method of the class is getNeighbors(), which
    returns another VoronoiSelection but with memory of the atoms/points already selected
    so that getNeighbors() calls can be chained to generate a set of shells around a
    region or layers starting from virtual points above the system

    - All arguments must be keyword arguments for clarity
    - the Voronoi tessellation itself is only done once per structure and the resulting
        dictionary of neighbors is passed into the next object generated by getNeighbors()

    The main data the class works with is:
        1. a set of 3D points ("points")
        2. a set of indices ("startingPointsIdx") corresponding to the initial region
            around which you are selecting (can be virtual points)
        3. a running tally of already-selected regions ("alreadySelected"), automatic if
            "rememberInner" is True (default) but can be reset to the original region at any point
        4. optionally, an MDAnalysis atom group with topology info ("atomInfo", required for residue-based selection)

        occasional messages like "Order 3 vertex memory scaled up to 512" are from voro++ and can be ignored
        they most likely mean that the residual regularity of the virtual grid is stressing the algorithm
        even though the points have been moved randomly in x- and y-directions
    """

    def __init__(self, *, startingPointsIdx=None, points=None, atomInfo=None, dim=None,
                 rememberInner=True, neighborList=None, alreadySelected=None,
                 **kwargs):

        self.__startingPointsIdx = startingPointsIdx
        self.__points = points
        self.__atomInfo = atomInfo
        self.__dim = dim
        self.__rememberInner = rememberInner
        # initially None, gets filled at the first chance with a dict of neighbors
        # this way the Voronoi tesselation gets done only once
        self.__neighborList = neighborList

        if alreadySelected is None:
            try:
                self.__alreadySelected = set(tuple(self.__startingPointsIdx))
            except TypeError:
                self.__alreadySelected = set()
        else:
            self.__alreadySelected = set(alreadySelected)

        if self.__atomInfo is not None and not isinstance(self.__atomInfo, mda.AtomGroup):
            raise ValueError("VoronoiSelection: atomInfo must be an MDAnalysis AtomGroup")

        if self.__points is not None:
            if len(self.__points) < 2:
                raise ValueError("VoronoiSelection: you must provide at least two points")
            if self.__points.shape[1] != 3:
                raise ValueError("VoronoiSelection: points must be three-dimensional")

        if self.__startingPointsIdx is not None and len(self.__startingPointsIdx) < 2:
            raise ValueError("VoronoiSelection: length of startingPointsIdx is less than 2 - you've run out of points!")

        if self.__dim is not None and len(self.__dim) < 3:
            raise ValueError("VoronoiSelection: box dimensions must be three-dimensional")

    # getters and setters
    def setStartingPointsIdx(self, startingPointsIdx):
        self.__startingPointsIdx = startingPointsIdx
        if self.__alreadySelected is None:
            self.__alreadySelected = set(tuple(self.__startingPointsIdx))

    def resetAlreadySelected(self):
        self.__alreadySelected = set(tuple(self.__startingPointsIdx))

    def getAlreadySelected(self):
        return self.__alreadySelected

    def setPoints(self, points):
        self.__points = points

    def getPoints(self):
        return self.__points

    def setDimensions(self, dim):
        self.__dim = dim

    def getDimensions(self):
        return self.__dim

    def setAtomInfo(self, atomInfo):
        self.__atomInfo = atomInfo

    def getAtomInfo(self):
        return self.__atomInfo

    def getRememberInner(self):
        return self.__rememberInner

    def setRememberInner(self, rememberInner):
        self.__rememberInner = rememberInner

    def getNeighborList(self):
        return self.__neighborList

    def setNeighborList(self, neighborList):
        """
        there are times when you need to make many starting objects
        and recalculating the tessellation every time is inefficient
        so just set the neighbor list after the first getNeighbors() run
        """
        self.__neighborList = neighborList

    def getIdx(self):
        return self.__startingPointsIdx

    def __fillNeighborList(self, neighborListRaw):
        """
        turns voro++'s neighbor list into a dictionary (subsequently makes things faster)

        neighborListRaw: ndarray
            neighbor list outputted by freud/voro++
        """
        self.__neighborList = {}
        for i in range(len(neighborListRaw[:, 0])):
            neighbor1 = neighborListRaw[i, 0]
            try:
                self.__neighborList[neighbor1].append(neighborListRaw[i, 1])
            except KeyError:
                self.__neighborList[neighbor1] = [neighborListRaw[i, 1]]

        # for key, value in self.__neighborList.items():
        #     self.__neighborList[key] = list(np.sort(self.__neighborList[key]))
        # exit()
        # print(neighborList)

    def getNeighbors(self, *, neighborIdx, byResidue=True):
        """
        neighborIdx: list or ndarray
            the atom indices among which to search for neighbors (can be all indices or fewer)
        byResidue: bool
            whether to select the whole residue if at least one atom is a neighbor
        """

        neighborIdx = set(list(neighborIdx))

        if self.__points is None or self.__dim is None or self.__startingPointsIdx is None:
            raise ValueError("VoronoiSelection: points, starting points, and dimensions must be set before obtaining neighbors")

        if self.__neighborList is None:
            print("VoronoiSelection: neighborList is empty: calculating Voronoi tessellation...")
            box = Box(self.__dim[0], self.__dim[1], self.__dim[2])
            neighborListRawRaw = Voronoi().compute((box, self.__points))  # 0.47 seconds for LK7B system
            neighborListRaw = neighborListRawRaw.nlist[:]

            print("filling neighbor list...")
            self.__fillNeighborList(neighborListRaw)  # 0.15 seconds for LK7B system

        currNeighbors = set(self.__startingPointsIdx)
        currNeighbors = self.__getNextNeighbors(currNeighbors, neighborIdx, byResidue)
        currNeighbors = np.array(list(currNeighbors))

        return VoronoiSelection(startingPointsIdx=currNeighbors.copy(), points=self.__points, dim=self.__dim,
                                rememberInner=self.__rememberInner, neighborList=self.__neighborList,
                                alreadySelected=self.__alreadySelected,
                                atomInfo=self.__atomInfo)

    def __getNextNeighbors(self, currNeighbors, neighborIdx, byResidue):
        """use fast neighbor analysis to identify Voronoi neighbors

        currNeighbors: set
            the current set of neighbors
        neighborIdx: set
            the region among which to search for neighbors
        byResidue: bool
            select by residue (only if atomInfo is provided)
        Returns
        -------
        set of new neighbors
        """
        newNeighbors = []

        if byResidue is True:
            if self.__atomInfo is None:
                raise ValueError("byResidue selection requires atomInfo!")

            idx = np.arange(0, len(self.__atomInfo))  # all atom indices (not counting virtual points)
            for key in self.__neighborList:
                if key in currNeighbors:
                    for value in self.__neighborList[key]:
                        if value not in self.__alreadySelected:
                            if value in neighborIdx:
                                currResid = self.__atomInfo[value].resid
                                currSegid = self.__atomInfo[value].segid
                                resAtomsRaw = idx[np.logical_and(self.__atomInfo.resids == currResid, self.__atomInfo.segids == currSegid)]
                                resAtomsClean = []
                                for atom in resAtomsRaw:
                                    if idx[atom] in neighborIdx:
                                        resAtomsClean.append(atom)
                                newNeighbors += list(resAtomsClean)
                                if self.__rememberInner is True:
                                    for atom in resAtomsClean:
                                        self.__alreadySelected.add(atom)
        else:
            for key in self.__neighborList:
                if key in currNeighbors:
                    for value in self.__neighborList[key]:
                        if value not in self.__alreadySelected:
                            if value in neighborIdx:
                                newNeighbors.append(value)
                                if self.__rememberInner is True:
                                    self.__alreadySelected.add(value)

        return newNeighbors

    def toAtoms(self, system):
        """
        system: MDAnalysis.Universe

        Returns
        -------
            MDAnalysis.atomGroup corresponding to the starting region
        """
        return system.atoms[self.getIdx()]

    def __str__(self):
        if self.__points is None or self.__startingPointsIdx is None:
            return ">>> empty VoronoiSelection object <<< missing data, please fill"
        else:
            return f"""
>>> VoronoiSelection object <<<
    points:
        {type(self.__points)}
    startingPointsIdx:
        {type(self.__startingPointsIdx)}
    neighborList:
        {type(self.__neighborList)}
    rememberInner: {self.__rememberInner}
"""

    def __len__(self):
        return len(self.__startingPointsIdx)

    def __add__(self, other):
        if self.__rememberInner != other.__rememberInner:
            raise ValueError("VoronoiSelections add: must have same rememberInner value!")
        if self.__startingPointsIdx is None or other.__startingPointsIdx is None:
            raise ValueError("VoronoiSelections add: can't add empty VoronoiSelections!")
        if self.__points is None or other.__points is None:
            raise ValueError("VoronoiSelections add: can't add VoronoiSelections without points!")

        return VoronoiSelection(
            startingPointsIdx=np.array(list(set(list(self.__startingPointsIdx) + list(other.__startingPointsIdx)))),
            points=self.__points,
            dim=self.__dim,
            rememberInner=self.__rememberInner,
            neighborList=self.__neighborList,
            alreadySelected=list(set(list(self.__alreadySelected) + list(other.__alreadySelected))),
            atomInfo=self.__atomInfo + other.__atomInfo)
