# VoronoiSelection

see demo_voronoiSelection.py for information on how to select the first and second
hydration shells and how to partition the air-water interface into layers, as done
in the paper


if you need to make many starting atom group objects, recalculating the Voronoi tessellation every time is wasteful. Avoid this by grabbing the neighborList after the first getNeighbors() run and setting it in future VoronoiSelection objects using setNeighborList() as in the below example

```python
# say you need a bigNumber of hydration shells for different parts of the system
# you can do the following for efficiency:

# get the first of the hydration shells
startIdx = 0
endIdx = startIdx + nAtomsPerGroup
startingGroup = system.atoms[startIdx:endIdx]

vor = VoronoiSelection(startingPointsIdx=startingGroup.ix,
                       points=system.atoms.positions,
                       atomInfo=system.atoms,
                       dim=system.dimensions,
                       rememberInner=True)  # remember neighbors already found (see demo)
hydrationShell[0] = vor.getNeighbors(neighborIdx=waters.ix, byResidue=True)  # tessellation calculated here
neighborList = vor.getNeighborList()  # grab it for later

# never calculate Voronoi tessellation again, just set it
for i in range(1, bigNumber):
    startIdx = i * nAtomsPerGroup
    endIdx = startIdx + nAtomsPerGroup
    startingGroup = system.atoms[startIdx:endIdx]

    vor = VoronoiSelection(startingPointsIdx=startingGroup.ix,
                       points=system.atoms.positions,
                       atomInfo=system.atoms,
                       dim=system.dimensions,
                       rememberInner=True)
    vor.setNeighborList(neighborList)  # <-----------
    hydrationShell[i] = vor.getNeighbors(neighborIdx=waters.ix, byResidue=True)
```
