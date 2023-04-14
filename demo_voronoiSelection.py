import MDAnalysis as mda
from voronoiSelection import VoronoiSelection, prepareVirtualPointsAboveSurface


# ################################################################################
# demo: how to select the first and second hydration shells around a protein
# ################################################################################

# import structure of LK7B solvated by TIP4P-Ew water
system = mda.Universe("LK7B_1nm_tip4p.pdb")

# since this is an interfacial system, need to add virtual points above surface
allPoints, firstLayerIdx, secondLayerIdx = prepareVirtualPointsAboveSurface(system)

# select relevant AtomGroups
waters = system.select_atoms("resname HOH")
protein = system.select_atoms("not resname HOH NA CL Na Cl Na+ Cl-")

# define starting point for VoronoiSelection (protein atoms)
# keyword arguments only allowed for the VoronoiSelection class
vor = VoronoiSelection(startingPointsIdx=protein.ix,
                       points=allPoints,
                       atomInfo=system.atoms,
                       dim=system.dimensions,
                       rememberInner=True)  # remember neighbors already found

# calculate first hydration shell
firstHydrationShell = vor.getNeighbors(neighborIdx=waters.ix, byResidue=True)

# calculate second hydration shell
secondHydrationShell = firstHydrationShell.getNeighbors(neighborIdx=waters.ix, byResidue=True)

# convert VoronoiSelections to MDAnalysis AtomGroups and write to PDB file
protein.write("protein.pdb")
firstHydrationShell.toAtoms(system).write("first_hydration_shell.pdb")
secondHydrationShell.toAtoms(system).write("second_hydration_shell.pdb")


# ###############################################################################
# demo: how to partition the air-water interface into Voronoi layers
# ###############################################################################

# import structure of air-water interface (TIP4P-Ew)
system = mda.Universe("allWater.pdb")

# since this is an interfacial system, need to add virtual points above surface
allPoints, firstLayerIdx, secondLayerIdx = prepareVirtualPointsAboveSurface(system)

# define starting point as the first layer of virtual points (right above surface)
layer = VoronoiSelection(startingPointsIdx=firstLayerIdx,
                         points=allPoints,
                         atomInfo=system.atoms,
                         dim=system.dimensions,
                         rememberInner=True)

# define layers recursively and write to PDB file
for i in range(4):  # so that input value of 0 gives surface
    layer = layer.getNeighbors(neighborIdx=system.atoms.ix, byResidue=True)
    layer.toAtoms(system).write(f"layer_{i + 1}.pdb")
