"""
CEASIOMpy: Conceptual Aircraft Design Software

Developed by CFS ENGINEERING, 1015 Lausanne, Switzerland

Use .brep files parts of an airplane to generate a fused airplane in GMSH with
the OCC kernel. Then Spherical farfield is created around the airplane and the
resulting domain is meshed using gmsh

Python version: >=3.8

| Author: Guido Vallifuoco
| Creation: 2024-02-01

TODO:

    - It may be good to move all the function and some of the code in generategmsh()
    that are related to disk actuator to another python script and import it here

    - Add mesh sizing for each aircraft part and as consequence add marker

    - Integrate other parts during fragmentation

    - Use run software function instead subprocess.call

"""

# =================================================================================================
#   IMPORTS
# =================================================================================================

import os
import subprocess
from pathlib import Path

import gmsh
from ceasiompy.CPACS2GMSH.func.generategmesh import (
    # duplicate_disk_actuator_surfaces,
    # control_disk_actuator_normal,
    # get_entities_from_volume,
    ModelPart,
    add_disk_actuator,
    fuselage_size,
    process_gmsh_log,
)
from ceasiompy.utils.ceasiomlogger import get_logger

# from ceasiompy.utils.commonnames import (
#     ACTUATOR_DISK_OUTLET_SUFFIX,
#     ENGINE_EXHAUST_SUFFIX,
#     ENGINE_INTAKE_SUFFIX,
#     GMSH_ENGINE_CONFIG_NAME,
# )
from ceasiompy.utils.ceasiompyutils import get_part_type

# from ceasiompy.utils.commonxpath import GMSH_MESH_SIZE_WINGS_XPATH
from ceasiompy.utils.configfiles import ConfigFile

log = get_logger()


# =================================================================================================
#   FUNCTIONS
# =================================================================================================


def generate_2d_mesh_for_pentagrow(
    cpacs, cpacs_path, brep_dir, results_dir, open_gmsh, min_max_mesh_factor, symmetry=False
):
    """
    Function to generate a mesh from brep files forming an airplane
    Function 'generate_gmsh' is a subfunction of CPACS2GMSH which return a
    mesh file useful for pentagrow.
    The airplane is fused with the different brep files : fuselage, wings and
    other parts are identified and fused together in order to obtain a watertight volume.
    Args:
    ----------
    cpacs : CPACS
        CPACS object
    brep_dir : Path
        Path to the directory containing the brep files
    results_dir : Path
        Path to the directory containing the result (mesh) files
    open_gmsh : bool
        Open gmsh GUI after the mesh generation if set to true
    symmetry : bool
        If set to true, the mesh will be generated with symmetry wrt the x,z plane
    mesh_size_fuselage : float
        Size of the fuselage mesh
    mesh_size_wings : float
        Size of the wing mesh
    mesh_size_engines : float
        Size of the engine mesh
    mesh_size_propellers : float
        Size of the propeller mesh
    advance_mesh : bool
        If set to true, the mesh will be generated with advanced meshing options
    refine_factor : float
        refine factor for the mesh le and te edge
    refine_truncated : bool
        If set to true, the refinement can change to match the truncated te thickness
    auto_refine : bool
        If set to true, the mesh will be checked for quality
    testing_gmsh : bool
        If set to true, the gmsh sessions will not be clear and killed at the end of
        the function, this allow to test the gmsh feature after the call of generate_gmsh()
    ...
    Returns:
    ----------
    mesh_file : Path
        Path to the mesh file generated by gmsh


    """
    # Determine if rotor are present in the aircraft model
    rotor_model = False
    if Path(brep_dir, "config_rotors.cfg").exists():
        rotor_model = True

    if rotor_model:
        log.info("Adding disk actuator")
        config_file = ConfigFile(Path(brep_dir, "config_rotors.cfg"))
        add_disk_actuator(brep_dir, config_file)

    # Retrieve all brep
    brep_files = list(brep_dir.glob("*.brep"))
    brep_files.sort()

    # initialize gmsh
    gmsh.initialize()
    # Stop gmsh output log in the terminal
    gmsh.option.setNumber("General.Terminal", 0)
    # Log complexity
    gmsh.option.setNumber("General.Verbosity", 5)

    # Import each aircraft original parts / parent parts
    fuselage_volume_dimtags = []
    wings_volume_dimtags = []
    enginePylons_enginePylon_volume_dimtags = []
    engine_nacelle_fanCowl_volume_dimtags = []
    engine_nacelle_coreCowl_volume_dimtags = []
    vehicles_engines_engine_volume_dimtags = []
    vehicles_rotorcraft_model_rotors_rotor_volume_dimtags = []

    log.info(f"Importing files from {brep_dir}")

    for brep_file in brep_files:
        # Import the part and create the aircraft part object
        part_entities = gmsh.model.occ.importShapes(str(brep_file), highestDimOnly=False)
        gmsh.model.occ.synchronize()

        # Create the aircraft part object
        part_obj = ModelPart(uid=brep_file.stem)
        # maybe to cut off -->
        part_obj.part_type = get_part_type(cpacs.tixi, part_obj.uid)

        if part_obj.part_type == "fuselage":
            fuselage_volume_dimtags.append(part_entities[0])
            model_bb = gmsh.model.get_bounding_box(
                fuselage_volume_dimtags[0][0], fuselage_volume_dimtags[0][1]
            )

        elif part_obj.part_type == "wing":
            wings_volume_dimtags.append(part_entities[0])
            # return wings_volume_dimtags

        elif part_obj.part_type == "enginePylons/enginePylon":
            enginePylons_enginePylon_volume_dimtags.append(part_entities[0])
            # return enginePylons_enginePylon_volume_dimtags

        elif part_obj.part_type == "engine/nacelle/fanCowl":
            engine_nacelle_fanCowl_volume_dimtags.append(part_entities[0])

        elif part_obj.part_type == "engine/nacelle/coreCowl":
            engine_nacelle_coreCowl_volume_dimtags.append(part_entities[0])

        elif part_obj.part_type == "vehicles/engines/engine":
            vehicles_engines_engine_volume_dimtags.append(part_entities[0])

        elif part_obj.part_type == "vehicles/rotorcraft/model/rotors/rotor":
            vehicles_rotorcraft_model_rotors_rotor_volume_dimtags.append(part_entities[0])
        else:
            log.warning(f"'{brep_file}' cannot be categorized!")
            return None
    gmsh.model.occ.synchronize()
    log.info("Start manipulation to obtain a watertight volume")
    # we have to obtain a wathertight volume
    gmsh.model.occ.cut(wings_volume_dimtags, fuselage_volume_dimtags, -1, True, False)

    gmsh.model.occ.synchronize()

    gmsh.model.occ.fuse(wings_volume_dimtags, fuselage_volume_dimtags, -1, True, True)

    gmsh.model.occ.synchronize()

    model_dimensions = [
        abs(model_bb[0] - model_bb[3]),
        abs(model_bb[1] - model_bb[4]),
        abs(model_bb[2] - model_bb[5]),
    ]

    fuselage_maxlen, _ = fuselage_size(cpacs_path)

    gmsh.model.occ.translate(
        [(3, 1)],
        -((model_bb[0]) + (model_dimensions[0] / 2)),
        -((model_bb[1]) + (model_dimensions[1] / 2)),
        -((model_bb[2]) + (model_dimensions[2] / 2)),
    )

    gmsh.model.occ.synchronize()
    log.info("Manipulation finished")

    aircraft_surface_dimtags = gmsh.model.get_entities(2)
    len_aircraft_surface = len(aircraft_surface_dimtags)
    surface = []

    for i in range(len_aircraft_surface):
        tags = aircraft_surface_dimtags[i][1]
        surface.append(tags)

    gmsh.model.add_physical_group(2, surface, -1, name="aircraft_surface")

    # Mesh generation
    log.info("Start of gmsh 2D surface meshing process")

    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.option.setNumber("Mesh.LcIntegrationPrecision", 1e-6)
    mesh_size = model_dimensions[0] * float(min_max_mesh_factor) * (10**-3)
    gmsh.option.set_number("Mesh.MeshSizeMin", mesh_size)
    gmsh.option.set_number("Mesh.MeshSizeMax", mesh_size)
    gmsh.option.setNumber("Mesh.StlOneSolidPerSurface", 2)

    gmsh.model.occ.synchronize()
    gmsh.logger.start()
    gmsh.model.mesh.generate(1)
    gmsh.model.mesh.generate(2)
    if open_gmsh:
        log.info("Result of 2D surface mesh")
        log.info("GMSH GUI is open, close it to continue...")
        gmsh.fltk.run()

    gmsh.model.occ.synchronize()

    gmesh_path = Path(results_dir, "mesh_2d.stl")
    gmsh.write(str(gmesh_path))

    process_gmsh_log(gmsh.logger.get())

    return gmesh_path, fuselage_maxlen


def pentagrow_3d_mesh(
    result_dir,
    fuselage_maxlen,
    farfield_factor,
    n_layer,
    h_first_layer,
    max_layer_thickness,
    growth_factor,
    growth_ratio,
    feature_angle,
) -> None:
    # create the config file for pentagrow
    config_penta_path = Path(result_dir, "config.cfg")
    # Variables
    InputFormat = "stl"
    NLayers = n_layer
    FeatureAngle = feature_angle
    InitialHeight = h_first_layer * (10**-5)
    MaxGrowthRatio = growth_ratio
    MaxLayerThickness = max_layer_thickness / 10
    FarfieldRadius = fuselage_maxlen * farfield_factor * 100
    FarfieldCenter = "0.0 0.0 0.0"
    OutputFormat = "su2"
    HolePosition = "0.0 0.0 0.0"
    TetgenOptions = "-pq1.3VY"
    TetGrowthFactor = growth_factor
    HeightIterations = 8
    NormalIterations = 8
    MaxCritIterations = 128
    LaplaceIterations = 8

    # writing to file
    file = open(config_penta_path, "w")
    file.write(f"InputFormat = {InputFormat}\n")
    file.write(f"NLayers = {NLayers}\n")
    file.write(f"FeatureAngle = {FeatureAngle}\n")
    file.write(f"InitialHeight = {InitialHeight}\n")
    file.write(f"MaxGrowthRatio = {MaxGrowthRatio}\n")
    file.write(f"MaxLayerThickness = {MaxLayerThickness}\n")
    file.write(f"FarfieldRadius = {FarfieldRadius}\n")
    file.write(f"OutputFormat = {OutputFormat}\n")
    file.write(f"HolePosition = {HolePosition}\n")
    file.write(f"FarfieldCenter = {FarfieldCenter}\n")
    file.write(f"TetgenOptions = {TetgenOptions}\n")
    file.write(f"TetGrowthFactor = {TetGrowthFactor}\n")
    file.write(f"HeightIterations = {HeightIterations}\n")
    file.write(f"NormalIterations = {NormalIterations}\n")
    file.write(f"MaxCritIterations = {MaxCritIterations}\n")
    file.write(f"LaplaceIterations = {LaplaceIterations}\n")

    os.chdir("Results/GMSH")

    if os.path.exists("mesh_2d.stl"):
        log.info("mesh_2d.stl exists")
    else:
        log.warning("mesh_2d.stl does not exist")

    if os.path.exists("config.cfg"):
        log.info("config.cfg exists")
    else:
        log.warning("config.cfg does not exist")

    current_dir = os.getcwd()
    os.chdir(current_dir)

    # command = "pentagrow mesh_2d.stl config.cfg"
    command = ["pentagrow", "mesh_2d.stl", "config.cfg"]
    # Specify the file path
    file_path = "command.txt"

    command_str = " ".join(command)

    with open(file_path, "w") as file:
        file.write(command_str)

    subprocess.call(command, cwd=current_dir, start_new_session=False)

    mesh_path = Path(result_dir, "hybrid.su2")
    log.info(f"Mesh path:{mesh_path}")

    return mesh_path
