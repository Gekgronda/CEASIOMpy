import os
import numpy as np
import pandas as pd
from tixi3.tixi3wrapper import Tixi3
from ceasiompy.utils.commonxpath import (
    AVL_XPATH,
    AVL_AEROMAP_UID_XPATH,
    REF_XPATH,
    GMSH_XPATH,
    GMSH_SYMMETRY_XPATH,
    GMSH_FARFIELD_FACTOR_XPATH,
    GMSH_MESH_SIZE_ENGINES_XPATH,
    GMSH_MESH_SIZE_FACTOR_FUSELAGE_XPATH,
    GMSH_MESH_SIZE_FACTOR_WINGS_XPATH,
    GMSH_MESH_SIZE_FARFIELD_XPATH,
    SU2_XPATH,
    SU2_AEROMAP_UID_XPATH,
)


def create_or_update_element(tixi, xpath, value):
    """
    Create an element in the CPACS file if it doesn't exist, otherwise update it.

    Parameters:
        tixi (Tixi3): Tixi object to manipulate the CPACS file.
        xpath (str): XPath of the element.
        value (str | int | float | list | tuple): Value to assign to the element.
    """
    parent_xpath = "/".join(xpath.split("/")[:-1])
    element_name = xpath.split("/")[-1]

    # Convert lists or tuples to semicolon-separated strings
    if isinstance(value, (list, tuple)):
        value = "; ".join(map(str, value))
    if isinstance(value, np.ndarray):  # if 'value' is a numpy array (es 4 aeromap)
        value = "; ".join(map(str, value.tolist()))  # convert to list then to str
    else:
        value = str(value)

    # Check and create the parent node if it does not exist
    if not tixi.checkElement(parent_xpath):
        grandparent_xpath = "/".join(parent_xpath.split("/")[:-1])
        parent_name = parent_xpath.split("/")[-1]
        tixi.createElement(grandparent_xpath, parent_name)

    # Create or update the element
    if not tixi.checkElement(xpath):
        tixi.createElement(parent_xpath, element_name)
    tixi.updateTextElement(xpath, value)


def change_reference_value(tixi, ref_val):
    for key, values in ref_val.items():
        ref_path = f"{REF_XPATH}/{key}"
        create_or_update_element(tixi, f"{ref_path}", values)


def add_new_aeromap(tixi, aeromap, aeromap_uid, aeromap_name):
    """
    Create a new aeromap in the CPACS file under aeroPerformance.

    Args:
        tixi (Tixi3): Tixi object to manipulate the CPACS file.
        aeromap (dict): Aeromap data containing keys: altitude, machNumber, angleOfAttack, angleOfSideslip.
        aeromap_uid (str): Unique ID for the new aeroMap.
        aeromap_name (str): Name of the new aeroMap.

    Returns:
        None
    """
    AEROPERFORMANCE_PATH = "/cpacs/vehicles/aircraft/model/analyses/aeroPerformance"
    NEW_AEROMAP_PATH = f"{AEROPERFORMANCE_PATH}/aeroMap[@uID='{aeromap_uid}']"

    if not isinstance(aeromap, dict):
        raise TypeError(f"'aeromap' must be a dictionary, got {type(aeromap)} instead.")

    # Ensure aeroPerformance exists
    if not tixi.checkElement(AEROPERFORMANCE_PATH):
        tixi.createElement("/cpacs/vehicles/aircraft/model/analyses", "aeroPerformance")

    # Ensure the aeroMap exists
    if not tixi.checkElement(NEW_AEROMAP_PATH):
        tixi.createElement(AEROPERFORMANCE_PATH, "aeroMap")
        tixi.addTextAttribute(f"{AEROPERFORMANCE_PATH}/aeroMap[last()]", "uID", aeromap_uid)

    # Add general information
    create_or_update_element(tixi, f"{NEW_AEROMAP_PATH}/name", aeromap_name)
    create_or_update_element(
        tixi, f"{NEW_AEROMAP_PATH}/description", f"Aeromap for {aeromap_name}"
    )

    # Add boundary conditions
    create_or_update_element(
        tixi, f"{NEW_AEROMAP_PATH}/boundaryConditions/atmosphericModel", "ISA"
    )

    # Add aeroPerformanceMap elements
    performance_map_path = f"{NEW_AEROMAP_PATH}/aeroPerformanceMap"
    for key, values in aeromap.items():
        element_path = f"{performance_map_path}/{key}"
        if not tixi.checkElement(element_path):
            tixi.createElement(performance_map_path, key)
        # Add the mapType attribute
        tixi.addTextAttribute(element_path, "mapType", "vector")
        create_or_update_element(tixi, f"{element_path}", values)


def avl_update(tixi, aeromap_name, avl_params):
    """
    Update CPACS with parameters needed
    """
    # Create avl_xpath
    if not tixi.checkElement(AVL_XPATH):
        tixi.createElement("/cpacs/toolspecific/CEASIOMpy", "aerodynamics")
        tixi.createElement("/cpacs/toolspecific/CEASIOMpy/aerodynamics", "avl")

    # Add aeromap name
    create_or_update_element(tixi, f"{AVL_AEROMAP_UID_XPATH}", aeromap_name)

    #
    for key, values in avl_params.items():
        params_path = f"{AVL_XPATH}/{key}"
        create_or_update_element(tixi, f"{params_path}", values)


def euler_update(tixi, aeromap_name, common_params, mesh_params, su2_params):
    """
    Aggiorna o crea i valori nel file CPACS per parametri aerodinamici, della mesh e SU2.

    Parameters:
        tixi (Tixi3): Oggetto Tixi per manipolare il file CPACS.
        common_params(list[dict]):
        mesh_params (list[dict]): Lista di dizionari con path e value per i parametri principali della mesh.
        su2_params (list[dict]): Lista di dizionari con path e value per i parametri SU2.
    """
    if not tixi.checkElement(GMSH_XPATH):
        tixi.createElement("/cpacs/toolspecific/CEASIOMpy", "mesh")
        tixi.createElement("/cpacs/toolspecific/CEASIOMpy/mesh", "gmshOptions")

    for key, values in common_params.items():
        common_path = f"{GMSH_XPATH}/{key}"
        create_or_update_element(tixi, f"{common_path}", values)

    for key, values in mesh_params.items():
        mesh_path = f"{GMSH_XPATH}/{key}"
        create_or_update_element(tixi, f"{mesh_path}", values)
        # Add aeromap name

    create_or_update_element(tixi, f"{SU2_AEROMAP_UID_XPATH}", aeromap_name)

    if not tixi.checkElement(SU2_XPATH):
        tixi.createElement("/cpacs/toolspecific/CEASIOMpy", "aerodynamics")
        tixi.createElement("/cpacs/toolspecific/CEASIOMpy/aerodynamics", "su2")

    for key, values in su2_params.items():
        su2_path = f"{SU2_XPATH}/{key}"
        create_or_update_element(tixi, f"{su2_path}", values)
