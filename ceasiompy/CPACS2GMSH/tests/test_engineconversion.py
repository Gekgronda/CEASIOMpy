"""
CEASIOMpy: Conceptual Aircraft Design Software

Developed by CFS ENGINEERING, 1015 Lausanne, Switzerland

Test functions for 'ceasiompy/CPACS2GMSH/engineconversion.py'

Python version: >=3.7

| Author : Tony Govoni
| Creation: 2022-05-17

"""

# ==============================================================================
#   IMPORTS
# ==============================================================================


import os
import shutil
import sys
from pathlib import Path

import gmsh
import pytest
from ceasiompy.CPACS2GMSH.func.engineconversion import close_engine
from ceasiompy.utils.commonpaths import CPACS_FILES_PATH
from cpacspy.cpacspy import CPACS

MODULE_DIR = Path(__file__).parent
TEST_OUT_PATH = Path(MODULE_DIR, "ToolOutput")
TEST_IN_PATH = Path(MODULE_DIR, "ToolInput")


# ==============================================================================
#   CLASSES
# ==============================================================================


# ==============================================================================
#   FUNCTIONS
# ==============================================================================


@pytest.mark.skipif(
    sys.platform == "darwin", reason="'synchronize' function causes segmentation fault on macOS"
)
def test_close_engine():
    """Test the close_engine function with a simple engine"""

    # create a a new file where the test will be made
    if not (Path(TEST_IN_PATH, "test_close_engine")).exists():
        os.mkdir(Path(TEST_IN_PATH, "test_close_engine"))
    TEST_DIR_PATH = Path(TEST_IN_PATH, "test_close_engine")

    # copy nacelle brep files and cfg file in the test dir
    shutil.copy(Path(TEST_IN_PATH, "SimpleNacelle_fanCowl.brep"), TEST_DIR_PATH)
    shutil.copy(Path(TEST_IN_PATH, "SimpleNacelle_centerCowl.brep"), TEST_DIR_PATH)
    shutil.copy(Path(TEST_IN_PATH, "config_engines.cfg"), TEST_DIR_PATH)

    # test function
    engine_uids = ["SimpleEngine", "SimpleNacelle_centerCowl", "SimpleNacelle_fanCowl"]
    engine_files_path = [
        Path(TEST_DIR_PATH, "SimpleNacelle_fanCowl.brep"),
        Path(TEST_DIR_PATH, "SimpleNacelle_centerCowl.brep"),
    ]
    engines_cfg_file_path = Path(TEST_DIR_PATH, "config_engines.cfg")

    nacelle_parts = {
        "fanCowl": engine_files_path[0],
        "centerCowl": engine_files_path[1],
    }
    # Test the function
    closed_engine_path = close_engine(
        nacelle_parts,
        engine_uids,
        TEST_DIR_PATH,
        engines_cfg_file_path,
        engine_surface_percent=(0.2, 0.2),
    )

    # Check the output file was generated

    assert closed_engine_path == Path(TEST_DIR_PATH, "SimpleEngine.brep")

    # Check the output file with gmsh

    gmsh.initialize()

    # Import the closed engine
    gmsh.model.occ.importShapes(str(closed_engine_path), highestDimOnly=False)
    gmsh.model.occ.synchronize()

    # Check the engine is closed and only one volume is present
    assert len(gmsh.model.occ.getEntities(dim=3)) == 1

    # Check if the engine is meshable
    gmsh.model.mesh.generate(3)

    # Clear gmsh api session
    gmsh.clear()
    gmsh.finalize()

    # Delete the closed engine file
    if TEST_DIR_PATH.exists():
        shutil.rmtree(TEST_DIR_PATH)


# ==============================================================================
#    MAIN
# ==============================================================================

if __name__ == "__main__":

    print("Test CPACS2GMSH")
    print("To run test use the following command:")
    print(">> pytest -v")