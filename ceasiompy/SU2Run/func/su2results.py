"""
CEASIOMpy: Conceptual Aircraft Design Software

Developed by CFS ENGINEERING, 1015 Lausanne, Switzerland

Extract results from SU2 calculations and save them in a CPACS file.

Python version: >=3.8

| Author: Aidan Jungo
| Creation: 2019-10-02

TODO:

    * Saving for Control surface deflections

"""

# =================================================================================================
#   IMPORTS
# =================================================================================================

from pathlib import Path

from ceasiompy.SU2Run.func.extractloads import extract_loads
from ceasiompy.SU2Run.func.su2utils import (
    get_efficiency_and_aoa,
    get_su2_aerocoefs,
    get_wetted_area,
)
from ceasiompy.utils.ceasiomlogger import get_logger
from ceasiompy.utils.commonnames import SU2_FORCES_BREAKDOWN_NAME
from ceasiompy.utils.commonxpath import (
    GMSH_SYMMETRY_XPATH,
    RANGE_LD_RATIO_XPATH,
    SU2_AEROMAP_UID_XPATH,
    SU2_EXTRACT_LOAD_XPATH,
    SU2_FIXED_CL_XPATH,
    SU2_ROTATION_RATE_XPATH,
    SU2_UPDATE_WETTED_AREA_XPATH,
    WETTED_AREA_XPATH,
)
from cpacspy.cpacsfunctions import create_branch, get_value, get_value_or_default
from cpacspy.cpacspy import CPACS
from cpacspy.utils import COEFS

log = get_logger()


# =================================================================================================
#   CLASSES
# =================================================================================================


# =================================================================================================
#   FUNCTIONS
# =================================================================================================


def get_su2_results(cpacs_path, cpacs_out_path, wkdir):
    """Function to write SU2 results in a CPACS file.

    Function 'get_su2_results' gets available results from the latest SU2
    calculation and put them at the correct place in the CPACS file.

    '/cpacs/vehicles/aircraft/model/analyses/aeroPerformance/aeroMap[n]/aeroPerformanceMap'

    Args:
        cpacs_path (Path): Path to input CPACS file
        cpacs_out_path (Path): Path to output CPACS file
        wkdir (Path): Path to the working directory

    """

    cpacs = CPACS(cpacs_path)

    if not wkdir.exists():
        raise OSError(f"The working directory : {wkdir} does not exit!")

    fixed_cl = get_value_or_default(cpacs.tixi, SU2_FIXED_CL_XPATH, "NO")

    if fixed_cl == "YES":
        aeromap_uid = "aeroMap_fixedCL_SU2"
    elif fixed_cl == "NO":
        aeromap_uid = get_value(cpacs.tixi, SU2_AEROMAP_UID_XPATH)
    else:
        raise ValueError("The value for fixed_cl is not valid! Should be YES or NO")

    log.info(f"The aeromap uid is: {aeromap_uid}")
    aeromap = cpacs.get_aeromap_by_uid(aeromap_uid)

    alt_list = aeromap.get("altitude").tolist()
    mach_list = aeromap.get("machNumber").tolist()
    aoa_list = aeromap.get("angleOfAttack").tolist()
    aos_list = aeromap.get("angleOfSideslip").tolist()

    case_dir_list = [case_dir for case_dir in wkdir.iterdir() if "Case" in case_dir.name]

    found_wetted_area = False

    for config_dir in sorted(case_dir_list):

        if not config_dir.is_dir():
            continue

        force_file_path = Path(config_dir, SU2_FORCES_BREAKDOWN_NAME)
        if not force_file_path.exists():
            raise OSError("No result force file have been found!")

        baseline_coef = True

        case_nb = int(config_dir.name.split("_")[0].split("Case")[1])

        aoa = aoa_list[case_nb]
        aos = aos_list[case_nb]
        mach = mach_list[case_nb]
        alt = alt_list[case_nb]

        if fixed_cl == "YES":
            cl_cd, aoa = get_efficiency_and_aoa(force_file_path)

            # Replace aoa with the with the value from fixed cl calculation
            aeromap.df.loc[0, ["angleOfAttack"]] = aoa

            # Save cl/cd found during the fixed CL calculation (useful for range analysis)
            create_branch(cpacs.tixi, RANGE_LD_RATIO_XPATH)
            cpacs.tixi.updateDoubleElement(RANGE_LD_RATIO_XPATH, cl_cd, "%g")

        cl, cd, cs, cmd, cms, cml, velocity = get_su2_aerocoefs(force_file_path)

        # Damping derivatives
        rotation_rate = get_value_or_default(cpacs.tixi, SU2_ROTATION_RATE_XPATH, -1.0)
        ref_len = cpacs.aircraft.ref_length
        adim_rot_rate = rotation_rate * ref_len / velocity

        coefs = {"cl": cl, "cd": cd, "cs": cs, "cmd": cmd, "cms": cms, "cml": cml}

        for axis in ["dp", "dq", "dr"]:

            if f"_{axis}" not in config_dir.name:
                continue

            baseline_coef = False

            for coef in COEFS:
                coef_baseline = aeromap.get(coef, alt=alt, mach=mach, aoa=aoa, aos=aos)
                dcoef = (coefs[coef] - coef_baseline) / adim_rot_rate
                aeromap.add_damping_derivatives(
                    alt=alt,
                    mach=mach,
                    aos=aos,
                    aoa=aoa,
                    coef=coef,
                    axis=axis,
                    value=dcoef,
                    rate=rotation_rate,
                )

        if "_TED_" in config_dir.name:

            # TODO: convert when it is possible to save TED in cpacspy
            raise NotImplementedError("TED not implemented yet")

            # baseline_coef = False
            # config_dir_split = config_dir.split('_')
            # ted_idx = config_dir_split.index('TED')
            # ted_uid = config_dir_split[ted_idx+1]
            # defl_angle = float(config_dir.split('_defl')[1])
            # try:
            #     print(Coef.IncrMap.dcl)
            # except AttributeError:
            #     Coef.IncrMap = a.p.m.f.IncrementMap(ted_uid)
            # dcl = (cl-Coef.cl[-1])
            # dcd = (cd-Coef.cd[-1])
            # dcs = (cs-Coef.cs[-1])
            # dcml = (cml-Coef.cml[-1])
            # dcmd = (cmd-Coef.cmd[-1])
            # dcms = (cms-Coef.cms[-1])
            # control_parameter = -1
            # Coef.IncrMap.add_cs_coef(dcl,dcd,dcs,dcml,dcmd,dcms,ted_uid,control_parameter)

        # Baseline coefficients (no damping derivative or control surfaces case)
        if baseline_coef:
            aeromap.add_coefficients(
                alt=alt,
                mach=mach,
                aos=aos,
                aoa=aoa,
                cd=cd,
                cl=cl,
                cs=cs,
                cml=cml,
                cmd=cmd,
                cms=cms,
            )

        update_wetted_area = get_value_or_default(cpacs.tixi, SU2_UPDATE_WETTED_AREA_XPATH, False)
        if not found_wetted_area and update_wetted_area:
            wetted_area = get_wetted_area(Path(config_dir, "logfile_SU2_CFD.log"))

            # Check if symmetry plane is defined (Default: False)
            sym_factor = 1.0
            if get_value_or_default(cpacs.tixi, GMSH_SYMMETRY_XPATH, False):
                log.info("Symmetry plane is defined. The wetted area will be multiplied by 2.")
                sym_factor = 2.0

            create_branch(cpacs.tixi, WETTED_AREA_XPATH)
            cpacs.tixi.updateDoubleElement(WETTED_AREA_XPATH, wetted_area * sym_factor, "%g")
            found_wetted_area = True

        if get_value_or_default(cpacs.tixi, SU2_EXTRACT_LOAD_XPATH, False):
            extract_loads(config_dir)

    aeromap.save()
    cpacs.save_cpacs(cpacs_out_path, overwrite=True)


# =================================================================================================
#    MAIN
# =================================================================================================

if __name__ == "__main__":

    log.info("Nothing to execute!")
