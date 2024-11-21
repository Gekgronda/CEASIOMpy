"""
CEASIOMpy: Conceptual Aircraft Design Software

Developed by CFS ENGINEERING, 1015 Lausanne, Switzerland

This module can be called to generate a surrogate model based on specified
inputs and outputs. A CSV file describing the entries and containing the data
must be provided to train the model, except if the values are taken from an
aeromap, in which case they can all be found in the CPACS file.

Python version: >=3.8

| Author: Vivien Riolo
| Creation: 2020-07-06

TODO:
    * Enable model-specific settings for user through the GUI

"""

import datetime
import pickle
import re
from pathlib import Path

# =================================================================================================
#   IMPORTS
# =================================================================================================
from re import split as splt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import smt.surrogate_models as sms

from ceasiompy.SMUse.smuse import Surrogate_model
from ceasiompy.utils.ceasiomlogger import get_logger
from ceasiompy.utils.ceasiompyutils import get_results_directory
from ceasiompy.utils.moduleinterfaces import (
    get_toolinput_file_path,
    get_tooloutput_file_path,
)
from ceasiompy.utils.commonxpath import OPTWKDIR_XPATH, SMFILE_XPATH, SMTRAIN_XPATH
from cpacspy.cpacsfunctions import create_branch, get_value_or_default
from cpacspy.cpacspy import CPACS
from cpacspy.utils import COEFS, PARAMS


# def main(cpacs_path, cpacs_out_path):
