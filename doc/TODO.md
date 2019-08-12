# TODO

## CEASIOMpy project

* Add a recognizable logo
* Make installation easier (Tixi, Tigl, TIGLCreator, sumo, SU2, pyTornado, RCE, conda?)
    * Connecting CEASIOMpy with RCE?
* File structure
    * Separate code from user working directory

## Module dependencies

* Add `__specs__.py` files for each modules
* Add output for `ceasiompy.CLCalculator.__specs__`
* Add General File in/out Class
* Maybe create a function to check consistency (does same variable name are stored a the same xpath, ...)
* Add kind of a GUI to enter missing input requirements...?

## Documentation

* Host the documentation on ReadTheDocs (https://readthedocs.org/)?
* Automatic syncronisation of CPACS input/output
* Fix docstring translation/formatting
    * https://www.python.org/dev/peps/pep-0257/
    * http://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html
* Make a page for users
        * How to install
        * Possible workflow, module order/compatibility
* Make a page for contributors
    * Guidelines, etc.
    * Example for how to structure and add a new module

## Aerodynamic modules
* How to store inputs data in CPACS (alpha,beta, mach, alt)
* How to store results in CPACS (AeroPerformanceMap, damping derivatives, ...)
* Folder structure for the results

## Modules

* To adapt and integrate
    * Template Module
    * PyTornado
    * GUI Edit AeroPerformanceMap
    * SUMO2CPACS
    * CoefCPACS2CSV
    * PlotTool

* To develop
    * StaticStablility
    * DynamicStability
    * FlightModel
    * EngineTool
    * LaTeXReport
    * TestSU2Convergence

## RCE Integration
    * Upuload .json file
    * Add a script to automatically set the ToolDirectory path of each module
    * Small logo for each module