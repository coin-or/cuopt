===============================
Third-Party Modeling Languages
===============================


--------------------------
AMPL Support
--------------------------

AMPL can be used with near zero code changes: simply switch to cuOpt as a solver to solve linear and mixed-integer programming problems. Please refer to the `AMPL documentation <https://www.ampl.com/>`_ for more information. Also, see the example notebook in the `colab <https://colab.research.google.com/drive/1eEQik_pae4g_tJQ61QJFlO1fFBXazpBr?usp=sharing>`_.

--------------------------
GAMS and GAMSPy Support
--------------------------

GAMS and GAMSPy models can be used with near zero code changes after setting up the solver link: simply switch to cuOpt as a solver to solve linear and mixed-integer programming problems (e.g. ``gams trnsport lp=cuopt``). Please refer to the `GAMS cuOpt link repository <https://github.com/GAMS-dev/cuoptlink-builder>`_ for more information on how to setup GAMS and GAMSPy for cuOpt. Also, see the example notebook in the `cuopt-examples <https://github.com/NVIDIA/cuopt-examples>`_ repository.

--------------------------
PuLP Support
--------------------------

PuLP can be used with near zero code changes: simply switch to cuOpt as a solver to solve linear and mixed-integer programming problems.
Please refer to the `PuLP documentation <https://pypi.org/project/PuLP/>`_ for more information. Also, see the example notebook in the `cuopt-examples <https://github.com/NVIDIA/cuopt-examples>`_ repository.

--------------------------
JuMP Support
--------------------------

JuMP can be used with near zero code changes: simply switch to cuOpt as a solver to solve linear and mixed-integer programming problems.
Please refer to the `JuMP documentation <https://github.com/jump-dev/cuOpt.jl>`_ for more information.

cuOpt.jl loads ``libcuopt_mathopt.so`` through the system dynamic loader, so the directory containing it must be on ``LD_LIBRARY_PATH``. This is not the case by default for a pip install, which places the library inside the Python environment:

.. code-block:: bash

   pip install --extra-index-url=https://pypi.nvidia.com libcuopt-cu13==26.10.*
   export LD_LIBRARY_PATH="$(python -c 'import libcuopt_mathopt, os; print(os.path.join(os.path.dirname(libcuopt_mathopt.__file__), "lib64"))'):$LD_LIBRARY_PATH"

``libcuopt.so`` is only a linker script and cannot be loaded directly, so pointing the loader at the ``libcuopt`` wheel directory is not enough.

--------------------------
Pyomo Support
--------------------------

Pyomo models can be used with near zero code changes via cuOpt's direct solver interface: simply select cuOpt as the solver to solve linear and mixed-integer programming problems. Please refer to the `Pyomo documentation <https://www.pyomo.org/>`_ for more information.

--------------------------
CVXPY Support
--------------------------

CVXPY can be used with near zero code changes: simply select cuOpt as the solver to solve linear and quadratic programs, as well as QCQP/SOCP problems (beta). Please refer to the `CVXPY documentation <https://www.cvxpy.org/>`_ for more information.
