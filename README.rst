==================
Material Mechanics
==================


.. image:: https://img.shields.io/pypi/v/material_mechanics.svg
        :target: https://pypi.python.org/pypi/material_mechanics

.. image:: https://img.shields.io/travis/kemeen/material_mechanics.svg
        :target: https://travis-ci.org/kemeen/material_mechanics

.. image:: https://readthedocs.org/projects/material-mechanics/badge/?version=latest
        :target: https://material-mechanics.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status




The "Material Mechanics" package contains tools needed in the analysis of the mechanics of materials,
including fiber reinforced materials and laminates.


* Free software: MIT license
* Documentation: https://material-mechanics.readthedocs.io.

Installation
------------
To install simply use pip

>>> pip install material_mechanics

Features
--------

Materials:
    - Isotropic materials
    - Transverse isotropic materials
    - Orthotropic materials
    - fiber reinforced plastics (FRP)
    - Laminates

Analytics:
    - Stiffness analysis
    - fracture mechanics of FRP
        - Puck 2D and 3D
    - Classical Lamination Theory (CLT)

Roadmap
-------

Materials
    - Non linear material laws

Analytics
    - Fracture mechanics for isotropic materials (von Mises Stress)
    - Addition of damage criteria for FRP
        - strain criteria for whole FRP laminates
        - Tsai-Wu criterion
    - integration of fatigue damage analysis

Usage
-----


Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
