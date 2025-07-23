.. gdar documentation master file

Welcome to the GDAR Model Documentation!
========================================

The **Graph Diffusion Autoregressive (GDAR)** library provides tools fitting the GDAR model to neural data and
estimating the GDAR flow signal The model integrates a structural connectivity graph with an autoregressive model
to estimate directed, time-resolved information flow in networks. The library also provides functions for visulaizing
the flow signal and decomposing it into various flow modes.

This documentation includes installation instructions, usage examples, API reference, and background on the GDAR model.

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Table of Contents

   getting_started
   usage
   modules


About GDAR
----------
GDAR combines classical vector autoregression (VAR) with graph diffusion processes, enabling:

- Integration of spatial priors through the graph Laplacians
- Estimation of directed communication flow signals with high temporal resolution
- Modeling of complex spatiotemporal neural dynamics

For more details, see the full paper:

Schwock, F., Bloch, J., Khateeb, K., Zhou, J., Atlas, L., & Yazdan-Shahmorad, A.
*Inferring Neural Communication Dynamics from Field Potentials Using Graph Diffusion Autoregression*.


Indices and Tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


