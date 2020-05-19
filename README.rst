==========================
GP System Identification
==========================

.. image:: https://img.shields.io/github/license/upb-lea/openmodelica-microgrid-gym
     :target: LICENSE

Research project with the aim to build `Gaussian process`_ models to efficiently predict the states of a dynamic system.
The dynamic system is expected to comply to the `OpenAi Gym`_ interface.
The Gaussian process processes are implemented with the use of GPyTorch_.
All regressors implement the `scikit-learn`_ interfaces.

The system identification can be used in `model-based reinforcement learning`_ algorithms.

.. _`OpenAI Gym`: https://gym.openai.com/
.. _`Gaussian Process`: https://www.youtube.com/watch?v=92-98SYOdlY
.. _`GPyTorch`: https://github.com/cornellius-gp/gpytorch
.. _`scikit-learn`: https://scikit-learn.org/
.. _`model-based reinforcement learning`: https://www.youtube.com/watch?v=ItMutbeOHtc
.. _`openmodelica-microgrid-gym`: https://github.com/upb-lea/openmodelica-microgrid-gym


Installation
------------

Since the goal of my research is to predict the states of an `openmodelica-microgrid-gym`_ fmu this is also an
installation requirement.

* Clone the repository::

  $ git clone https://github.com/stheid/gp-systemident.git

* Create a :code:`conda` environment for Python 3.8 and enter a conda environment (i recommend to create this in PyCharm if you are using this IDE)
* Install PyFMI through :code:`conda`::

  $ conda install -c conda-forge pyfmi

* Install the complete package::

  $ pip install .[examples]


Only core deps
``````````````

If you want to use this code for another gym you can leave out the optional dependencies::

$ git clone https://github.com/stheid/gp-systemident.git
$ pip install .

Usage
-----

* `simple GP model`_

.. _`simple GP model`: examples/simpleGP.py