About
-----

TranQL-Jupyter introduces the ``%tranql_query`` magic for querying the TranQL interpreter within Jupyter.
Additionally, it adds utilties for working with and visualizing knowledge graphs using NetworkX, Seaborn, and Plotly.

Installation
------------

TranQL-Jupyter is not on PyPI, and must be installed manually:

::

  git clone https://github.com/frostyfan109/TranQL-Jupyter.git
  pip install ./TranQL-Jupyter

or

::

  git clone https://github.com/frostyfan109/TranQL-Jupyter.git
  cd TranQL-Jupyter
  python setup.py install

Currently, TranQL_ is not a pip-installable package. Therefore, it must be installed and directly
interfaced with by this package in order to function properly. Before running the notebook server,
the TranQL path needs to be specified as an environment variable:

Linux: ::

  export tranql_path="~/random_dir/tranql/"
  jupyter notebook

Windows: ::

  set tranql_path=C:\\random_dir\\tranql
  jupyter notebook

.. _TranQL: https://github.com/NCATS-Tangerine/tranql

Usage
-----

TranQL-Jupyter can be loaded using:
``%load_ext tranql_jupyter``

To run a query, use the ``%tranql_query`` line magic:

::

  In [1]: knowledge_graph = %tranql_query SELECT chemical_substance->disease from "/schema" where chemical_substance="CHEMBL:CHEMBL1261"

Interpreter options can be configured as follows:

::

  In [2]: %config TranQLMagic.asynchronous=False

And all options can be listed using ``%config TranQLMagic``

A ``KnowledgeGraph`` object is returned by ``%tranql_query`` which supports a variety of operations such as ``union``, ``difference``, and ``symmetric_difference``.
For the entire list, see `NetworkX Operators`_. There are various ways to go about creating visualizations of a ``KnowledgeGraph``. The simplest is to use
NetworkX's built in drawing methods:

::

  In [3]: import networkx as nx
          nx.draw_networkx(knowledge_graph.net)

It is important to note that a ``KnowledgeGraph`` is a just a wrapper of a NetworkX graph, and the underlying NetworkX instance has to be accessed
with the ``net`` field.

Additionally, a 3D force-directed graph can be rendered like so:

::

  In [4]: knowledge_graph.render_force_graph()


.. _NetworkX Operators: https://networkx.github.io/documentation/stable/reference/algorithms/operators.html
