#####
About
#####

TranQL-Jupyter introduces the ``%tranql_query`` magic for querying the TranQL interpreter within Jupyter.
Additionally, it adds utilties for working with and visualizing knowledge graphs using NetworkX, Seaborn, and Plotly.

############
Installation
############

TranQL-Jupyter is not on PyPI, and must be installed manually:

::

  git clone https://github.com/frostyfan109/TranQL-Jupyter.git
  pip install ./TranQL-Jupyter

or

::

  git clone https://github.com/frostyfan109/TranQL-Jupyter.git
  cd TranQL-Jupyter
  python setup.py install

TranQL-Jupyter also requires the `TranQL interpreter`_, which is not on PyPI either. It takes a sizeable
amount of time to manually install (~5-10 minutes), so you can either manually install in the same manner as TranQL-Jupyter:

::

  git clone https://github.com/frostyfan109/tranql.git
  pip install ./TranQL

or

::

  git clone https://github.com/frostyfan109/tranql.git
  cd TranQL
  python setup.py install

or, you can export an environment variable prior to running the notebook server which points to
your own installation of TranQL:

Linux: ::

  export tranql_path="~/random_dir/tranql/"
  jupyter notebook

Windows: ::

  set tranql_path=C:\\random_dir\\tranql
  jupyter notebook

.. _TranQL interpreter: https://github.com/NCATS-Tangerine/tranql

#####
Usage
#####

TranQL-Jupyter can be loaded using:

::

  %load_ext tranql_jupyter

Line Magics
-----------

To run a query, use the ``%tranql_query`` line magic:

::

  In [1]: knowledge_graph = %tranql_query SELECT chemical_substance->disease from "/schema" where chemical_substance="CHEMBL:CHEMBL1261"

Interpreter options can be configured with %config:

::

  In [2]: %config TranQLMagic.asynchronous=False


Knowledge Graphs
----------------

A ``KnowledgeGraph`` object is returned by ``%tranql_query`` which supports a variety of operations such as ``union``, ``difference``, and ``symmetric_difference``.
For the entire list, see `NetworkX Operators`_. The ``+`` and ``-`` operators are overloaded to perform the ``simple_union`` and ``difference`` operations.

Pandas
""""""

``KnowledgeGraph`` has two built-in methods for usage with Pandas:

- ``to_dataframe_dict`` converts a ``KnowledgeGraph`` object to a dict of ``{"nodes": DataFrame, "edges": DataFrame}``
- ``from_dataframe_dict`` converts the former's structure back to a ``KnowledgeGraph``

Visualization
"""""""""""""

There are various ways to go about creating visualizations of a ``KnowledgeGraph``. The simplest is to use any of NetworkX's built in `drawing methods`_,
for example:

::

  In [3]: import networkx as nx
          nx.draw_networkx(knowledge_graph.net)

It is important to note that a ``KnowledgeGraph`` is a just a wrapper of a NetworkX graph, and the underlying NetworkX instance has to be accessed
using the ``net`` field.

.. _NetworkX Operators: https://networkx.github.io/documentation/stable/reference/algorithms/operators.html
.. _drawing methods: https://networkx.github.io/documentation/networkx-1.10/reference/drawing.html#id2

However, ``KnowledgeGraph`` also offers more comprehensive built-in visualization methods. A 3D force-directed graph can be rendered like so:

::

  In [4]: knowledge_graph.render_force_graph_3d()

Its 2D counterpart can be rendered as well using the ``render_force_graph_2d`` method.

Additionally, a 2D variant can be rendered using Plotly, although it takes much longer to render:

::

  In [5]: knowledge_graph.render_plotly_force_graph()

The tabular view of a knowledge graph can be rendered using the ``render_node_table`` and ``render_edge_table`` methods.

::

  In [6]: knowledge_graph.render_node_table()
          knowledge_graph.render_edge_table()

By default, only a few columns are shown in the table view. The ``columns`` keyword argument can be used to specify which
columns to include. If columns is ``None``, all columns will be displayed.

Statistical Visualizations
""""""""""""""""""""""""""

Various graphs can be rendered to compare data from multiple knowledge graphs. These are static methods
of ``KnowledgeGraph``, and in order to access them, ``KnowledgeGraph`` has to be explicitly imported:

::

  In [7]: from tranql_jupyter.tranql_response import KnowledgeGraph

Using ``plot_graph_sizes``, the number of nodes and edges in each knowledge graph can quickly be determined:

::

  In [8]: named_knowledge_graph.graph_name = "Named Knowledge Graph"
          KnowledgeGraph.plot_graph_sizes(
            (knowledge_graph, "Knowledge Graph #1"),
            (other_knowledge_graph, "Knowledge Graph #2"),
            named_knowledge_graph
          )

For each ``KnowledgeGraph``, a tuple must be passed of ``(KnowledgeGraph, name)``. If the ``graph_name`` field is set on a ``KnowledgeGraph`` object,
just the knowledge graph may be specified.

The ``plot_node_type_distributions`` and ``plot_edge_type_distributions`` methods allows for easy comparison of the quantities of each type of node/edge in the graphs:

::

  In [9]: KnowledgeGraph.plot_node_type_distributions(
            (knowledge_graph, "Knowledge Graph #1"),
            (other_knowledge_graph, "Knowledge Graph #2"),
            named_knowledge_graph
          )

Demos
-----
For a comprehensive demo, see the ``test_notebooks/test_mock`` notebook.
