# About

TranQL-Jupyter introduces the `%tranql_query` magic for querying the
TranQL interpreter within Jupyter. It also adds utilties for
working with and visualizing knowledge graphs using NetworkX, Seaborn,
and Plotly through its `KnowledgeGraph` class.

# Installation

TranQL-Jupyter is not on PyPI, and must be installed manually:

    git clone https://github.com/frostyfan109/TranQL-Jupyter.git
    pip install ./TranQL-Jupyter

or

    git clone https://github.com/frostyfan109/TranQL-Jupyter.git
    cd TranQL-Jupyter
    python setup.py install

TranQL-Jupyter also requires the [TranQL interpreter](https://github.com/NCATS-Tangerine/tranql), which is not on
PyPI either. It takes a sizeable amount of time to manually install
(\~5-10 minutes), so you can either manually install in the same manner
as TranQL-Jupyter:

    git clone https://github.com/frostyfan109/tranql.git
    pip install ./TranQL

or

    git clone https://github.com/frostyfan109/tranql.git
    cd TranQL
    python setup.py install

or, you can export an environment variable prior to running the notebook
server which points to your own installation of TranQL:

Linux:

    export tranql_path="~/random_dir/tranql/"
    jupyter notebook

Windows:

    set tranql_path=C:\\random_dir\\tranql
    jupyter notebook

# Usage

TranQL-Jupyter can be loaded using:

    %load_ext tranql_jupyter

## Line Magics

To run a query, use the `%tranql_query` line magic:

    In [1]: knowledge_graph = %tranql_query SELECT chemical_substance->disease from "/schema" where chemical_substance="CHEMBL:CHEMBL1261"

Interpreter options can be configured with `%config`:

    In [2]: %config TranQLMagic.asynchronous=False

For all configurable options, see [the TranQLMagic class](https://github.com/frostyfan109/TranQL-Jupyter/blob/master/tranql_jupyter/tranql_magics.py).

## Knowledge Graphs

A `KnowledgeGraph` object is returned by `%tranql_query` which supports
a variety of operations such as `union`, `difference`, and
`symmetric_difference`. For the entire list, see [NetworkX Operators](https://networkx.github.io/documentation/stable/reference/algorithms/operators.html).
The `+` and `-` operators are overloaded to perform the `simple_union` and `simple_difference` operations.

    In[1]: simple_union = mock1 + mock2
           simple_difference = mock1 - mock2
           simple_symmetric_difference = mock1.simple_symmetric_difference(mock2)

Note: operators prefixed with `simple` are rewrites of NetworkX operators which do **not** require identical node sets.

### Pandas

The `KnowledgeGraph` class has two built-in methods for usage with Pandas:

  - `to_dataframe_dict` converts a `KnowledgeGraph` object to a dict of
    `{"nodes": DataFrame, "edges": DataFrame}`
  - `from_dataframe_dict` converts the former's structure back to a
    `KnowledgeGraph`

### Visualization

There are various ways to go about creating visualizations of a
`KnowledgeGraph`. A 3D force-directed graph can be rendered like
so:

    In [4]: knowledge_graph.render_force_graph_3d()

Its 2D counterpart can be rendered as well:

    In [4]: knowledge_graph.render_force_graph_2d()

Additionally, a 2D variant can be rendered using Plotly, although it
takes much longer to render:

    In [5]: knowledge_graph.render_plotly_force_graph()

The tabular view of a knowledge graph can be rendered using the
`render_node_table` and `render_edge_table` methods:

    In [6]: knowledge_graph.render_node_table()
            knowledge_graph.render_edge_table()

By default, only a few columns are shown in the table view. The
`columns` keyword argument can be used to specify which columns to
include. If columns is `None`, all columns will be displayed.

### Statistical Visualizations

Various graphs can be rendered to compare data from multiple knowledge
graphs. These are static methods of `KnowledgeGraph`, and in order to
access them, `KnowledgeGraph` has to be explicitly imported:

    In [7]: from tranql_jupyter.tranql_response import KnowledgeGraph

Using `plot_graph_sizes`, the number of nodes and edges in each
knowledge graph can quickly be determined:

    In [8]: named_knowledge_graph.graph_name = "Named Knowledge Graph"
            KnowledgeGraph.plot_graph_sizes(
              (knowledge_graph, "Knowledge Graph #1"),
              (other_knowledge_graph, "Knowledge Graph #2"),
              named_knowledge_graph
            )

For each `KnowledgeGraph`, a tuple must be passed of `(KnowledgeGraph,
name)`. If the `graph_name` field is set on a `KnowledgeGraph` object,
just the knowledge graph may be specified.

The `plot_node_type_distributions` and `plot_edge_type_distributions`
methods allows for easy comparison of the quantities of each type of
node/edge in the graphs:

    In [9]: KnowledgeGraph.plot_node_type_distributions(
              (knowledge_graph, "Knowledge Graph #1"),
              (other_knowledge_graph, "Knowledge Graph #2"),
              named_knowledge_graph
            )

## API Reference

### Importing
```
%load_ext tranql_jupyter
```

### Querying
```
%tranql_query query_body
```
| Name | Description | Default |
| --- | --- | :--: |
| query_body: str | Standard TranQL query | |

### Operations

#### NetworkX Rewrites (simple)
| Method | Description | Default |
| --- | --- | :--: |
| simple_difference(other_kg: KnowledgeGraph) | Returns a KnowledgeGraph containing all nodes/edges in self that are not in other_kg | other_kg: *required* |
| simple_intersection(other_kg: KnowledgeGraph) | Returns a KnowledgeGraph containing only nodes/edges that exist in both self and other_kg | other_kg: *required* |
| simple_symmetric_difference(other_kg: KnowledgeGraph) | Returns a KnowledgeGraph containing only nodes/edges that exist in self or other_kg but not both | other_kg: *required* |
| \_\_sub__(other_kg: KnowledgeGraph) | Overloads the `-` operator. Alias of `simple_difference` | other_kg: *required* |

#### NetworkX
| Method | Description | Default |
| --- | --- | :--: |
| compose(other_kg: KnowledgeGraph) | Returns a KnowledgeGraph of the simple union of self and other_kg (node sets do not have to be disjoint) | other_kg: *required* |
| simple_union(other_kg: KnowledgeGraph) | Alias of `compose` | other_kg: *required* |
| \_\_add__(other_kg: KnowledgeGraph) | Overloads the `+` operator. Alias of `compose` | other_kg: *required* |
| union(other_kg: KnowledgeGraph) | Returns a KnowledgeGraph of the union of self and other_kg  (node sets **must** be disjoint) | other_kg: *required* |
| disjoint_union(other_kg: KnowledgeGraph) | Returns a KnowledgeGraph of the disjoint union of self and other_kg| other_kg: *required* |
| intersection(other_kg: KnowledgeGraph) | Returns a KnowledgeGraph containing only edges that exist in both self and other_kg | other_kg: *required* |
| difference(other_kg: KnowledgeGraph) | Returns a KnowledgeGraph containing edges that exist in self but not in other_kg | other_kg: *required* |
| symmetric_difference(other_kg: KnowledgeGraph) | Returns a KnowledgeGraph containing edges that exist in self or other_kg but not both | other_kg: *required* |
| cartesian_product(other_kg: KnowledgeGraph) | Returns a KnowledgeGraph of the Cartesian product of self and other_kg | other_kg: *required* |

### Pandas
| Method | Description | Default | Returns |
| --- | --- | :--: | :--: |
| to_dataframe_dict() | Converts the `KnowledgeGraph` to a dict of DataFrames |  | `{"nodes": DataFrame, "edges": DataFrame}` |
| static from_dataframe_dict(df_dict: dict) | Converts the return type of `to_dataframe_dict` back into a `KnowledgeGraph` | df_dict: *required* | `KnowledgeGraph` |

### Visualization
| Method | Description | Default |
| --- | --- | :--: |
| render_force_graph_2d(title: str) | Renders a 2D force graph of the knowledge graph | title: *automatic* |
| render_force_graph_3d(title: str) | Renders a 3D force graph of the knowledge graph | title: *automatic* |
| render_plotly_force_graph(title: str) | Renders a force graph of the knowledge graph using Plotly | title: `Knowledge Graph` |
| render_node_table(columns: list\|tuple) | Renders a tabular view of the nodes in the knowledge graph | columns: `["name", "id", "type", "equivalent_identifiers"]` |
| render_edge_table(columns: list\|tuple) | Renders a tabular view of the edges in the knowledge graph | columns: `["source_id", "target_id", "type", "weight"]` |

### Statistical Visualization
| Method | Description | Arguments |
| --- | --- | --- |
| static plot_graph_sizes(*graphs: tuple\|KnowledgeGraph) | Plots a grouped bar chart of the amount of nodes and edges in each graph | If the graph does not have its `graph_name` set, then it should be a tuple of `(graph: KnowledgeGraph, name: str)` |
| static plot_node_type_distributions(*graphs: tuple\|KnowledgeGraph) | Plots a grouped bar chart of the amount of each node type in the graphs | See above |
| static plot_edge_type_distributions(*graphs: tuple\|KnowledgeGraph) | Plots a grouped bar chart of the amount of each edge type in the graphs | See above |

### Misc
| Method | Description | Returns |
| --- | --- | :--: |
| build_knowledge_graph() | Builds a knowledge graph in dictionary form | `{"nodes": dict[], "edges": dict[]}` |

| Field | Description | Default |
| --- | --- | :--: |
| graph_name: str | When set, methods that require a title or name will use this if nothing is provided | `None` |

## Demos

For a comprehensive demo, see the `test_notebooks/test_mock` notebook.
