import json
import os
import ipywidgets
import networkx as nx
import netcomp as nc
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objs as go
from plotly.offline import iplot
from IPython.utils import capture
from IPython.display import display, HTML
from . import force_graph
from . import table_view
from . import graph_utils
from tranql.tranql_schema import NetworkxGraph

try:
    import importlib.resources as pkg_resources
except ImportError:
    import importlib_resources as pkg_resources

class TranQLResponse:
    """ Initialize a TranQL response. Currently implemented for extensibility.

    :ivar knowledge_graph: The knowledge graph component of the response
    :vartype knowledge_graph: :class:`.KnowledgeGraph`
    :ivar knowledge_map: The knowledge map component of the response
    :vartype knowledge_map: dict
    :ivar question_graph: The question graph component of the response
    :vartype question_graph: dict
    """
    def __init__(self, message):
        """
        :param message: A dictionary with keys "knowledge_map", "knowledge_graph", and "question_graph"
        :type message: dict
        """
        self.knowledge_graph = KnowledgeGraph(message["knowledge_graph"])
        self.knowledge_map = message["knowledge_map"]
        self.question_graph = message["question_graph"]

class KnowledgeGraph(NetworkxGraph):
    """ Initialize a TranQL knowledge graph. Includes various utilities for working with and visualizing knowledge graphs.

    :ivar graph_name: A reference to the name of the graph. Not required, just helpful for some methods.
    :vartype graph_name: str, None
    :ivar net: Internal NetworkX instance of the graph. Should be used for any NetworkX-related operations. Modifications will result in mutation of the graph.
    :vartype net: :class:`networkx.MultiDiGraph`
    :ivar node_click_history: List containing the nodes clicked in force graph renderings. Can get the most recently clicked node `node_click_history[-1]` with
        :py:attr:`~.selected_graph_node`
    :vartype node_click_history: list
    """
    def __init__(self, knowledge_graph, sources=None):
        """
        :param knowledge_graph: Either a dict containing the keys "nodes" and "edges" or a networkx.MultiDiGraph instance
            with node/edge attributes structured in the same manner as done in :py:meth:`~.build_networkx_graph`
        :type knowledge_graph: dict, :class:`networkx.MultiDiGraph`
        """
        super().__init__()

        # You may set the graph_name of a KnowledgeGraph for statistical visualizations
        self.graph_name = None

        if isinstance(knowledge_graph, dict):
            self.build_networkx_graph(knowledge_graph)
        else:
            # Constructed with existing NetworkX graph
            self.net = knowledge_graph

        # Garbage parameter that was being experimented with and never got removed (does nothing)
        self.sources = sources

        self.node_click_history = []


    @classmethod
    def mock1(cls):
        """ Returns a mock :class:`.KnowledgeGraph` containing the results of the query
        select chemical_substance->gene->disease from "/graph/gamma/quick" where disease="asthma"
        """
        return cls.mock("mock1.json")

    @classmethod
    def mock2(cls):
        """ Returns a mock :class:`.KnowledgeGraph` containing the results of a query
        """
        return cls.mock("mock2.json")

    @staticmethod
    def mock(name):
        mock_response = json.loads(pkg_resources.read_text(__package__, name))
        return KnowledgeGraph(mock_response["knowledge_graph"])


    # Pandas integration
    def to_dataframe_dict(self):
        """ Turns the knowledge graph into a dictionary containing two dataframes "nodes" and "edges"

        :return: A dict of "nodes" and "edges" dataframes
        :rtype: dict
        """
        kg = self.build_knowledge_graph()

        return {
            "nodes": pd.DataFrame(kg["nodes"]),
            "edges": pd.DataFrame(kg["edges"])
        }

    @staticmethod
    def from_dataframe_dict(df_dict):
        """ Takes a dict containing keys "nodes" and "edges" and values of :class:`pandas.DataFrame`
        and converts it into a knowledge graph instance

        :param df_dict: A dictionary containing keys "nodes" and "edges"
        :type df_dict: dict

        :return: An instance of a :class:`.KnowledgeGraph`
        :rtype: :class:`.KnowledgeGraph`
        """
        # A knowledge graph cannot be represented by a data frame, but nodes and edges can be their own separate data frames
        nodes = df_dict["nodes"].to_dict("records")
        edges = df_dict["edges"].to_dict("records")

        return KnowledgeGraph({
            "nodes": nodes,
            "edges": edges
        })



    # Statistical visualizations
    @staticmethod
    def plot_graph_sizes(*graphs):
        """ Makes a Plotly graph displaying how many nodes and edges are in each graph

        :param *graphs: Varargs of (title, :class:`.KnowledgeGraph`) or just :class:`.KnowledgeGraph` if :py:attr:`~graph_name` is set
        :type *graphs: list
        """
        names = []
        node_values = []
        edge_values = []
        for graph in graphs:
            if isinstance(graph, KnowledgeGraph):
                knowledge_graph = graph
                name = knowledge_graph.graph_name
            else:
                knowledge_graph, name = graph
            kg = knowledge_graph.build_knowledge_graph()
            names.append(name)
            node_values.append(len(kg["nodes"]))
            edge_values.append(len(kg["edges"]))

        fig = go.Figure(data=[
            go.Bar(
                name="Nodes",
                x=names,
                y=node_values
            ),
            go.Bar(
                name="Edges",
                x=names,
                y=edge_values
            )
        ])
        fig.update_layout(title_text="Graph Sizes", barmode="group")
        fig.show()

    @classmethod
    def plot_node_type_distributions(cls, *graphs):
        """ Makes a Plotly graph detailing how many nodes of each type there are in the graph

        :param *graphs: Varargs of (title, :class:`.KnowledgeGraph`) or just :class:`.KnowledgeGraph` if :py:attr:`~graph_name` is set
        :type *graphs: list
        """
        return cls.plot_type_distributions("nodes", graphs)

    @classmethod
    def plot_edge_type_distributions(cls, *graphs):
        """ Makes a Plotly graph detailing how many edges of each type there are in the graph

        :param *graphs: Varargs of (title, :class:`.KnowledgeGraph`) or just :class:`.KnowledgeGraph` if :py:attr:`~graph_name` is set
        :type *graphs: list
        """
        return cls.plot_type_distributions("edges", graphs)

    @staticmethod
    def plot_type_distributions(type, graphs):
        def format_data(graph):
            if isinstance(graph, KnowledgeGraph):
                knowledge_graph = graph
                name = knowledge_graph.graph_name
            else:
                knowledge_graph, name = graph
            return [
                pd.DataFrame(knowledge_graph.build_knowledge_graph()[type]),
                name
            ]
        data = [format_data(graph) for graph in graphs]
        for graph_data in data:
            graph_data[0] = graph_utils.count_series_list(
                graph_data[0].type
            )
        fig = go.Figure(data=[
            go.Bar(
                name=graph_data[1],
                x=list(graph_data[0].keys()),
                y=list(graph_data[0].values())
            ) for graph_data in data
        ])
        fig.update_layout(title_text="Type Distributions", barmode="group")
        fig.show()

        # plt.legend([i for i, j in data], [j for i, j in data])
        # for graph_data in data:
        #     graph_data[0] = graph_utils.count_series_list(
        #         graph_data[0].type
        #     )
        # bars = graph_utils.plot_dicts([graph_data[0] for graph_data in data])
        # plt.legend(bars, [j for i,j in data])
        # plt.title("Type Distributions")
        # plt.show()


    # ipywidgets.Output doesn't respect script tags, which makes this not work
    '''
    @staticmethod
    def render_graph_grid(kg_array, render_method=None):
        """
        3 ways to pass in a value for rendering:
            1) kg_array=[[my_kg, ...]], render_method=KnowledgeGraph.x_method
                - KnowledgeGraph.x_method(my_kg) will be called
            2) kg_array=[[(my_kg, 5, 3), ...]], render_method=KnowledgeGraph.x_method
                - KnowledgeGraph.x_method(my_kg, 5, 3) will be called
            3) kg_array=[[my_kg.render_x, ...]], render_method=None
                - my_kg.render_x() will be called

        Example:
            1) KnowledgeGraph.render_graph_grid([
                [kg1, kg2],
                [kg1 + kg2]
               ], KnowledgeGraph.render_force_graph_2d)

            2) KnowledgeGraph.render_graph_grid([
                [kg1, (kg2, 5)],
                [     kg3     ]
               ], KnowledgeGraph.render_force_graph_2d)

            3) KnowledgeGraph.render_graph_grid([
                [kg1.render_force_graph_2d, kg2.render_force_graph_2d],
                [(kg1 + kg2).render_force_graph_3d]
            ])
        """
        # display captures
        output = []
        for kg_row in kg_array:
            out_row = []
            for graph_args in kg_row:
                output_widget = ipywidgets.Output()
                with output_widget:
                    # Check if function reference
                    if callable(graph_args):
                        # E.g. `lambda: my_graph.render_force_graph_2d(test_arg, other_arg=5)`
                        # or just `my_graph.render_force_graph_2d`
                        graph_args()
                    elif isinstance(graph_args, (list, tuple)):
                        # E.g. graph_args=(x, 5, my_graph) for def graph_func(test_var, num_tries, graph)
                        render_method(*graph_args)
                    else:
                        # E.g. graph_args=graph for def render_graph(graph)
                        render_method(graph_args)
                out_row.append(output_widget)
            output.append(out_row)
        for row in output:
            """
            [
                [
                    [RichOutput],
                    [RichOutput]
                ],
                [
                    [RichOutput]
                ]
            ]
            """
            hbox = ipywidgets.HBox(row)
            # Then display the row
            display(hbox)

    '''

    # Build self.net from a knowledge graph
    def build_networkx_graph(self, knowledge_graph):
        self.delete()
        for node in knowledge_graph["nodes"]:
            # Store the entire node as its properties so that its data it preserved when converted back to a knowledge graph
            self.add_node(
                node["id"],
                properties=node
            )
        for edge in knowledge_graph["edges"]:
            for predicate in edge["type"]:
                self.add_edge(
                    edge["source_id"],
                    predicate,
                    edge["target_id"],
                    properties=edge
                )
    # Build a knowledge_graph dict from self.net
    def build_knowledge_graph(self):
        """ Builds the KnowledgeGraph instance into a dictionary of "nodes" and "edges"

        :return: A dict of "nodes" and "edges"
        :rtype: dict
        """
        kg = {
            "nodes": [],
            "edges": []
        }
        for node in self.net.nodes(data=True):
            kg["nodes"].append(
                node[1]["attr_dict"]
            )
        for edge in self.net.edges(data=True):
            kg["edges"].append(
                edge[2]
            )
        return kg


    # Define rendering methods
    def render_force_graph_3d(self, **kwargs):
        """ Renders a 3D force graph of the knowledge graph instance using three.js

        :param title: Titles the force graph rendering. Defaults to :py:attr:`~graph_name` if set, or an automatically generated title.
        :type title: str
        """
        return force_graph.render3d(self, **kwargs)
    render_force_graph = render_force_graph_3d # alias

    def render_force_graph_2d(self, **kwargs):
        """ Renders a 2D force graph of the knowledge graph instance

        :param title: Titles the force graph rendering. Defaults to :py:attr:`~graph_name` if set, or an automatically generated title.
        :type title: str
        """
        return force_graph.render2d(self, **kwargs)

    def render_plotly_force_graph(self, title="Knowledge Graph"):
        """ Renders a 2D force graph of the knowledge graph instance using Plotly

        :param title: Titles the force graph rendering. Defaults to "Knowledge Graph".
        :type title: str
        """
        G = self.net.copy()
        pos = nx.spring_layout(G)
        for n, p in pos.items():
            G.node[n]["pos"] = p

        edge_trace = go.Scatter(
            x=[],
            y=[],
            line={
                "width": 0.5,
                "color": "#888"
            },
            hoverinfo="none",
            showlegend=False,
            mode="lines"
        )
        for edge in G.edges():
            x0, y0 = G.node[edge[0]]["pos"]
            x1, y1 = G.node[edge[1]]["pos"]
            edge_trace["x"] += (x0, x1, None)
            edge_trace["y"] += (y0, y1, None)

        node_trace = go.Scatter(
            x=[],
            y=[],
            text=[],
            mode="markers",
            hoverinfo="text",
            showlegend=False,
            marker={
                "size": 15,
                "color": []
            }
        )
        # Nodes are colored based on the first type in their `type` attribute - we need to generate a color palette that has enough colors
        colored_types = []
        for node in G.nodes(data=True):
            properties = node[1]["attr_dict"]
            colored_types.append(properties["type"][0])
        # Remove duplicates
        colored_types = list(set(colored_types))
        palette = sns.color_palette("hls", len(colored_types)).as_hex()
        color_dict = {colored_types[i]: palette[i] for i in range(len(colored_types))}

        for node in G.nodes(data=True):
            node, attr_dict = node
            properties = attr_dict["attr_dict"]

            x, y = G.node[node]["pos"]
            node_trace["x"] += (x,) # comma for tuple
            node_trace["y"] += (y,) # comma for tuple
            node_trace["text"] += ((properties.get("name") or node),) # sometimes a node doesn't have name--if so, use its id
            node_trace["marker"]["color"] += (color_dict[properties["type"][0]],)

        fake_legend = []
        for node_type in color_dict:
            fake_legend.append(go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker={
                    "size": 10,
                    "color": color_dict[node_type]
                },
                legendgroup=node_type,
                showlegend=True,
                name=node_type
            ))
        fig = go.Figure(
            data=[edge_trace, node_trace, *fake_legend],
            layout=go.Layout(
                title=title,
                titlefont={
                    "size": 16
                },
                showlegend=True,
                hovermode="closest",
                margin={
                    "b": 20,
                    "l": 5,
                    "r": 5,
                    "t": 40
                },
                xaxis={
                    "showgrid": False,
                    "zeroline": False,
                    "showticklabels": False
                },
                yaxis={
                    "showgrid": False,
                    "zeroline": False,
                    "showticklabels": False
                }
            )
        )
        return iplot(fig)

    def render_node_table(self, columns=("name", "id", "type", "equivalent_identifiers")):
        """ Renders a tabular view of the nodes in the knowledge graph

        :param columns: A list/tuple of node properties that should be displayed in the table
        :type columns: list, tuple
        """
        return table_view.render_nodes(self, columns=columns)
    def render_edge_table(self, columns=("source_id", "target_id", "type", "weight")):
        """ Renders a tabular view of the edges in the knowledge graph

        :param columns: A list/tuple of edge properties that should be displayed in the table
        :type columns: list, tuple
        """
        return table_view.render_edges(self, columns=columns)

    # Utility for getting available node/edge columns for the tabular view
    def get_node_properties(self):
        """ List all node properties inside of the knowledge graph.
        Can help for choosing table viewer columns and for selecting nodes by property.

        :return: List of node properties in the graph
        :rtype: list
        """
        properties = []
        for node in self.net.nodes(data=True):
            for property in node[1]["attr_dict"]:
                if property not in properties:
                    properties.append(property)
        return properties
    def get_edge_properties(self):
        """ List all edge properties inside of the knowledge graph.
        Can help for choosing table viewer columns and for selecting edges by property.

        :return: List of edge properties in the graph
        :rtype: list
        """
        properties = []
        for edge in self.net.edges(data=True):
            for property in edge[2]:
                if property not in properties:
                    properties.append(property)
        return properties

    """ Define helper methods for working with a knowledge graph """
    def get_nodes_by_property(self, property, value):
        """ Selects nodes in the knowledge graph by a property

        :param property: A node property
        :type property: str
        :param value: The value of the property to select nodes by

        :return: A list of nodes
        :rtype: list
        """
        nodes = []
        for node in self.net.nodes(data=True):
            if property in node[1]["attr_dict"] and node[1]["attr_dict"][property] == value:
                nodes.append(node[1]["attr_dict"])
        return nodes
    def get_edges_by_property(self, property, value):
        """ Selects edges in the knowledge graph by a property

        :param property: An edge property
        :type property: str
        :param value: The value of the property to select nodes by

        :return: A list of edges
        :rtype: list
        """
        edges = []
        for edge in self.net.edges(data=True):
            if property in edge[2] and edge[2][property] == value:
                edges.append(edge[2])
        return edges
    def get_node_by_name(self, name):
        """ Selects a node by its name property

        :param name: The name of the node
        :type name: str

        :return: A node or None
        :rtype: dict, None
        """
        nodes = self.get_nodes_by_property("name", name)
        if len(nodes) == 0:
            return None
        else:
            return nodes[0]
    def get_node_by_id(self, id):
        """ Selects a node by its id property

        :param id: The id of the node
        :type name: str

        :return: A node or None
        :rtype: dict, None
        """
        nodes = self.get_nodes_by_property("id", id)
        if len(nodes) == 0:
            return None
        else:
            return nodes[0]
    def get_edge(self, source, target, pred=None):
        """ Selects an edge/edges from source to target

        :param source: The source_id of the edge
        :type source: str
        :param target: The target_id of the edge
        :type target: str
        :param pred: The predicate of the edge.
        :type pred: str

        :return: If no predicate is specified, a list of edges is returned. If a predicate is specified, an edge or None is returned.
        :rtype: list, dict, None
        """
        # If `pred` is None, returns a list of all edges between `source` and `target`
        # Else returns single edge
        if pred == None:
            try:
                return list(self.net[source][target].values())
            except KeyError:
                # No edges between the two
                return []
        else:
            # Return None if no edge exists
            try:
                return self.net[source][target][pred]
            except KeyError:
                return None

    """ Force graph visualizations nodes clicked """
    @property
    def selected_graph_node(self):
        """ Read-only property that returns the most recently clicked node in the force graph visualization.

        :return: A node or None if no nodes have been clicked
        :rtype: dict, None
        """
        return self.node_click_history[-1] if len(self.node_click_history) > 0 else None

    """ Define graph operations (see https://networkx.github.io/documentation/stable/reference/algorithms/operators.html) """
    def simple_union(self, other_kg):
        """ Returns the simple union of the knowledge graph and other_kg

        :param other_kg: The other knowledge graph
        :type other_kg: :class:`.KnowledgeGraph`
        """
        # Returns a KnowledgeGraph of the simple union of self and other_kg (node sets do not have to be disjoint)
        return KnowledgeGraph(nx.compose(self.net, other_kg.net), sources=[self, other_kg])
    compose = simple_union # alias of simple_union
    def union(self, other_kg):
        """ Returns the union of the knowledge graph and other_kg

        :param other_kg: The other knowledge graph
        :type other_kg: :class:`.KnowledgeGraph`
        """
        # Returns a KnowledgeGraph of the union of self and other_kg  (node sets must be disjoint)
        return KnowledgeGraph(nx.union(self.net, other_kg.net))
    def disjoint_union(self, other_kg):
        """ Returns the disjoint union of the knowledge graph and other_kg

        :param other_kg: The other knowledge graph
        :type other_kg: :class:`.KnowledgeGraph`
        """
        # Returns a KnowledgeGraph of the disjoint union of self and other_kg
        return KnowledgeGraph(nx.disjoint_union(self.net, other_kg.net))
    def intersection(self, other_kg):
        """ Returns the intersection of the knowledge graph and other_kg

        :param other_kg: The other knowledge graph
        :type other_kg: :class:`.KnowledgeGraph`
        """
        # Returns a KnowledgeGraph containing only edges that exist in both self and other_kg
        return KnowledgeGraph(nx.intersection(self.net, other_kg.net))
    def difference(self, other_kg):
        """ Returns the difference of the knowledge graph and other_kg

        :param other_kg: The other knowledge graph
        :type other_kg: :class:`.KnowledgeGraph`
        """
        # Returns a KnowledgeGraph containing edges that exist in self but not in other_kg
        return KnowledgeGraph(nx.difference(self.net, other_kg.net))
    def symmetric_difference(self, other_kg):
        """ Returns the symmetric difference of the knowledge graph and other_kg

        :param other_kg: The other knowledge graph
        :type other_kg: :class:`.KnowledgeGraph`
        """
        # Returns a KnowledgeGraph containing edges that exist in self or other_kg but not both
        return KnowledgeGraph(nx.symmetric_difference(self.net, other_kg.net))
    def cartesian_product(self, other_kg):
        """ Returns the cartesian product of the knowledge graph and other_kg

        :param other_kg: The other knowledge graph
        :type other_kg: :class:`.KnowledgeGraph`
        """
        # Returns a KnowledgeGraph of the Cartesian product of self and other_kg
        return KnowledgeGraph(nx.cartesian_product(self.net, other_kg.net))

    def simple_difference(self, other_kg):
        """ Returns the simple difference of the knowledge graph and other_kg

        :param other_kg: The other knowledge graph
        :type other_kg: :class:`.KnowledgeGraph`
        """
        # Returns a KnowledgeGraph containing all nodes/edges in self that are not in other_kg

        # nx.create_empty_copy will return a MultiDiGraph with all nodes in self.net but 0 edges (retaining properties of nodes)
        new = KnowledgeGraph(nx.create_empty_copy(self.net))

        # Go through and propogate edges that only exist in self.net
        # It is important that new.net currently has all the nodes of self.net - otherwise, when we create a new edge,
        # it will automatically add a node if that node doesn't exist yet (with no properties)
        for edge in self.net.edges(keys=True, data=True):
            source, target, key, properties = edge
            # Only add edge to new if other_kg doesn't have the edge
            if not other_kg.net.has_edge(source, target, key):
                new.add_edge(source, key, target, properties)

        # Now we can remove all nodes that exist in other_kg
        new.net.remove_nodes_from(n for n in self.net if n in other_kg.net)

        return new

    def simple_intersection(self, other_kg, edges=True):
        """ Returns the simple intersection of the knowledge graph and other_kg

        :param other_kg: The other knowledge graph
        :type other_kg: :class:`.KnowledgeGraph`
        :param edges: Performs the intersection on the edge sets as well
        :type edges: bool
        """
        # Returns a KnowledgeGraph containing only nodes/edges that exist in both self and other_kg
        # Default behavior is to do the intersection of both nodes and edges, but edge intersection may not be desireable

        # First, establish which has the fewest edges
        if self.net.number_of_edges() <= other_kg.net.number_of_edges():
            least_edges = self
            most_edges = other_kg
        else:
            least_edges = other_kg
            most_edges = self

        # Create a graph containing the union of the node sets
        # This is just to retain node properties, and will be fixed later
        new = KnowledgeGraph(nx.MultiDiGraph())
        new.net.add_nodes_from(
            list(self.net.nodes(data=True)) +
            list(other_kg.net.nodes(data=True))
        )
        if edges:
            for edge in least_edges.net.edges(keys=True, data=True):
                source, target, key, properties = edge
                # Only add edge to new if both graphs have the edge
                if other_kg.net.has_edge(source, target, key):
                    new.add_edge(source, key, target, properties)

        else:
            # Otherwise, add all edges from both graphs - this is probably the right behavior? (as oppposed to adding just self's edges)
            new.add_edges_from(
                list(self.net.edges(data=True)) +
                list(other_kg.net.edges(data=True))
            )

        # Remove all nodes from new that aren't in both self and other_kg
        # Generator statement so convert new.net to tuple so it doesn't change size during iteration
        new.net.remove_nodes_from(n for n in tuple(new.net) if not (n in self.net and n in other_kg.net))

        return new

    def simple_node_intersection(self, other_kg):
        return self.simple_intersection(other_kg, edges=False)

    def simple_symmetric_difference(self, other_kg):
        """ Returns the simple symmetric difference of the knowledge graph and other_kg

        :param other_kg: The other knowledge graph
        :type other_kg: :class:`.KnowledgeGraph`
        """
        # Returns a KnowledgeGraph containing only nodes/edges that exist in self or other_kg but not both

        # Create a graph containing the symmetric difference of both node sets
        # Can't use built in `set->symmetric_difference` algorithm because properties have to be retained
        # and dicts aren't hashable
        new = KnowledgeGraph(nx.MultiDiGraph())

        new.net.add_nodes_from(
            n for n in self.net.nodes(data=True) if n[0] not in other_kg.net
        )
        new.net.add_nodes_from(
            n for n in other_kg.net.nodes(data=True) if n[0] not in self.net
        )

        for edge in self.net.edges(keys=True, data=True):
            source, target, key, properties = edge
            # Only add edge if other doesn't have it (and make sure it doesn't add edges for nodes that don't exist anymore)
            if not other_kg.net.has_edge(source, target, key) and new.has_node(source) and new.has_node(target):
                new.add_edge(source, key, target, properties)

        for edge in other_kg.net.edges(keys=True, data=True):
            source, target, key, properties = edge
            # Only add edge if self doesn't have it (and make sure it doesn't add edges for nodes that don't exist anymore)
            if not self.net.has_edge(source, target, key) and new.has_node(source) and new.has_node(target):
                new.add_edge(source, key, target, properties)

        return new


    """ Define NetComp operations (see: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0228728, https://github.com/peterewills/NetComp) """
    def deltacon0(self, other_kg, **kwargs):
        A, B = make_adjacency_matrices(self, other_kg)
        return nc.deltacon0(A, B, **kwargs)

    def vertex_edge_distance(self, other_kg):
        A, B = make_adjacency_matrices(self, other_kg)
        return nc.vertex_edge_distance(A, B)

    def lambda_dist(self, other_kg, **kwargs):
        A, B = make_adjacency_matrices(self, other_kg)
        return nc.lambda_dist(A, B, **kwargs)

    def resistance_distance(self, other_kg, **kwargs):
        A, B = make_adjacency_matrices(self, other_kg)
        return nc.resistance_distance(A, B, **kwargs)

    def conductance_matrix(self):
        A, = make_adjacency_matrices(self) # make_adjacency_matrices returns a list, so trailing comma is shorthand of [0] at end
        return nc.conductance_matrix(A)

    def resistance_matrix(self, **kwargs):
        A, = make_adjacency_matrices(self)
        return nc.resistance_matrix(A, **kwargs)


    """ Define Reasoner Diff operations (see: https://github.com/frostyfan109/NCATS-ReasonerStdAPI-diff) """


    # Override operators for graph operations
    def __add__(self, other):
        """ Returns the simple union of the knowledge graph and other_kg

        :param other_kg: The other knowledge graph
        :type other_kg: :class:`.KnowledgeGraph`
        """
        return self.simple_union(other)
    def __sub__(self, other):
        """ Returns the simple difference of the knowledge graph and other_kg

        :param other_kg: The other knowledge graph
        :type other_kg: :class:`.KnowledgeGraph`
        """
        return self.simple_difference(other)
    def __mul__(self, other):
        return self.cartesian_product(other)

def make_adjacency_matrices(*knowledge_graphs):
    return [nx.adjacency_matrix(G.net) for G in knowledge_graphs]

# Reimplemtnations of NetworkX operators
# See: https://networkx.github.io/documentation/stable/_modules/networkx/algorithms/operators/binary.html
"""
def difference(G, H):
    # Temporary reimplementation of networkx.difference that retains node/edge attributes
    R = nx.create_empty_copy(G)
    edges = G.edges(keys=True, data=True)
    for e in edges:
        source, target, predicate, properties = e
        if not H.has_edge(source, target, key=predicate):
            R.add_edge(source, target, key=predicate, **properties)

    return R

def intersection(G, H):
    # Temporary reimplemtnation of networkx.intersection that retains node/edge attributes
    R = nx.create_empty_copy(G)

    if G.number_of_edges() <= H.number_of_edges():
        edges = G.edges(keys=True, data=True)
        for e in edges:
            source, target, predicate, properties = e
            if H.has_edge(source, target, key=predicate):
                R.add_edge(source, target, key=predicate, **properties)
    else:
        edges = H.edges(keys=True, data=True)
        for e in edges:
            source, target, predicate, properties = e
            if G.has_edge(source, target, key=predicate):
                R.add_edge(source, target, key=predicate, **properties)

    return R
"""
