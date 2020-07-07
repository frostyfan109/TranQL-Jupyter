import json
import os
import ipywidgets
import networkx as nx
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
    def __init__(self, message):
        self.knowledge_graph = KnowledgeGraph(message["knowledge_graph"])
        self.knowledge_map = message["knowledge_map"]
        self.question_graph = message["question_graph"]

class KnowledgeGraph(NetworkxGraph):
    def __init__(self, knowledge_graph):
        super().__init__()

        # You may set the graph_name of a KnowledgeGraph for statistical visualizations
        self.graph_name = None

        if isinstance(knowledge_graph, dict):
            self.build_networkx_graph(knowledge_graph)
        else:
            # Constructed with existing NetworkX graph
            self.net = knowledge_graph


    @classmethod
    def mock1(cls):
        return cls.mock("mock1.json")

    @classmethod
    def mock2(cls):
        return cls.mock("mock2.json")

    @staticmethod
    def mock(name):
        mock_response = json.loads(pkg_resources.read_text(__package__, name))
        return KnowledgeGraph(mock_response["knowledge_graph"])


    # Pandas integration
    @staticmethod
    def from_dataframe_dict(df_dict):
        """ Turns a KnowledgeGraph to a dict of two dataframes (nodes & edges) """
        # A knowledge graph cannot be represented by a data frame, but nodes and edges can be their own separate data frames
        nodes = df_dict["nodes"].to_dict("records")
        edges = df_dict["edges"].to_dict("records")

        return KnowledgeGraph({
            "nodes": nodes,
            "edges": edges
        })

    def to_dataframe_dict(self):
        kg = self.build_knowledge_graph()

        return {
            "nodes": pd.DataFrame(kg["nodes"]),
            "edges": pd.DataFrame(kg["edges"])
        }



    # Statistical visualizations
    @staticmethod
    def plot_graph_sizes(*graphs):
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
        return cls.plot_type_distributions("nodes", graphs)

    @classmethod
    def plot_edge_type_distributions(cls, *graphs):
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
        return force_graph.render3d(self, **kwargs)
    render_force_graph = render_force_graph_3d # alias

    def render_force_graph_2d(self, **kwargs):
        return force_graph.render2d(self, **kwargs)

    def render_plotly_force_graph(self, title="Knowledge Graph"):
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

    def render_node_table(self, columns=("name", "id", "type")):
        return table_view.render_nodes(self, columns=columns)
    def render_edge_table(self, columns=("source_id", "target_id", "type", "weight")):
        return table_view.render_edges(self, columns=columns)

    # Define graph operations (see https://networkx.github.io/documentation/stable/reference/algorithms/operators.html)
    def simple_union(self, other_kg):
        # Returns a KnowledgeGraph of the simple union of self and other_kg (node sets do not have to be disjoint)
        return KnowledgeGraph(nx.compose(self.net, other_kg.net))
    compose = simple_union # alias of simple_union
    def union(self, other_kg):
        # Returns a KnowledgeGraph of the union of self and other_kg  (node sets must be disjoint)
        return KnowledgeGraph(nx.union(self.net, other_kg.net))
    def disjoint_union(self, other_kg):
        # Returns a KnowledgeGraph of the disjoint union of self and other_kg
        return KnowledgeGraph(nx.disjoint_union(self.net, other_kg.net))
    def intersection(self, other_kg):
        # Returns a KnowledgeGraph containing only edges that exist in both self and other_kg
        return KnowledgeGraph(nx.intersection(self.net, other_kg.net))
    def difference(self, other_kg):
        # Returns a KnowledgeGraph containing edges that exist in self but not in other_kg
        return KnowledgeGraph(nx.difference(self.net, other_kg.net))
    def symmetric_difference(self, other_kg):
        # Returns a KnowledgeGraph containing edges that exist in self or other_kg but not both
        return KnowledgeGraph(nx.symmetric_difference(self.net, other_kg.net))
    def cartesian_product(self, other_kg):
        # Returns a KnowledgeGraph of the Cartesian product of self and other_kg
        return KnowledgeGraph(nx.cartesian_product(self.net, other_kg.net))

    def simple_difference(self, other_kg):
        # Return a KnowledgeGraph containing all nodes/edges in self that are not in other_kg

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
        # Return a KnowledgeGraph containing only nodes/edges that exist in both self and other_kg
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
        # Return a KnowledgeGraph containing only nodes/edges that exist in self or other_kg but not both

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



    """
    # nx.difference requires identical node sets and doesn't preserve edge attributes
    def edge_difference(self, other):
        # self_kg, other_kg = self.union_nodes(other)

        return KnowledgeGraph(difference(self.net, other.net))

    def edge_intersection(self, other):
        return KnowledgeGraph(intersection(self.net, other.net))


    def union_nodes(self, other):
        union = self.simple_union(other) # get all nodes
        union.net.remove_edges_from(list(union.net.edges)) # remove edges

        self_kg = KnowledgeGraph(union.net.copy())
        self_kg.net.add_edges_from(self.net.edges(data=True))

        other_kg = KnowledgeGraph(union.net.copy())
        other_kg.net.add_edges_from(other.net.edges(data=True))

        return self_kg, other_kg
    """

    # Override operators for graph operations
    def __add__(self, other):
        return self.simple_union(other)
    def __sub__(self, other):
        return self.simple_difference(other)
    def __mul__(self, other):
        return self.cartesian_product(other)


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
