import json
import os
import networkx as nx
import seaborn as sns
import plotly.graph_objs as go
from plotly.offline import iplot
from . import force_graph
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
    def render_force_graph_3d(self):
        return force_graph.render3d(self.build_knowledge_graph())
    render_force_graph = render_force_graph_3d # alias

    def render_force_graph_2d(self):
        return force_graph.render2d(self.build_knowledge_graph())

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

    # Override operators for graph operations
    def __add__(self, other):
        return self.simple_union(other)
    def __sub__(self, other):
        return self.difference(other)
    def __mul__(self, other):
        return self.cartesian_product(other)
