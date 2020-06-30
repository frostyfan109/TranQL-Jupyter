import json
import os
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

        self.build_networkx_graph(knowledge_graph)


    @staticmethod
    def mock():
        mock_response = json.loads(pkg_resources.read_text(__package__, "mock.json"))
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
    def render_force_graph(self):
        return force_graph.render(self)


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

    # Override operators for graph operations
    def __add__(self, other):
        return self.simple_union(other)
    def __sub__(self, other):
        return self.difference(other)
