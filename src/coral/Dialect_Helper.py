from typing import List

import graphviz
import treelib


class Dialect_Helper:
    """Helper class for handling dialects."""

    def __init__(self):
        """Initialize the dialect tree. and build dialect tree."""
        self.dialect_tree = treelib.Tree()
        self.build_tree()

    def build_tree(self):
        """Build the dialect tree."""
        # Add root node

        # Add the root node
        self.dialect_tree.create_node("dansk", "dansk")  # Root node

        # Data to be added to the tree
        data = [
            ("dansk", "jysk", "nørrejysk", "vestjysk", "thybomål"),
            ("dansk", "jysk", "nørrejysk", "vestjysk", "morsingmål"),
            ("dansk", "jysk", "nørrejysk", "vestjysk", "sallingmål"),
            ("dansk", "jysk", "nørrejysk", "vestjysk", "hardsysselsk"),
            ("dansk", "jysk", "nørrejysk", "vestjysk", "fjandbomål"),
            ("dansk", "jysk", "nørrejysk", "vestjysk", "sydvestjysk (m. fanø)"),
            ("dansk", "jysk", "nørrejysk", "vestjysk", "sydøstjysk"),
            (
                "dansk",
                "jysk",
                "nørrejysk",
                "østjysk",
                "vendsysselsk (m. hanherred og læsø)",
            ),
            ("dansk", "jysk", "nørrejysk", "østjysk", "himmerlandsk"),
            ("dansk", "jysk", "nørrejysk", "østjysk", "ommersysselsk"),
            (
                "dansk",
                "jysk",
                "nørrejysk",
                "østjysk",
                "djurslandsk (nord-, syddjurs m. nord- og sydsamsø, anholt)",
            ),
            ("dansk", "jysk", "nørrejysk", "østjysk", "midtøstjysk"),
            (
                "dansk",
                "jysk",
                "sønderjysk",
                "sønderjysk",
                "vestlig sønderjysk (m. mandø og rømø)",
            ),
            (
                "dansk",
                "jysk",
                "sønderjysk",
                "sønderjysk",
                "østligt sønderjysk (m. als)",
            ),
            (
                "dansk",
                "jysk",
                "sønderjysk",
                "sønderjysk",
                "syd for rigsgrænsen: mellemslesvisk, angelmål, fjoldemål",
            ),
            ("dansk", "bornholmsk", "bornholmsk", "bornholmsk", "bornholmsk"),
            ("dansk", "ømål", "amagermål", "amagermål", "københavnsk"),
            ("dansk", "ømål", "sjællandsk", "sjællandsk", "sjællandsk"),
            ("dansk", "ømål", "sjællandsk", "sjællandsk", "nordsjællandsk"),
            ("dansk", "ømål", "sjællandsk", "sjællandsk", "nordvestsjællandsk"),
            ("dansk", "ømål", "sjællandsk", "sjællandsk", "sydvestsjællandsk"),
            ("dansk", "ømål", "sjællandsk", "sjællandsk", "østsjællandsk"),
            (
                "dansk",
                "ømål",
                "sjællandsk",
                "sjællandsk",
                "sydsjællandsk (sydligt sydsjællandsk)",
            ),
            ("dansk", "ømål", "sydømål", "sydømål", "østmønsk"),
            ("dansk", "ømål", "sydømål", "sydømål", "vestmønsk"),
            ("dansk", "ømål", "sydømål", "sydømål", "nordfalstersk"),
            ("dansk", "ømål", "sydømål", "sydømål", "sydfalstersk"),
            ("dansk", "ømål", "sydømål", "sydømål", "lollandsk"),
            ("dansk", "ømål", "fynsk", "fynsk", "østfynsk"),
            ("dansk", "ømål", "fynsk", "fynsk", "vestfynsk (nordvest-, sydvestfynsk)"),
            ("dansk", "ømål", "fynsk", "fynsk", "sydfynsk"),
            ("dansk", "ømål", "fynsk", "fynsk", "langelandsk"),
            ("dansk", "ømål", "fynsk", "fynsk", "tåsingsk (m. thurø)"),
            (
                "dansk",
                "ømål",
                "fynsk",
                "fynsk",
                "ærøsk (m. lyø, avernakø, strynø, birkholm, drejø)",
            ),
        ]

        # Add nodes to the tree
        for path in data:
            parent_id = "dansk"  # Start from the root node ID

            for level, tag in enumerate(
                path[1:]
            ):  # Skip the root node as it's already handled
                node_id = (
                    "_".join(path[: level + 2]).lower().replace(":", "")
                )  # Create a unique identifier for the node in lowercase
                if not self.dialect_tree.contains(
                    node_id
                ):  # Check if the node is already in the tree
                    self.dialect_tree.create_node(
                        tag, node_id, parent=parent_id
                    )  # Add node
                parent_id = node_id  # Update parent for the next level

    def convert_to_graph(self):
        """Convert the tree to a Graphviz graph."""
        try:
            dot = graphviz.Digraph(comment="Dialect Tree")

            # Add nodes to the Graphviz graph
            for node in self.dialect_tree.all_nodes():
                dot.node(node.identifier, label=node.tag)

            # Add edges to the Graphviz graph
            for node in self.dialect_tree.all_nodes():
                parent = self.dialect_tree.parent(
                    node.identifier
                )  # Get the parent node
                if parent:
                    dot.edge(parent.identifier, node.identifier)

            return dot
        except ImportError:
            print("Graphviz library is not installed. ")
            return None

    def visualize_tree(self):
        """Visualize the dialect tree."""
        self.dialect_tree.show()
        return

    def get_nodes_sorted_by_depth(self, reverse=False) -> List[treelib.node.Node]:
        """Returns a list of all nodes in the tree sorted by depth.

        :return: List of nodes sorted by depth.
        """
        # Dictionary to hold nodes by depth
        nodes_by_depth: dict[int, treelib.node.Node] = {}

        # Traverse all nodes in the tree
        for node in self.dialect_tree.all_nodes():
            depth = self.dialect_tree.depth(node)
            if depth not in nodes_by_depth:
                nodes_by_depth[depth] = []
            nodes_by_depth[depth].append(node)

        # Sort nodes by depth
        sorted_nodes = []
        for depth in sorted(nodes_by_depth.keys(), reverse=reverse):
            sorted_nodes.extend(nodes_by_depth[depth])

        return sorted_nodes

    def _get_node_by_dialect(self, dialect):
        """Get the node by dialect.

        :param dialect: The dialect to lookup.
        :return: The node with the specified tag, or None if not found.
        """
        for node in self.get_nodes_sorted_by_depth(reverse=True):
            if node.tag == dialect:
                return node
        return None

    def convert_to_depth(self, dialect, depth):
        """Find the parent node at a specific depth.

        :param dialect: The dialect to lookup.
        :param depth: The depth of the parent node to retrieve.
        :return: The parent node at the specified depth, or None if not found.
        """
        node = self._get_node_by_dialect(dialect)
        depth_current = self.dialect_tree.depth(node)

        if node is None:
            return None

        if depth_current < depth:
            return "uspecificeret på niveau " + str(depth)
        else:
            while depth_current > depth:
                node = self.dialect_tree.parent(node.identifier)
                depth_current = self.dialect_tree.depth(node)

            return node.tag


if __name__ == "__main__":
    # Usage example
    dialects = Dialect_Helper()
    nodes = dialects.get_nodes_sorted_by_depth()

    nodes = nodes
    # dialects.visualize_tree()
    # print(dialects.convert_to_depth('himmerlandsk', 2))
    # a = 0
