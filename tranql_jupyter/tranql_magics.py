from IPython.core.magic import (
    Magics, cell_magic, line_magic,
    magics_class
)
# from IPython.core.magic_arguments import argument, magic_arguments, parse_argstring
try:
    from tranql import TranQL
except ImportError:
    import sys, os
    tranql_path = os.environ["tranql_path"]
    sys.path.append(tranql_path)
    from tranql import TranQL

from .tranql_response import TranQLResponse

try:
    from traitlets.config.configurable import Configurable
    from traitlets import Bool, Int, Unicode
except ImportError:
    from IPython.config.configurable import Configurable
    from IPython.utils.traitlets import Bool, Int, Unicode


@magics_class
class TranQLMagic(Magics, Configurable):
    asynchronous = Bool(
        None,
        config=True,
        allow_none=True,
        help="Asynchronously query services for results"
    )
    name_based_merging = Bool(
        None,
        config=True,
        allow_none=True,
        help="Merge nodes with the same name together"
    )
    resolve_names = Bool(
        None,
        config=True,
        allow_none=True,
        help="Resolve equivalent_identifiers of nodes to improve merging of results (should be left disabled due to current implementation problems)"
    )
    dynamic_id_resolution = Bool(
        None,
        config=True,
        allow_none=True,
        help="Enables the interpreter to dynamically resolve common-names into curies (deprecated)"
    )
    registry = Bool(
        None,
        config=True,
        allow_none=True
    )


    def __init__(self, shell):
        Configurable.__init__(self, config=shell.config)
        Magics.__init__(self, shell=shell)

        # Add self to list of modules configurable via %config
        self.shell.configurables.append(self)


    @line_magic("tranql_query")
    def execute(self, line=""):
        tranql = TranQL(options=self._build_interpreter_options())
        context = tranql.execute(line)

        result = context.mem.get("result", None)
        request_errors = context.mem.get("requestErrors", [])

        print(f"Query completed with {len(request_errors)} errors.")

        # Currently scrapping knowledge_map, question_graph, etc. in favor of directly returning a knowledge graph object
        return TranQLResponse(result).knowledge_graph if result != None else None


    def _build_interpreter_options(self):
        options = {}
        if self.asynchronous != None: options["asynchronous"] = self.asynchronous
        if self.name_based_merging != None: options["name_based_merging"] = self.name_based_merging
        if self.resolve_names != None: options["resolve_names"] = self.resolve_names
        if self.dynamic_id_resolution != None: options["dynamic_id_resolution"] = self.dynamic_id_resolution
        if self.registry != None: options["registry"] = self.registry
        return options


def load_ipython_extension(ipython):
    ipython.register_magics(TranQLMagic)
