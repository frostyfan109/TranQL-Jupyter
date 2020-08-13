import IPython
from IPython.display import display, HTML
import json
try:
    import importlib.resources as pkg_resources
except ImportError:
    import importlib_resources as pkg_resources

max_id = 0

class Mode:
    RENDER_2D = 0
    RENDER_3D = 1

def render2d(knowledge_graph, **kwargs):
    return render(knowledge_graph, Mode.RENDER_2D, **kwargs)
def render3d(knowledge_graph, **kwargs):
    return render(knowledge_graph, Mode.RENDER_3D, **kwargs)

def render(knowledge_graph, mode, title=None, width=None, height=400):
    # Very poor docs on how to use JavaScript component libraries within server extensions
    # See source of py_d3 for reference of how they do it
    # Note: max_id is what py_d3 does to identify elements
    global max_id
    max_id += 1
    if mode == Mode.RENDER_2D:
        url = "https://unpkg.com/force-graph"
    elif mode == Mode.RENDER_3D:
        url = "https://unpkg.com/3d-force-graph"
    else:
        raise ValueError(f'Unrecognized mode "{mode}"')
    # Strip .js because for some reason requirejs really wants to add it onto the end making it ".js.js"
    if url[-3:] == ".js": url = url[:-3]

    id = f"force-graph-{max_id}"

    if not isinstance(title, str):
        if isinstance(knowledge_graph.graph_name, str): title = knowledge_graph.graph_name
        else: title = "Knowledge Graph " + str(max_id)
    if not isinstance(width, int):
        width = None
    if not isinstance(height, int):
        height = None

    kg = knowledge_graph.build_knowledge_graph()
    graph_data = {"nodes": [], "links": []}
    for node in kg["nodes"]:
        graph_data["nodes"].append({
            "id": node["id"],
            "name": node.get("name", "<no name>")
        })
    for edge in kg["edges"]:
        graph_data["links"].append({
            "source": edge["source_id"],
            "target": edge["target_id"],
            "name": edge["type"]
        })

    def target_func(comm, open_msg):
        print("Creating comm")
        @comm.on_msg
        def _recv(msg):
            events = msg["content"]["data"]
            for event in events:
                data = events[event]
                if event == "node_clicked":
                    d3_node = data
                    node_id = d3_node["id"]

                    node = knowledge_graph.get_node_by_id(node_id)
                    knowledge_graph.node_click_history.append(node)

                if event == "close_comm":
                    comm.close()
                else:
                    raise Exception(f'Error in Javascript->Python Jupyter Comm message: unknown event type "{event}".')

    IPython.get_ipython().kernel.comm_manager.register_target(id, target_func)

    return display(HTML("""
<div class="%s title-container"></div>
<div class="%s graph-container"></div>
<style>
.%s canvas {
    width: 100%% !important;
    border: 1px solid gray;
}
</style>
<script>
require(['%s', 'https://cdn.jsdelivr.net/npm/element-resize-detector@1.2.1/dist/element-resize-detector.min.js'], function(ForceGraph, resizeMaker) {
    const data = %s;
    const id = "%s";
    const resizeDetector = resizeMaker({ strategy: 'scroll' });
    const container = document.querySelector(`.${id}.graph-container`);
    container.style.overflow = "hidden";
    const width = %s;
    const height = %s;
    if (width !== null) container.style.width = width + "px";
    if (height !== null) container.style.height = height + "px";

    const titleElement = document.querySelector(`.${id}.title-container`);

    const title = document.createElement("span");
    title.textContent = "%s";
    title.style.fontWeight = "bold";

    const titleInfo = document.createElement("span");
    titleInfo.textContent = ` (${data.nodes.length} nodes, ${data.links.length} edges)`;

    titleElement.appendChild(title);
    titleElement.appendChild(titleInfo);

    const graph = ForceGraph()(container).graphData(data)
                                         .width(width)
                                         .height(height);
    resizeDetector.listenTo(container, function(element) {
        graph.width(container.querySelector("canvas").offsetWidth);
        // graph.height(container.querySelector("canvas").offsetHeight);
    });

    /* Create Jupyter Comm connection with Python backend */
    const comm = Jupyter.notebook.kernel.comm_manager.new_comm(id);
    graph.onNodeClick((node) => {
        comm.send({"node_clicked": node});
    });

    // Prevent a memory leak by emptying force graph upon unmount
    // also uninstall the detector
    // also close Comm socket
    // Could use a MutationObserver but it's not really worth the effort,
    // since uninstalling isn't time sensitive, it just has to get done eventually
    const removedInterval = setInterval(() => {
        if (!document.contains(container)) {
            resizeDetector.uninstall(container);
            graph.graphData({nodes: [], links: []});
            comm.send({"close_comm": null})
            comm.close();
            clearInterval(removedInterval);
        }
    }, 1000);

});
</script>
    """ % (
            id,
            id,
            id,
            url,
            json.dumps(graph_data),
            id,
            json.dumps(width),
            json.dumps(height),
            title
        )
    ))
