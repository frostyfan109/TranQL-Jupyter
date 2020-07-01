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

def render2d(knowledge_graph):
    return render(knowledge_graph, Mode.RENDER_2D)
def render3d(knowledge_graph):
    return render(knowledge_graph, Mode.RENDER_3D)

def render(knowledge_graph, mode):
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
    id = f"force-graph-{max_id}" + str(max_id)
    return display(HTML("""
<div id="%s"></div>
<style>
#%s canvas {
    width: 100%% !important;
    height: 75%% !important;
}
</style>
<script>
require(['%s', 'https://cdn.jsdelivr.net/npm/element-resize-detector@1.2.1/dist/element-resize-detector.min.js'], function(ForceGraph, resizeMaker) {
    const parsed_kg = %s;
    const data = {
        nodes: parsed_kg.nodes.map((node) => ({ id: node.id, name: node.name })),
        links: parsed_kg.edges.map((edge) => ({
            source: edge.source_id,
            target: edge.target_id,
            name: edge.type
        }))
    };
    const resizeDetector = resizeMaker({ strategy: 'scroll' });
    const container = document.querySelector("#%s");
    container.style.width = "100%%";
    container.style.height = "100%%";
    const graph = ForceGraph()(container).graphData(data);
    resizeDetector.listenTo(container, function(element) {
        graph.width(container.querySelector("canvas").offsetWidth);
        graph.height(container.querySelector("canvas").offsetHeight);
    });

});
</script>
    """ % (
            id,
            id,
            url,
            json.dumps(knowledge_graph),
            id
        )
    ))
