from IPython.display import HTML
import json
try:
    import importlib.resources as pkg_resources
except ImportError:
    import importlib_resources as pkg_resources
try:
    from urllib.request import urlopen
except ImportError:
    from urllib import urlopen

max_id = 0

def render(knowledge_graph):
    # Very poor docs on how to use JavaScript component libraries within server extensions
    # See source of py_d3 for reference of how they do it
    # Note: max_id is what py_d3 does to identify elements
    global max_id
    max_id += 1
    url = urlopen("https://unpkg.com/3d-force-graph").url
    # Strip .js because for some reason requirejs really wants to add it onto the end making it ".js.js"
    if url[-3:] == ".js": url = url[:-3]
    id = "force-graph-3d-" + str(max_id)
    return HTML("""
<div id="%s"></div>
<style>
#%s canvas {
    width: 100%% !important;
    height: 75%% !important;
}
</style>
<script>
requirejs.config({
  paths: {
    '3d-force-graph': '%s',
    'element-resize-detector': 'https://cdn.jsdelivr.net/npm/element-resize-detector@1.2.1/dist/element-resize-detector.min'
  }
});
require(['3d-force-graph', 'element-resize-detector'], function(ForceGraph3D, resizeMaker) {
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
    const graph = ForceGraph3D()(container).graphData(data);
    resizeDetector.listenTo(container, function(element) {
        graph.width(container.querySelector("canvas").offsetWidth);
        graph.height(container.querySelector("canvas").offsetHeight);
    });

});
</script>
    """ % (id, id, url, json.dumps(knowledge_graph.build_knowledge_graph()), id))
