from IPython.display import display, HTML
import itables.interactive
from itables import show

max_id = 0

def render_nodes(knowledge_graph, **kwargs):
    dfs = knowledge_graph.to_dataframe_dict()
    df = dfs["nodes"]
    return render(df, **kwargs)

def render_edges(knowledge_graph, **kwargs):
    dfs = knowledge_graph.to_dataframe_dict()
    df = dfs["edges"]
    return render(df, **kwargs)

def render(df, columns=None):
    global max_id
    max_id += 1
    table_id = f"itable-{max_id}"
    plugin_id = f"for-{table_id}"

    # If `columns` is specified, select only those columns in the data frame
    if isinstance(columns, (list, tuple)): df = df[list(columns)]

    show(df, table_id=table_id)
    '''
    display(HTML("""
<div id="%s"></div>
    """ % plugin_id))
    '''
    '''
    display(HTML("""
<script>
// Private namespace
(function() {
    const pluginId = "%s";
    const tableId = "%s";
    const pluginContainer = document.querySelector("#" + pluginId);
    const table = document.querySelector("#" + tableId);
    const x = document.createElement("p");
    x.textContent = "test";
    pluginContainer.appendChild(x);
})()
</script>
    """ % plugin_id, table_id))
    '''
