from IPython.display import display, HTML
# import itables.interactive
# from itables import show
import json

max_id = 0

def render_nodes(knowledge_graph, **kwargs):
    return render(knowledge_graph.build_knowledge_graph()["nodes"], **kwargs)

def render_edges(knowledge_graph, **kwargs):
    return render(knowledge_graph.build_knowledge_graph()["edges"], **kwargs)

def render(graph, columns=None):
    global max_id
    max_id += 1
    table_id = f"itable-{max_id}"

    display(HTML("""
<div id="%s" class="ag-theme-alpine"></div>
<script>
// Private namespace
function loadCSS(href) {
    // Include ag-grid styles
    const link = document.createElement("link");
    link.rel = "stylesheet";
    link.type = "text/css";
    link.href = href;
    link.media = "all";
    document.head.appendChild(link);

}
loadCSS("https://cdnjs.cloudflare.com/ajax/libs/ag-grid/23.2.1/styles/ag-grid.min.css");
loadCSS("https://cdnjs.cloudflare.com/ajax/libs/ag-grid/23.2.1/styles/ag-theme-alpine.min.css");
require(['https://cdnjs.cloudflare.com/ajax/libs/ag-grid/23.2.1/ag-grid-community.min.js'], function(AgGrid) {
    const parsed_kg = %s;
    let columns = %s;
    const tableId = "%s";
    const table = document.querySelector("#" + tableId);

    table.style.height = "400px";

    if (columns === null) {
        // Get keys in a flat array, then remove duplicates (long method to support pre-es6)
        columns = parsed_kg.flatMap((e) => Object.keys(e)).reduce(function(acc, cur) {
            if (acc.indexOf(cur) < 0) acc.push(cur);
            return acc;
        }, []);
    }
    function makeColumn(columnName) {
        return {
            headerName: columnName,
            field: columnName,
            sortable: true,
            filter: true
        };
    }
    const columnDefs = columns.map(makeColumn);
    const tableRef = new AgGrid.Grid(table, {
        columnDefs: columnDefs,
        rowData: parsed_kg
    });

    const header = document.createElement("div");

    const addColumn = document.createElement("input");
    addColumn.placeholder = "Add a column...";

    const btn = document.createElement("button");
    btn.textContent = "Add";
    btn.onClick = function() {
        const columnName = addColumn.value;
        addColumn.value = "";
        const defs = tableRef.gridOptions.columnDefs;
        defs.push(makeColumn(columnName));
        tableRef.gridOptions.api.setColumnDefs(defs);
    }

    header.appendChild(addColumn);
    header.appendChild(btn);
    // table.parentNode.insertBefore(header, table);
})
</script>
    """ % (table_id, json.dumps(graph), json.dumps(columns), table_id)))
