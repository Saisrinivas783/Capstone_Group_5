import networkx as nx

# Create a directed graph
G = nx.DiGraph()

#################### Trekking #################################

G.add_node("trekking", type="activity")


G.add_node("night", type="context")
G.add_node("rain", type="context")
G.add_node("cold", type="context")


G.add_node("torch", type="item")
G.add_node("raincoat", type="item")
G.add_node("waterproof boots", type="item")
G.add_node("dry bag", type="item")
G.add_node("gps device", type="item")

# Core activity → item
G.add_edge("trekking", "gps device", relation="requires")
G.add_edge("trekking", "dry bag", relation="optional")

# Context → item
G.add_edge("night", "torch", relation="requires")
G.add_edge("rain", "raincoat", relation="requires")
G.add_edge("rain", "waterproof boots", relation="requires")

# Activity → context (if you want to connect them too)
G.add_edge("trekking", "night", relation="contextual")
G.add_edge("trekking", "rain", relation="contextual")

# nx.write_gpickle(G, "knowledge_graph.gpickle")  # Save
# To load later:
# G = nx.read_gpickle("knowledge_graph.gpickle")


###########################################
def get_items_for_activity_and_context(G, activity, context_list=None):
    core_items = set()
    context_items = set()

    # Step 1: Get items directly linked to the activity
    for neighbor in G.successors(activity):
        edge = G.get_edge_data(activity, neighbor)
        if G.nodes[neighbor]["type"] == "item" and edge["relation"] in ["requires", "optional"]:
            core_items.add(neighbor)

    # Step 2: Determine which contexts to check
    # If context_list is empty, auto-fetch all contexts linked to this activity
    if not context_list:
        context_list = [
            neighbor for neighbor in G.successors(activity)
            if G.nodes[neighbor]["type"] == "context"
        ]

    # Step 3: Collect items from context nodes
    for ctx in context_list:
        if G.has_node(ctx):
            for neighbor in G.successors(ctx):
                edge = G.get_edge_data(ctx, neighbor)
                if G.nodes[neighbor]["type"] == "item" and edge["relation"] in ["requires", "optional"]:
                    context_items.add(neighbor)

    return list(core_items), list(context_items)



activity = "trekking"
context = ["rain"]

core_items, contextual_items = get_items_for_activity_and_context(G, activity, context)

print("Core Items from Activity:", core_items)
print("Contextual Items from Context:", contextual_items)
