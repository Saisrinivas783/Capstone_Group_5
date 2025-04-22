import json
import networkx as nx
import os
import pickle


GRAPH_PATH = "knowledge_graph.pkl"

activity_aliases = {
    "trek": "trekking",
    "trekking": "trekking",
    "hiking": "trekking",
    "camp": "camping",
    "cycle": "cycling",
    "biking": "cycling",
    "gymming": "gym",
    "swim": "swimming",
    "run": "running"
}



def load_knowledge_data(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return data["knowledge_data"]  # nested under "knowledge_data"

def build_graph_with_scoped_contexts(data):
    G = nx.DiGraph()

    for activity, act_info in data.items():
        G.add_node(activity, type="activity")

        # Add core items
        for item, relation in act_info.get("items", {}).items():
            G.add_node(item, type="item")
            G.add_edge(activity, item, relation=relation)

        # Add activity-specific scoped context nodes
        for context, ctx_items in act_info.get("contexts", {}).items():
            scoped_context = f"{activity}_{context}"
            G.add_node(scoped_context, type="context")
            G.add_edge(activity, scoped_context, relation="contextual")

            for item, relation in ctx_items.items():
                G.add_node(item, type="item")
                G.add_edge(scoped_context, item, relation=relation)

    return G

def normalize_activity(activity):
    activity = activity.lower().strip()
    return activity_aliases.get(activity, activity)


def get_activity_items_by_context(G, activity, context_list=None):

    activity=normalize_activity(activity)
    print(activity)
    if not G.has_node(activity):
        return [], []


    core_items = set()
    contextual_items = set()

    # Core activity items
    for neighbor in G.successors(activity):
        if G.nodes[neighbor]["type"] == "item":
            core_items.add(neighbor)

    # Contextual items (only those tied to the activity)
    if context_list:
        for ctx in context_list:
            scoped_ctx = f"{activity}_{ctx}"
            if G.has_edge(activity, scoped_ctx) and G.nodes[scoped_ctx]["type"] == "context":
                for item in G.successors(scoped_ctx):
                    if G.nodes[item]["type"] == "item":
                        contextual_items.add(item)

    return list(core_items), list(contextual_items)



def add_activity_with_items(G, activity_name, item_data):
    # Step 1: Add the activity node
    if not G.has_node(activity_name):
        G.add_node(activity_name, type="activity")
        print(f"🟢 Added activity: {activity_name}")
    else:
        print(f"ℹ️ Activity '{activity_name}' already exists.")

    # Step 2: Add each item and create edges
    for item, relation in item_data.items():
        if not G.has_node(item):
            G.add_node(item, type="item")
            print(f"🟢 Added item: {item}")

        if not G.has_edge(activity_name, item):
            G.add_edge(activity_name, item, relation=relation)
            print(f"🔗 Linked: {activity_name} --[{relation}]--> {item}")
        else:
            print(f"ℹ️ Edge already exists: {activity_name} --[{relation}]--> {item}")

    with open("knowledge_graph.pkl", "wb") as f:
        pickle.dump(G, f)


def load_or_build_graph():
    if os.path.exists(GRAPH_PATH):
        print("📥 Loading existing graph from disk...")
        with open(GRAPH_PATH, "rb") as f:
            G = pickle.load(f)
    else:
        print("🛠️ Building graph from JSON...")
        with open("knowledge_data.json", "r") as f:
            data = json.load(f)["knowledge_data"]
        G = build_graph_with_scoped_contexts(data)
        with open(GRAPH_PATH, "wb") as f:
            pickle.dump(G, f)
        print("💾 Graph saved to disk.")
    return G



if __name__ == "__main__":


    G = load_or_build_graph()

    activity="trekking"
    context=['rain', 'morning']
    core_items, context_items = get_activity_items_by_context(G,activity,context)
    print("core items: ", core_items)
    print("context items: ", context_items)



    # activity_name = "running"
    # item_data= {
    #     "running shoes": "requires",
    #     "water bottle": "requires",
    #     "fitness tracker": "optional",
    #     "cap": "optional"
    # }
    #
    # add_activity_with_items(G, activity_name, item_data)







