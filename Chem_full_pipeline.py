import spacy
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network  # For interactive graph visualization

# Paths and filenames
PUBTATOR_FILE = "/Users/adyatrisal/Desktop/chem-nlp-internship/CHR_corpus/train.pubtator"
ENTITIES_CSV = "train_entities_scispacy.csv"
RELATIONS_CSV = "extracted_relations.csv"
GRAPHML_FILE = "chemical_knowledge_graph.graphml"
INTERACTIVE_HTML = "chemical_knowledge_graph.html"  # Interactive graph output

def load_pubtator_texts(pubtator_path):
    texts = []
    with open(pubtator_path, "r", encoding="utf-8") as f:
        current_text = []
        for line in f:
            line = line.strip()
            if line == "":
                if current_text:
                    texts.append(" ".join(current_text))
                    current_text = []
            elif not line.startswith("T") and not line.startswith("A") and "\t" not in line:
                if "|" in line:
                    parts = line.split("|", 2)
                    if len(parts) == 3:
                        current_text.append(parts[2])
                else:
                    current_text.append(line)
        if current_text:
            texts.append(" ".join(current_text))
    return [t for t in texts if t.strip()]

def extract_entities(nlp, texts):
    results = []
    print(f"Processing {len(texts)} documents for NER...")
    for doc_id, text in enumerate(texts):  # Process ALL docs
        doc = nlp(text)
        for ent in doc.ents:
            results.append({
                "doc_id": doc_id,
                "entity": ent.text,
                "label": ent.label_,
                "score": None
            })
        if (doc_id + 1) % 100 == 0:
            print(f"Processed {doc_id+1} docs...")
    return pd.DataFrame(results)

def extract_relations(df_entities):
    relations = []
    grouped = df_entities.groupby("doc_id")
    for doc_id, group in grouped:
        entities = group["entity"].unique()
        for i in range(len(entities)):
            for j in range(i+1, len(entities)):
                relations.append({
                    "doc_id": doc_id,
                    "source": entities[i],
                    "target": entities[j],
                    "relation": "cooccurs"
                })
    return pd.DataFrame(relations)

def build_knowledge_graph(df_relations):
    G = nx.Graph()
    for _, row in df_relations.iterrows():
        G.add_node(row["source"])
        G.add_node(row["target"])
        rel = row["relation"] if pd.notna(row["relation"]) else "unknown"
        G.add_edge(row["source"], row["target"], relation=rel)
    return G

def save_graph(G, path):
    for n, d in G.nodes(data=True):
        for k, v in d.items():
            if v is None:
                d[k] = "None"
    for u, v, d in G.edges(data=True):
        for k, val in d.items():
            if val is None:
                d[k] = "None"
    nx.write_graphml(G, path)
    print(f"Knowledge graph saved to {path}")

def visualize_pruned_graph(df_relations, df_entities, top_n_nodes=50, edge_weight_threshold=2):
    # Keep only CHEMICAL entities
    df_entities = df_entities[df_entities['label'] == 'CHEMICAL']

    # Recompute relations after filtering
    df_relations = df_relations[
        (df_relations['source'].isin(df_entities['entity'])) &
        (df_relations['target'].isin(df_entities['entity']))
    ]

    entity_counts = df_entities['entity'].value_counts()
    top_entities = set(entity_counts.head(top_n_nodes).index)

    filtered_relations = df_relations[
        (df_relations['source'].isin(top_entities)) &
        (df_relations['target'].isin(top_entities))
    ]

    edge_weights = filtered_relations.groupby(['source', 'target']).size().reset_index(name='weight')
    strong_edges = edge_weights[edge_weights['weight'] >= edge_weight_threshold]

    G = nx.Graph()
    for _, row in strong_edges.iterrows():
        G.add_edge(row['source'], row['target'], weight=row['weight'])

    # Smaller nodes, distinctive boundaries
    node_sizes = [min(entity_counts.get(node, 1) * 3, 80) for node in G.nodes()]  # Reduced size
    pos = nx.spring_layout(G, k=0.5, seed=42)

    plt.figure(figsize=(12, 9))
    nx.draw_networkx_nodes(
        G, pos,
        node_size=node_sizes,
        node_color='skyblue',
        alpha=0.85,
        edgecolors='black',  # Distinct boundaries
        linewidths=1.5
    )
    nx.draw_networkx_edges(
        G, pos,
        width=[d['weight'] * 0.5 for (_, _, d) in G.edges(data=True)],
        alpha=0.4,
        edge_color='gray'
    )
    labels = {node: node for node, deg in G.degree() if deg > 1}
    offset = 0.08  # vertical offset to move label below node
    pos_labels = {node: (coords[0], coords[1] - offset) for node, coords in pos.items()}
    nx.draw_networkx_labels(G, pos_labels, labels, font_size=8, verticalalignment='top')

    plt.title(f'Chemical Knowledge Graph (Top {top_n_nodes} CHEMICAL entities, edges ≥ {edge_weight_threshold} co-occurrences)', fontsize=12)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("chemical_knowledge_graph.jpg", dpi=300, bbox_inches="tight")

    plt.show()

    # Interactive version (only CHEMICALS)
    net = Network(height="750px", width="100%", bgcolor="#ffffff", font_color="black", notebook=False)
    net.from_nx(G)
    net.toggle_physics(False)

    for node in net.nodes:
        node["size"] = min(entity_counts.get(node["id"], 1) * 3, 80)  # Smaller
        node["borderWidth"] = 2  # Distinct boundaries
        node["color"] = {
            "border": "black",
            "background": "skyblue"
        }



def main():
    texts = load_pubtator_texts(PUBTATOR_FILE)
    print(f"Loaded {len(texts)} documents from {PUBTATOR_FILE}")

    print("Loading SciSpacy model (en_ner_bc5cdr_md)...")
    nlp = spacy.load("en_ner_bc5cdr_md")

    df_entities = extract_entities(nlp, texts)  # Process all docs now
    df_entities.to_csv(ENTITIES_CSV, index=False)
    print(f"Saved extracted entities to {ENTITIES_CSV}")

    df_relations = extract_relations(df_entities)
    df_relations.to_csv(RELATIONS_CSV, index=False)
    print(f"Saved extracted chemical co-occurrence relations to {RELATIONS_CSV}")

    print("Building knowledge graph ...")
    G = build_knowledge_graph(df_relations)
    save_graph(G, GRAPHML_FILE)

    print("Visualizing pruned knowledge graph ...")
    visualize_pruned_graph(df_relations, df_entities, top_n_nodes=50, edge_weight_threshold=2)

if __name__ == "__main__":
    main()

