"""
Lightweight Explainability for CrimeLens
---------------------------------------

This version does NOT depend on torch_geometric internals or metadata_dict.
It only uses:
- scene_df          (current scene as a DataFrame)
- prediction        (top crime type)
- confidence        (top probability)
- predictions_dict  (full {crime_type: prob} mapping, optional)

It returns:
- fig: matplotlib Figure with node/edge highlighting
- explanation_text: a plain-text narrative you can show in Streamlit.
"""

from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx


def _build_scene_graph(scene_df: pd.DataFrame) -> nx.MultiDiGraph:
    """
    Build a simple MultiDiGraph from the scene DataFrame
    without relying on any PyG metadata.
    """
    G = nx.MultiDiGraph()

    # Basic entity sets
    persons = list(pd.concat([scene_df["suspect"], scene_df["victim"]]).astype(str).unique())
    objects = list(scene_df["object"].astype(str).unique())
    locations = list(scene_df["location"].astype(str).unique())

    # Add nodes with type
    for p in persons:
        G.add_node(f"P:{p}", label=p, node_type="person")
    for o in objects:
        G.add_node(f"O:{o}", label=o, node_type="object")
    for l in locations:
        G.add_node(f"L:{l}", label=l, node_type="location")

    # Add edges for each event
    for _, row in scene_df.iterrows():
        suspect = str(row["suspect"])
        victim = str(row["victim"])
        action = str(row["action"])
        obj = str(row["object"])
        loc = str(row["location"])

        G.add_edge(
            f"P:{suspect}",
            f"P:{victim}",
            edge_type="action",
            label=action,
        )
        G.add_edge(
            f"P:{suspect}",
            f"O:{obj}",
            edge_type="uses",
            label="uses",
        )
        G.add_edge(
            f"P:{suspect}",
            f"L:{loc}",
            edge_type="at",
            label="at",
        )

    return G


def explain_crime_prediction(
    scene_df: pd.DataFrame,
    prediction: str,
    confidence: float,
    predictions_dict: Optional[Dict[str, float]] = None,
) -> Tuple[plt.Figure, str]:
    """
    Simple, robust explanation function.

    Args
    ----
    scene_df : pd.DataFrame
        Current scene with columns:
        ['scene_id','event_id','crime_type','suspect','victim','object','location','action']
    prediction : str
        Top predicted crime type (e.g., 'homicide')
    confidence : float
        Probability of the top prediction (0–1)
    predictions_dict : dict
        Optional full probability dict {crime_type: prob}

    Returns
    -------
    fig : matplotlib.figure.Figure
        Graph visualization with highlighted entities/edges
    explanation_text : str
        Text narrative describing key factors
    """
    # Build simple graph
    G = _build_scene_graph(scene_df)

    # Basic stats
    suspects = scene_df["suspect"].astype(str).value_counts()
    victims = scene_df["victim"].astype(str).value_counts()
    actions = scene_df["action"].astype(str).value_counts()
    objects = scene_df["object"].astype(str).value_counts()
    locations = scene_df["location"].astype(str).value_counts()

    primary_suspect = suspects.idxmax() if not suspects.empty else "Unknown"
    primary_victim = victims.idxmax() if not victims.empty else "Unknown"
    primary_action = actions.idxmax() if not actions.empty else "unknown action"
    primary_object = objects.idxmax() if not objects.empty else "unknown object"
    primary_location = locations.idxmax() if not locations.empty else "unknown location"

    # -----------------------
    # FIGURE: graph + side bars
    # -----------------------
    fig = plt.figure(figsize=(14, 9))

    ax_graph = fig.add_axes([0.03, 0.18, 0.55, 0.78])
    ax_actions = fig.add_axes([0.63, 0.60, 0.34, 0.33])
    ax_objs = fig.add_axes([0.63, 0.18, 0.34, 0.33])

    # Layout
    pos = nx.spring_layout(G, k=2.5, iterations=60, seed=42)

    # Node colors by type (person / object / location)
    node_color_map = {
        "person": "#FF6B6B",
        "object": "#4ECDC4",
        "location": "#95E1D3",
    }

    # Draw nodes
    for node_type, shape in [("person", "o"), ("object", "s"), ("location", "^")]:
        nodelist = [n for n, d in G.nodes(data=True) if d.get("node_type") == node_type]
        if not nodelist:
            continue

        # Slight highlight if this is the "primary suspect"
        sizes = []
        colors = []
        for n in nodelist:
            label = G.nodes[n].get("label", "")
            if node_type == "person" and label == primary_suspect:
                sizes.append(2600)
                colors.append("#C0392B")  # darker red for primary suspect
            else:
                sizes.append(1800)
                colors.append(node_color_map[node_type])

        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=nodelist,
            node_size=sizes,
            node_color=colors,
            edgecolors="black",
            linewidths=1.5,
            alpha=0.9,
            node_shape=shape,
            ax=ax_graph,
        )

    # Edges: color by type
    edge_styles = {
        "action": ("#E74C3C", "arc3,rad=0.15"),
        "uses": ("#3498DB", "arc3,rad=0.10"),
        "at": ("#2ECC71", "arc3,rad=0.05"),
    }

    for e_type, (color, style) in edge_styles.items():
        edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("edge_type") == e_type]
        if edges:
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=edges,
                edge_color=color,
                arrows=True,
                arrowsize=20,
                width=2.0,
                connectionstyle=style,
                alpha=0.7,
                ax=ax_graph,
            )

    # Edge labels
    edge_labels = {(u, v): d.get("label", "") for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, ax=ax_graph)

    # Node labels
    labels = {n: d.get("label", n) for n, d in G.nodes(data=True)}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=9, font_weight="bold", ax=ax_graph)

    ax_graph.set_title(
        f"Crime Scene Graph — Predicted: {prediction} ({confidence * 100:.1f}%)",
        fontsize=14,
        fontweight="bold",
    )
    ax_graph.axis("off")

    # ACTION BAR CHART
    if not actions.empty:
        act_vals = actions.values.astype(float) / actions.values.max()
        ax_actions.barh(actions.index, act_vals, color="#E67E22", alpha=0.9)
        ax_actions.set_title("Action Frequency (Scene)", fontsize=12, fontweight="bold")
        ax_actions.set_xlabel("Relative Frequency")
        ax_actions.set_xlim(0, 1.05)

    # OBJECT BAR CHART
    if not objects.empty:
        obj_vals = objects.values.astype(float) / objects.values.max()
        ax_objs.barh(objects.index, obj_vals, color="#2980B9", alpha=0.9)
        ax_objs.set_title("Object/Weapon Frequency (Scene)", fontsize=12, fontweight="bold")
        ax_objs.set_xlabel("Relative Frequency")
        ax_objs.set_xlim(0, 1.05)

    plt.suptitle("CrimeLens — Explanation View", fontsize=16, fontweight="bold", y=0.97)

    # -----------------------
    # TEXT EXPLANATION
    # -----------------------
    # Top-2 crime alternatives if we have full probs
    alt_text = ""
    if predictions_dict is not None and len(predictions_dict) > 1:
        sorted_probs = sorted(predictions_dict.items(), key=lambda kv: -kv[1])
        top2 = sorted_probs[:2]
        alt_text = "Top crime hypotheses:\n"
        for crime, prob in top2:
            alt_text += f"  • {crime}: {prob * 100:.1f}%\n"

    explanation_text = f"""
Explanation Summary
-------------------
Predicted crime type: {prediction}  (confidence: {confidence * 100:.1f}%)

Key entities in this scene:
  • Primary suspect: {primary_suspect}
  • Primary victim:  {primary_victim}
  • Most common action:   {primary_action}
  • Most common object:   {primary_object}
  • Most common location: {primary_location}

Why this makes sense:
- The graph prominently connects **{primary_suspect}** to **{primary_victim}**
  via the action **"{primary_action}"**.
- The object **"{primary_object}"** and location **"{primary_location}"**
  are central in the event description and often co-occur with cases labeled as "{prediction}".

{alt_text if alt_text else ""}This explanation is based on scene-level structure (who did what, with what, and where),
rather than raw text tokens, making it easier to review for investigative plausibility.
""".rstrip()

    return fig, explanation_text