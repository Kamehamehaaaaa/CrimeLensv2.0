#!/usr/bin/env python
# coding: utf-8

# In[1]:


import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
import pandas as pd
import numpy as np
from typing import Dict, Optional


# In[2]:


class HypothesisVisualizer:
    """Visualize GNN inference results showing suspect probabilities and crime predictions."""
    
    def __init__(self, model, graph_builder):
        self.model = model
        self.graph_builder = graph_builder
        
        # Color maps
        self.suspect_cmap = LinearSegmentedColormap.from_list(
            'suspect', ['#90EE90', '#FFFF00', '#FF4500', '#8B0000']  # Green -> Yellow -> Red -> Dark Red
        )
        self.object_color = '#4ECDC4'
        self.location_color = '#95E1D3'
    
    def generate_hypothesis_with_viz(self, scene_df: pd.DataFrame, 
                                      figsize: tuple = (16, 10),
                                      save_path: Optional[str] = None) -> Dict:
        """
        Generate hypotheses and visualize the graph with prediction scores.
        
        Shows:
        - Person nodes colored by suspect likelihood (green=low, red=high)
        - Node size proportional to suspect score
        - Crime type predictions as a bar chart
        - Edge labels showing relationships
        """
        self.model.eval()
        
        with torch.no_grad():
            # Build graph and run inference
            graph = self.graph_builder.build_graph(scene_df)
            output = self.model(graph)
            
            # Get predictions
            crime_probs = F.softmax(output['crime_logits'], dim=1).squeeze().numpy()
            suspect_scores = torch.sigmoid(output['suspect_scores']).numpy()
            
            # Get entity names
            persons = graph.metadata_dict['persons']
            objects = graph.metadata_dict['objects']
            locations = graph.metadata_dict['locations']
        
        # Create figure with subplots
        fig = plt.figure(figsize=figsize)
        
        # Main graph visualization (left, larger)
        ax_graph = fig.add_axes([0.02, 0.1, 0.6, 0.8])
        
        # Crime type bar chart (top right)
        ax_crime = fig.add_axes([0.68, 0.55, 0.28, 0.35])
        
        # Suspect scores bar chart (bottom right)
        ax_suspect = fig.add_axes([0.68, 0.1, 0.28, 0.35])
        
        # ============ GRAPH VISUALIZATION ============
        G = nx.MultiDiGraph()
        
        # Add person nodes with suspect scores
        for i, (person, score) in enumerate(zip(persons, suspect_scores)):
            G.add_node(f"person_{i}", 
                      label=f"{person}\n({score*100:.1f}%)", 
                      node_type='person',
                      suspect_score=score)
        
        # Add object nodes
        for i, obj in enumerate(objects):
            G.add_node(f"object_{i}", label=obj, node_type='object')
        
        # Add location nodes
        for i, loc in enumerate(locations):
            G.add_node(f"location_{i}", label=loc, node_type='location')
        
        # Create mappings
        person_to_idx = {p: i for i, p in enumerate(persons)}
        object_to_idx = {o: i for i, o in enumerate(objects)}
        location_to_idx = {l: i for i, l in enumerate(locations)}
        
        # Add edges from events
        for _, event in scene_df.iterrows():
            suspect_idx = person_to_idx[event['suspect']]
            victim_idx = person_to_idx[event['victim']]
            obj_idx = object_to_idx[event['object']]
            loc_idx = location_to_idx[event['location']]
            action = event['action']
            
            G.add_edge(f"person_{suspect_idx}", f"person_{victim_idx}",
                      label=action, edge_type='acts_on')
            G.add_edge(f"person_{suspect_idx}", f"object_{obj_idx}",
                      label='uses', edge_type='uses')
            G.add_edge(f"person_{suspect_idx}", f"location_{loc_idx}",
                      label='at', edge_type='at')
        
        # Layout
        pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
        
        # Draw person nodes (colored by suspect score)
        person_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'person']
        person_colors = [self.suspect_cmap(G.nodes[n]['suspect_score']) for n in person_nodes]
        person_sizes = [1500 + 2000 * G.nodes[n]['suspect_score'] for n in person_nodes]
        
        nx.draw_networkx_nodes(G, pos, nodelist=person_nodes, node_color=person_colors,
                               node_size=person_sizes, alpha=0.9, ax=ax_graph)
        
        # Draw object nodes
        object_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'object']
        nx.draw_networkx_nodes(G, pos, nodelist=object_nodes, node_color=self.object_color,
                               node_size=1200, alpha=0.9, ax=ax_graph, node_shape='s')
        
        # Draw location nodes
        location_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'location']
        nx.draw_networkx_nodes(G, pos, nodelist=location_nodes, node_color=self.location_color,
                               node_size=1200, alpha=0.9, ax=ax_graph, node_shape='^')
        
        # Draw labels
        labels = {n: G.nodes[n]['label'] for n in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=9, 
                                font_weight='bold', ax=ax_graph)
        
        # Draw edges
        edge_colors_map = {'acts_on': '#E74C3C', 'uses': '#3498DB', 'at': '#2ECC71'}
        for edge_type, color in edge_colors_map.items():
            edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('edge_type') == edge_type]
            if edges:
                nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=color,
                                       width=2, alpha=0.7, arrows=True, arrowsize=20,
                                       connectionstyle="arc3,rad=0.1", ax=ax_graph)
        
        # Edge labels
        edge_labels = {(u, v): d['label'] for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                                     font_size=8, ax=ax_graph)
        
        # Graph title and legend
        crime_type_actual = scene_df['crime_type'].iloc[0]
        scene_id = scene_df['scene_id'].iloc[0]
        ax_graph.set_title(f"Scene {scene_id} - Hypothesis Graph\nActual Crime: {crime_type_actual}",
                          fontsize=14, fontweight='bold')
        
        # Add colorbar for suspect likelihood
        sm = plt.cm.ScalarMappable(cmap=self.suspect_cmap, norm=plt.Normalize(0, 1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax_graph, shrink=0.5, label='Suspect Likelihood')
        
        ax_graph.axis('off')
        
        # ============ CRIME TYPE PREDICTIONS ============
        crime_types = self.graph_builder.entity_encoders['crime_type'].classes_
        crime_types = [c for c in crime_types if c != '<UNK>']
        
        # Filter to only show probs for actual crime types
        display_probs = crime_probs[:len(crime_types)]
        
        colors = ['#E74C3C' if ct == crime_type_actual else '#3498DB' for ct in crime_types]
        bars = ax_crime.barh(crime_types, display_probs, color=colors, alpha=0.8)
        ax_crime.set_xlabel('Probability', fontsize=10)
        ax_crime.set_title('Crime Type Predictions', fontsize=12, fontweight='bold')
        ax_crime.set_xlim(0, 1)
        
        # Add percentage labels
        for bar, prob in zip(bars, display_probs):
            ax_crime.text(prob + 0.02, bar.get_y() + bar.get_height()/2,
                         f'{prob*100:.1f}%', va='center', fontsize=9)
        
        # Highlight actual crime type
        ax_crime.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
        
        # ============ SUSPECT LIKELIHOOD SCORES ============
        sorted_indices = np.argsort(suspect_scores)[::-1]
        sorted_persons = [persons[i] for i in sorted_indices]
        sorted_scores = [suspect_scores[i] for i in sorted_indices]
        
        # Color bars by score
        bar_colors = [self.suspect_cmap(s) for s in sorted_scores]
        bars = ax_suspect.barh(sorted_persons, sorted_scores, color=bar_colors, alpha=0.8)
        ax_suspect.set_xlabel('Suspect Likelihood', fontsize=10)
        ax_suspect.set_title('Person Suspect Scores', fontsize=12, fontweight='bold')
        ax_suspect.set_xlim(0, 1)
        
        # Add percentage labels
        for bar, score in zip(bars, sorted_scores):
            ax_suspect.text(score + 0.02, bar.get_y() + bar.get_height()/2,
                           f'{score*100:.1f}%', va='center', fontsize=9)
        
        # Mark actual suspect
        actual_suspect = scene_df['suspect'].iloc[0]
        for i, person in enumerate(sorted_persons):
            if person == actual_suspect:
                ax_suspect.get_yticklabels()[i].set_color('red')
                ax_suspect.get_yticklabels()[i].set_fontweight('bold')
        
        plt.suptitle('GNN Crime Hypothesis Generation Results', fontsize=16, 
                    fontweight='bold', y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()
        
        # Return hypothesis data
        return {
            'crime_predictions': {ct: float(p) for ct, p in zip(crime_types, display_probs)},
            'suspect_scores': {p: float(s) for p, s in zip(persons, suspect_scores)},
            'actual_crime': crime_type_actual,
            'actual_suspect': actual_suspect
        }


# In[3]:


# ============================================================
# USAGE FUNCTION
# ============================================================

def visualize_hypothesis(model, graph_builder, df, scene_id=None, save_path=None):
    """
    Quick function to visualize hypothesis for a scene.
    
    Usage:
        visualize_hypothesis(model, graph_builder, df)
        visualize_hypothesis(model, graph_builder, df, scene_id=5)
    """
    if scene_id is None:
        scene_id = df['scene_id'].unique()[0]
    
    scene_df = df[df['scene_id'] == scene_id]
    
    if scene_df.empty:
        print(f"Error: Scene {scene_id} not found!")
        return None
    
    visualizer = HypothesisVisualizer(model, graph_builder)
    results = visualizer.generate_hypothesis_with_viz(scene_df, save_path=save_path)
    
    print("\n" + "="*50)
    print("HYPOTHESIS SUMMARY")
    print("="*50)
    print(f"\nActual Crime Type: {results['actual_crime']}")
    print(f"Actual Suspect: {results['actual_suspect']}")
    print(f"\nTop Crime Prediction: {max(results['crime_predictions'], key=results['crime_predictions'].get)}")
    print(f"Top Suspect: {max(results['suspect_scores'], key=results['suspect_scores'].get)}")
    
    return results


# In[ ]:




