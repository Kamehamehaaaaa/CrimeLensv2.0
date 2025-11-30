"""
CrimeLens Streamlit UI
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import networkx as nx
import re
import os
from typing import Dict, List, Optional
from pathlib import Path

# Import from your FinalGenerator
from FinalGenerator import (
    CrimeGraphBuilder, 
    CrimeGNN, 
    CrimeGraphVisualizer,
    GNNExplainer,
    run_complete_pipeline
)

from NERExtraction import ImprovedCrimeSentenceParser

# ============================================================
# SENTENCE PARSER - Extracts entities from natural language
# ============================================================

BASE_DIR = Path(os.getcwd()).resolve()
MODEL_PATH = os.path.join(BASE_DIR, "model", "model_test1.pt")
DATA_PATH = os.path.join(BASE_DIR, "data", "realistic_crime_data_3.csv")

print(os.listdir(BASE_DIR))

class CrimeSentenceParser:
    """
    NER-based parser using spaCy for entity extraction.
    Falls back to heuristic patterns if spaCy fails.
    """
    
    def __init__(self, graph_builder: CrimeGraphBuilder):
        self.graph_builder = graph_builder
        
        # Get known entities from the trained model
        self.known_actions = set(graph_builder.action_encoder.classes_) if hasattr(graph_builder, 'action_encoder') else set()
        self.known_objects = set(graph_builder.object_encoder.classes_) if hasattr(graph_builder, 'object_encoder') else set()
        self.known_locations = set(graph_builder.location_encoder.classes_) if hasattr(graph_builder, 'location_encoder') else set()
        
        # Try to initialize NER parser
        self.use_ner = True
        try:
            from NERExtraction import NEREntityExtractor
            self.ner_extractor = NEREntityExtractor(
                known_actions=self.known_actions,
                known_objects=self.known_objects,
                known_locations=self.known_locations
            )
            # Get default crime type
            self.default_crime = graph_builder.crime_encoder.classes_[0] if hasattr(graph_builder, 'crime_encoder') else 'homicide'
            print("✓ Using NER-based parsing")
        except (ImportError, OSError) as e:
            self.use_ner = False
            print(f"⚠ NER unavailable ({e}), using heuristic parsing")
            self._init_heuristic_patterns()
    
    def _init_heuristic_patterns(self):
        """Initialize heuristic fallback parser patterns."""
        # Common action patterns
        self.action_patterns = {
            'stabbed': ['stabbed', 'stab', 'stabbing', 'knifed'],
            'shot': ['shot', 'shoot', 'shooting', 'fired at', 'gunned'],
            'threatened': ['threatened', 'threat', 'threatening', 'menaced'],
            'robbed': ['robbed', 'rob', 'robbing', 'mugged', 'stole from'],
            'attacked': ['attacked', 'attack', 'assaulted', 'assault', 'beat'],
            'punched': ['punched', 'punch', 'hit', 'struck'],
            'kicked': ['kicked', 'kick', 'kicking'],
            'fled': ['fled', 'escape', 'ran', 'ran away', 'got away'],
            'strangled': ['strangled', 'strangle', 'choked', 'choke'],
            'poisoned': ['poisoned', 'poison', 'drugged'],
            'broke into': ['broke into', 'break in', 'burgled', 'burglarized'],
            'grabbed': ['grabbed', 'grab', 'snatched', 'seized'],
        }
        
        # Common object patterns
        self.object_patterns = {
            'knife': ['knife', 'blade', 'dagger', 'machete', 'switchblade'],
            'gun': ['gun', 'pistol', 'firearm', 'revolver', 'handgun', 'rifle'],
            'bat': ['bat', 'baseball bat', 'club'],
            'wallet': ['wallet', 'purse', 'money'],
            'phone': ['phone', 'cellphone', 'mobile', 'smartphone'],
            'jewelry': ['jewelry', 'jewellery', 'necklace', 'ring', 'watch'],
        }
        
        # Common location patterns
        self.location_patterns = {
            'apartment': ['apartment', 'flat', 'unit', 'condo'],
            'home': ['home', 'house', 'residence', 'dwelling', 'room', 'bedroom', 'living room'],
            'street': ['street', 'road', 'sidewalk', 'street corner'],
            'parking lot': ['parking lot', 'parking', 'garage', 'car park'],
            'alleyway': ['alleyway', 'alley', 'back alley'],
            'bar': ['bar', 'pub', 'tavern', 'nightclub', 'club'],
            'park': ['park', 'playground', 'garden'],
            'office': ['office', 'workplace', 'building'],
            'warehouse': ['warehouse', 'storage', 'factory'],
        }
        
        # Get default crime type
        self.default_crime = self.graph_builder.crime_encoder.classes_[0] if hasattr(self.graph_builder, 'crime_encoder') else 'homicide'
        
        # Get known entities from the trained model
        self.known_actions = set(self.graph_builder.action_encoder.classes_)
        self.known_objects = set(self.graph_builder.object_encoder.classes_)
        self.known_locations = set(self.graph_builder.location_encoder.classes_)
        
        # Common action patterns
        self.action_patterns = {
            'stabbed': ['stabbed', 'stab', 'stabbing', 'knifed'],
            'shot': ['shot', 'shoot', 'shooting', 'fired at', 'gunned'],
            'threatened': ['threatened', 'threat', 'threatening', 'menaced'],
            'robbed': ['robbed', 'rob', 'robbing', 'mugged', 'stole from'],
            'attacked': ['attacked', 'attack', 'assaulted', 'assault', 'beat'],
            'punched': ['punched', 'punch', 'hit', 'struck'],
            'kicked': ['kicked', 'kick', 'kicking'],
            'fled': ['fled', 'escape', 'ran', 'ran away', 'got away'],
            'strangled': ['strangled', 'strangle', 'choked', 'choke'],
            'poisoned': ['poisoned', 'poison', 'drugged'],
            'broke into': ['broke into', 'break in', 'burgled', 'burglarized'],
            'grabbed': ['grabbed', 'grab', 'snatched', 'seized'],
            'demanded': ['demanded', 'demand', 'asked for'],
            'held at gunpoint': ['held at gunpoint', 'gunpoint', 'held up'],
        }
        
        # Common object patterns
        self.object_patterns = {
            'knife': ['knife', 'blade', 'dagger', 'machete', 'switchblade'],
            'gun': ['gun', 'pistol', 'firearm', 'revolver', 'handgun', 'rifle'],
            'bat': ['bat', 'baseball bat', 'club'],
            'wallet': ['wallet', 'purse', 'money'],
            'phone': ['phone', 'cellphone', 'mobile', 'smartphone'],
            'jewelry': ['jewelry', 'jewellery', 'necklace', 'ring', 'watch'],
            'rope': ['rope', 'cord', 'wire'],
            'hammer': ['hammer', 'mallet'],
            'crowbar': ['crowbar', 'pry bar'],
            'cash': ['cash', 'money', 'dollars'],
        }
        
        # Common location patterns
        self.location_patterns = {
            'apartment': ['apartment', 'flat', 'unit', 'condo'],
            'home': ['home', 'house', 'residence', 'dwelling'],
            'street': ['street', 'road', 'sidewalk'],
            'parking lot': ['parking lot', 'parking', 'garage', 'car park'],
            'alleyway': ['alleyway', 'alley', 'back alley'],
            'bar': ['bar', 'pub', 'tavern', 'nightclub', 'club'],
            'park': ['park', 'playground', 'garden'],
            'office': ['office', 'workplace', 'building'],
            'warehouse': ['warehouse', 'storage', 'factory'],
            'store': ['store', 'shop', 'market', 'mall'],
        }
    
    def _find_match(self, text: str, patterns: Dict[str, List[str]], known_set: set) -> Optional[str]:
        """Find a matching entity in text."""
        text_lower = text.lower()
        
        # First check exact matches with known entities
        for entity in known_set:
            if entity.lower() in text_lower:
                return entity
        
        # Then check patterns
        for canonical, variations in patterns.items():
            for var in variations:
                if var.lower() in text_lower:
                    # Return the canonical form if it's known
                    if canonical in known_set:
                        return canonical
                    # Otherwise return closest known entity
                    for known in known_set:
                        if canonical.lower() in known.lower() or known.lower() in canonical.lower():
                            return known
        
        return None
    
    def _extract_names(self, text: str) -> List[str]:
        """Extract potential person names from text."""
        # Simple heuristic: capitalized words that aren't at sentence start
        words = text.split()
        names = []
        
        for i, word in enumerate(words):
            # Clean word
            clean_word = re.sub(r'[^\w]', '', word)
            if clean_word and clean_word[0].isupper() and len(clean_word) > 1:
                # Skip common non-name capitalized words
                skip_words = {'The', 'A', 'An', 'He', 'She', 'They', 'It', 'This', 'That', 'At', 'In', 'On', 'When', 'Former', 'Latter'}
                if clean_word not in skip_words:
                    names.append(clean_word)
        
        # Handle special patterns like "the former" and "the latter"
        text_lower = text.lower()
        if 'former' in text_lower and 'latter' in text_lower:
            # Extract first two names mentioned
            if len(names) >= 2:
                # Keep only first two names (former and latter refer to them)
                names = names[:2]
        
        return names
    
    def _find_match(self, text: str, patterns: Dict[str, List[str]], known_set: set) -> Optional[str]:
        """Find a matching entity in text."""
        text_lower = text.lower()
        
        # Filter out common words that aren't entities
        skip_words = {'his', 'her', 'their', 'the', 'a', 'an', 'with', 'at', 'in', 'on', 'by', 'from'}
        
        # First check exact matches with known entities
        for entity in known_set:
            entity_lower = entity.lower()
            # Skip if entity is a skip word
            if entity_lower in skip_words:
                continue
            # Look for word boundaries
            if f" {entity_lower} " in f" {text_lower} " or text_lower.startswith(entity_lower + " ") or text_lower.endswith(" " + entity_lower):
                return entity
        
        # Then check patterns with word boundaries
        for canonical, variations in patterns.items():
            for var in variations:
                var_lower = var.lower()
                if var_lower in skip_words:
                    continue
                # Check with word boundaries to avoid partial matches
                if f" {var_lower} " in f" {text_lower} " or text_lower.startswith(var_lower + " ") or text_lower.endswith(" " + var_lower) or var_lower == text_lower:
                    # Return the canonical form if it's known
                    if canonical in known_set:
                        return canonical
                    # Otherwise return closest known entity
                    for known in known_set:
                        if canonical.lower() in known.lower() or known.lower() in canonical.lower():
                            return known
                    return canonical
        
        return None
    
    def _extract_names(self, text: str) -> List[str]:
        """Extract potential person names from text."""
        words = text.split()
        names = []
        
        for i, word in enumerate(words):
            clean_word = re.sub(r'[^\w]', '', word)
            if clean_word and clean_word[0].isupper() and len(clean_word) > 1:
                skip_words = {'The', 'A', 'An', 'He', 'She', 'They', 'It', 'This', 'That', 
                             'At', 'In', 'On', 'When', 'Former', 'Latter'}
                if clean_word not in skip_words:
                    names.append(clean_word)
        
        # Handle special patterns like "the former" and "the latter"
        text_lower = text.lower()
        if 'former' in text_lower and 'latter' in text_lower:
            if len(names) >= 2:
                names = names[:2]
        
        return names
    
    def parse(self, sentence: str) -> Optional[pd.DataFrame]:
        """
        Parse a crime description sentence.
        Uses NER if available, otherwise falls back to heuristics.
        """
        if self.use_ner:
            try:
                return self.ner_extractor.parse_sentence(sentence, self.default_crime)
            except Exception as e:
                print(f"NER parsing failed: {e}. Using heuristic fallback.")
                return self._heuristic_parse(sentence)
        else:
            return self._heuristic_parse(sentence)
    
    def _heuristic_parse(self, sentence: str) -> Optional[pd.DataFrame]:
        """Fallback heuristic parser (original implementation)."""
        # Extract entities
        names = self._extract_names(sentence)
        action = self._find_match(sentence, self.action_patterns, self.known_actions)
        obj = self._find_match(sentence, self.object_patterns, self.known_objects)
        location = self._find_match(sentence, self.location_patterns, self.known_locations)
        
        # Need at least an action to proceed
        if not action:
            return None
        
        # Assign names to suspect/victim
        suspect = names[0] if len(names) > 0 else "Unknown Suspect"
        victim = names[1] if len(names) > 1 else "Unknown Victim"
        
        # Handle object - ensure it's in known set
        if not obj or obj not in self.known_objects:
            if obj:
                # Try to map unknown object
                mapped_obj = self._map_to_known(obj, self.known_objects, self.object_patterns)
                obj = mapped_obj if mapped_obj in self.known_objects else sorted(list(self.known_objects))[0]
            else:
                obj = sorted(list(self.known_objects))[0] if self.known_objects else "knife"
        
        # Handle location - ensure it's in known set  
        if not location or location not in self.known_locations:
            if location:
                # Try to map unknown location
                mapped_loc = self._map_to_known(location, self.known_locations, self.location_patterns)
                location = mapped_loc if mapped_loc in self.known_locations else sorted(list(self.known_locations))[0]
            else:
                location = sorted(list(self.known_locations))[0] if self.known_locations else "home"
        
        # Handle action - ensure it's in known set
        if action and action not in self.known_actions:
            mapped_action = self._map_to_known(action, self.known_actions, self.action_patterns)
            action = mapped_action if mapped_action in self.known_actions else sorted(list(self.known_actions))[0]
        
        # Final validation - absolutely ensure all entities are known
        if obj not in self.known_objects:
            obj = sorted(list(self.known_objects))[0]
        if location not in self.known_locations:
            location = sorted(list(self.known_locations))[0]
        if action not in self.known_actions:
            action = sorted(list(self.known_actions))[0]
        
        # Use first known crime type as placeholder (will be predicted anyway)
        known_crimes = list(self.graph_builder.crime_encoder.classes_)
        default_crime = known_crimes[0] if known_crimes else "homicide"
        
        # Create DataFrame
        data = {
            'scene_id': [1],
            'event_id': [1],
            'crime_type': [default_crime],
            'suspect': [suspect],
            'victim': [victim],
            'object': [obj],
            'location': [location],
            'action': [action]
        }
        
        return pd.DataFrame(data)
    
    def _map_to_known(self, entity: str, known_set: set, patterns: Dict = None) -> str:
        """Map an unknown entity to closest known entity."""
        entity_lower = entity.lower()
        
        # Try exact match first
        for known in known_set:
            if entity_lower == known.lower():
                return known
        
        # Try substring match
        for known in known_set:
            if entity_lower in known.lower() or known.lower() in entity_lower:
                return known
        
        # Try pattern matching if provided
        if patterns:
            for canonical, variations in patterns.items():
                for var in variations:
                    if var in entity_lower or entity_lower in var:
                        if canonical in known_set:
                            return canonical
                        # Find closest known to canonical
                        for known in known_set:
                            if canonical.lower() in known.lower():
                                return known
        
        # Last resort: return first known entity (sorted for consistency)
        return sorted(list(known_set))[0] if known_set else entity
        # Extract entities
        names = self._extract_names(sentence)
        action = self._find_match(sentence, self.action_patterns, self.known_actions)
        obj = self._find_match(sentence, self.object_patterns, self.known_objects)
        location = self._find_match(sentence, self.location_patterns, self.known_locations)
        
        # Need at least an action to proceed
        if not action:
            return None
        
        # Assign names to suspect/victim
        suspect = names[0] if len(names) > 0 else "Unknown Suspect"
        victim = names[1] if len(names) > 1 else "Unknown Victim"
        
        # Use defaults for missing entities
        if not obj:
            # Use first known object as placeholder
            obj = sorted(list(self.known_objects))[0] if self.known_objects else "knife"
        if not location:
            # Use first known location as placeholder
            location = sorted(list(self.known_locations))[0] if self.known_locations else "street"
        
        # Use first known crime type as placeholder (will be predicted anyway)
        known_crimes = list(self.graph_builder.crime_encoder.classes_)
        default_crime = known_crimes[0] if known_crimes else "homicide"
        
        # Create DataFrame
        data = {
            'scene_id': [1],
            'event_id': [1],
            'crime_type': [default_crime],  # Placeholder - will be predicted
            'suspect': [suspect],
            'victim': [victim],
            'object': [obj],
            'location': [location],
            'action': [action]
        }
        
        return pd.DataFrame(data)
    
    def parse_multiple(self, sentences: List[str]) -> Optional[pd.DataFrame]:
        """Parse multiple sentences as events in a single scene."""
        events = []
        
        for i, sentence in enumerate(sentences):
            parsed = self.parse(sentence)
            if parsed is not None:
                parsed['event_id'] = i + 1
                events.append(parsed)
        
        if not events:
            return None
        
        result = pd.concat(events, ignore_index=True)
        result['scene_id'] = 1
        return result


# ============================================================
# STREAMLIT APP
# ============================================================

def load_model(model_path: str, builder_path: str = None):
    """Load trained model and graph builder."""
    # For this example, we'll train a new model if not exists
    # In production, you'd load from saved files
    pass


def create_graph_figure(scene_df: pd.DataFrame, prediction: str, confidence: float):
    """Create a NetworkX graph visualization."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    G = nx.DiGraph()
    
    # Colors
    node_colors = {
        'person': '#FF6B6B',
        'object': '#4ECDC4', 
        'location': '#95E1D3'
    }
    
    # Add nodes
    persons = list(pd.concat([scene_df['suspect'], scene_df['victim']]).unique())
    objects = list(scene_df['object'].unique())
    locations = list(scene_df['location'].unique())
    
    for p in persons:
        G.add_node(p, node_type='person')
    for o in objects:
        G.add_node(f"🔧 {o}", node_type='object')
    for l in locations:
        G.add_node(f"📍 {l}", node_type='location')
    
    # Add edges
    for _, event in scene_df.iterrows():
        suspect = event['suspect']
        victim = event['victim']
        obj = f"🔧 {event['object']}"
        loc = f"📍 {event['location']}"
        action = event['action']
        
        G.add_edge(suspect, victim, label=action)
        G.add_edge(suspect, obj, label='uses')
        G.add_edge(suspect, loc, label='at')
    
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Draw nodes by type
    for node_type, color in node_colors.items():
        nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == node_type]
        if nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=color,
                                   node_size=2500, alpha=0.9, ax=ax)
    
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color='gray', width=2, 
                           arrows=True, arrowsize=25, ax=ax,
                           connectionstyle="arc3,rad=0.1")
    
    # Edge labels
    edge_labels = {(u, v): d['label'] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9, ax=ax)
    
    ax.set_title(f"Crime Scene Graph\nPredicted: {prediction} ({confidence*100:.1f}%)", 
                fontsize=14, fontweight='bold')
    ax.axis('off')
    
    return fig


def create_prediction_chart(crime_probs: Dict[str, float], actual: str = None):
    """Create prediction probability chart."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    crimes = list(crime_probs.keys())
    probs = list(crime_probs.values())
    
    colors = ['#E74C3C' if (actual and c == actual) else '#3498DB' for c in crimes]
    
    bars = ax.barh(crimes, probs, color=colors, alpha=0.8)
    ax.set_xlabel('Probability', fontsize=12)
    ax.set_title('Crime Type Prediction', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1.1)
    
    for bar, prob in zip(bars, probs):
        ax.text(prob + 0.02, bar.get_y() + bar.get_height()/2,
               f'{prob*100:.1f}%', va='center', fontsize=11)
    
    plt.tight_layout()
    return fig


# ============================================================
# MAIN STREAMLIT APP
# ============================================================

def main():
    st.set_page_config(
        page_title="CrimeLens",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 CrimeLens: Crime Scene Analysis")
    st.markdown("*AI-powered crime scene reasoning using Graph Neural Networks*")
    
    # Sidebar
    st.sidebar.header("⚙️ Settings")
    
    # Initialize session state
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
        st.session_state.model = None
        st.session_state.graph_builder = None
        st.session_state.parser = None
    
    # Model loading section
    st.sidebar.subheader("Model")
    
    if st.sidebar.button("🚀 Load/Train Model"):
        with st.spinner("Loading model... This may take a minute."):
            try:
                # Check if pre-trained model exists
                if os.path.exists(MODEL_PATH):
                    st.sidebar.info("Loading pre-trained model...")
                    # Load existing model
                    # from realistic_data_generator import generate_realistic_data
                    # df = generate_realistic_data(n_scenes=100, seed=42)
                    from Architecture_2_generator import DataManager
                    dm = DataManager(DATA_PATH)
                    df = dm.getData()
                    
                    graph_builder = CrimeGraphBuilder()
                    graph_builder.fit(df)
                    
                    model = CrimeGNN(
                        n_actions=graph_builder.n_actions,
                        n_objects=graph_builder.n_objects,
                        n_locations=graph_builder.n_locations,
                        n_crimes=graph_builder.n_crimes,
                        embed_dim=32,
                        hidden_dim=128
                    )
                    model.load_state_dict(torch.load(MODEL_PATH))
                    model.eval()
                else:
                    st.sidebar.info("Training new model...")
                    # from realistic_data_generator import generate_realistic_data
                    # df = generate_realistic_data(n_scenes=200, seed=42)
                    from Architecture_2_generator import DataManager
                    dm = DataManager(DATA_PATH)
                    df = dm.getData()
                    results = run_complete_pipeline(df, save_dir="results", epochs=100)
                    model = results['model']
                    graph_builder = results['graph_builder']
                
                st.session_state.model = model
                st.session_state.graph_builder = graph_builder
                st.session_state.parser = ImprovedCrimeSentenceParser(graph_builder)
                st.session_state.model_loaded = True
                
                st.sidebar.success("✅ Model loaded!")
            except Exception as e:
                st.sidebar.error(f"Error: {str(e)}")
    
    if st.session_state.model_loaded:
        st.sidebar.success("✅ Model Ready")
        # Show parser type
        if hasattr(st.session_state, 'parser'):
            if st.session_state.parser.use_ner:
                st.sidebar.info("🔍 Using NER Parser")
            else:
                st.sidebar.info("📝 Using Heuristic Parser")
    else:
        st.sidebar.warning("⚠️ Please load model first")
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📝 Input", "🎯 Prediction", "🔮 Hypotheses", "📊 Graph View", "🔍 Explainer", "ℹ️ About"])
    
    # ==================== TAB 1: INPUT ====================
    with tab1:
        st.header("Crime Scene Input")
        
        input_mode = st.radio("Input Mode", ["Natural Language", "Structured Form", "CSV Upload"])
        
        if input_mode == "Natural Language":
            st.markdown("Enter a crime description in natural language:")
            
            example_sentences = [
                "John stabbed Maria with a knife at the apartment",
                "The suspect threatened the victim with a gun at the parking lot",
                "Mike robbed Sarah and took her wallet at the alley",
                "The attacker punched the victim at the bar and fled"
            ]
            
            st.markdown("**Examples:**")
            for ex in example_sentences:
                st.code(ex)
            
            user_input = st.text_area(
                "Enter crime description(s)",
                placeholder="Enter one sentence per line for multiple events...",
                height=150
            )
            
            if st.button("🔍 Parse Input", key="parse_btn"):
                if user_input and st.session_state.model_loaded:
                    sentences = [s.strip() for s in user_input.split('\n') if s.strip()]
                    parsed_df = st.session_state.parser.parse_multiple(sentences)
                    
                    if parsed_df is not None:
                        st.success("✅ Successfully parsed!")
                        st.dataframe(parsed_df)
                        st.session_state.current_scene = parsed_df
                    else:
                        st.error("❌ Could not parse input. Try using known actions like: stabbed, shot, robbed, attacked, threatened")
                elif not st.session_state.model_loaded:
                    st.warning("Please load the model first!")
        
        elif input_mode == "Structured Form":
            st.markdown("Enter crime details manually:")
            
            col1, col2 = st.columns(2)
            
            with col1:
                suspect = st.text_input("Suspect Name", value="John")
                victim = st.text_input("Victim Name", value="Maria")
                
            with col2:
                if st.session_state.model_loaded:
                    action = st.selectbox("Action", list(st.session_state.graph_builder.action_encoder.classes_))
                    obj = st.selectbox("Object/Weapon", list(st.session_state.graph_builder.object_encoder.classes_))
                    location = st.selectbox("Location", list(st.session_state.graph_builder.location_encoder.classes_))
                else:
                    action = st.text_input("Action", value="stabbed")
                    obj = st.text_input("Object/Weapon", value="knife")
                    location = st.text_input("Location", value="apartment")
            
            if st.button("📋 Create Scene", key="create_btn"):
                # Use a valid crime type as placeholder
                default_crime = list(st.session_state.graph_builder.crime_encoder.classes_)[0] if st.session_state.model_loaded else "homicide"
                
                data = {
                    'scene_id': [1],
                    'event_id': [1],
                    'crime_type': [default_crime],  # Placeholder
                    'suspect': [suspect],
                    'victim': [victim],
                    'object': [obj],
                    'location': [location],
                    'action': [action]
                }
                st.session_state.current_scene = pd.DataFrame(data)
                st.success("✅ Scene created!")
                st.dataframe(st.session_state.current_scene)
        
        else:  # CSV Upload
            uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
            if uploaded_file:
                df = pd.read_csv(uploaded_file)
                st.dataframe(df)
                st.session_state.current_scene = df
                st.success("✅ CSV loaded!")
    
    # ==================== TAB 2: PREDICTION ====================
    with tab2:
        st.header("Crime Type Prediction")
        
        if not st.session_state.model_loaded:
            st.warning("⚠️ Please load the model first (see sidebar)")
        elif 'current_scene' not in st.session_state:
            st.info("📝 Please enter a crime scene in the Input tab first")
        else:
            scene_df = st.session_state.current_scene
            
            st.subheader("Input Scene")
            st.info("ℹ️ Note: The 'crime_type' column below is just a placeholder. Scroll down to see the actual prediction.")
            st.dataframe(scene_df)
            
            if st.button("🎯 Predict Crime Type", key="predict_btn"):
                with st.spinner("Analyzing..."):
                    model = st.session_state.model
                    graph_builder = st.session_state.graph_builder
                    
                    model.eval()
                    with torch.no_grad():
                        graph = graph_builder.build_graph(scene_df)
                        output = model(graph)
                        crime_logits = output['crime_logits']
                        crime_probs = F.softmax(crime_logits, dim=1).squeeze().numpy()
                    
                    crime_types = graph_builder.crime_encoder.classes_
                    predictions = {ct: float(crime_probs[i]) for i, ct in enumerate(crime_types)}
                    
                    top_crime = crime_types[np.argmax(crime_probs)]
                    top_conf = float(crime_probs.max())
                    
                    # Store results
                    st.session_state.predictions = predictions
                    st.session_state.top_crime = top_crime
                    st.session_state.top_conf = top_conf
                    
                    # Display results
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.success(f"### Predicted: **{top_crime}** ({top_conf*100:.1f}% confidence)")
                    
                    with col2:
                        # Show if overconfident
                        if top_conf > 0.95:
                            st.warning("⚠️ Very high confidence - model may be overfit")
                        elif top_conf < 0.4:
                            st.info("ℹ️ Low confidence - ambiguous case")
                    
                    # Debug info
                    with st.expander("🔍 Debug Info"):
                        st.write("**Input Features:**")
                        st.write(f"- Action: {scene_df['action'].iloc[0]}")
                        st.write(f"- Object: {scene_df['object'].iloc[0]}")
                        st.write(f"- Location: {scene_df['location'].iloc[0]}")
                        st.write(f"- Suspect: {scene_df['suspect'].iloc[0]}")
                        st.write(f"- Victim: {scene_df['victim'].iloc[0]}")
                        
                        st.write("**Raw Logits:**")
                        logits_dict = {ct: float(crime_logits.squeeze()[i]) for i, ct in enumerate(crime_types)}
                        st.write(logits_dict)
                        
                        st.write("**Note:** The 'crime_type' in the input table is just a placeholder. The actual prediction is shown above.")
                    
                    # Prediction chart
                    fig = create_prediction_chart(predictions)
                    st.pyplot(fig)
                    
                    # Details
                    st.subheader("All Predictions")
                    pred_df = pd.DataFrame([
                        {'Crime Type': k, 'Probability': f"{v*100:.2f}%", 'Logit': f"{logits_dict[k]:.2f}"} 
                        for k, v in sorted(predictions.items(), key=lambda x: -x[1])
                    ])
                    st.dataframe(pred_df)
    
    # ==================== TAB 3: HYPOTHESES ====================
    with tab3:
        st.header("🔮 Alternative Hypotheses Generator")
        
        if not st.session_state.model_loaded:
            st.warning("⚠️ Please load the model first")
        elif 'current_scene' not in st.session_state:
            st.info("📝 Please enter a crime scene in the Input tab first")
        elif 'top_crime' not in st.session_state:
            st.info("🎯 Please run prediction first")
        else:
            scene_df = st.session_state.current_scene
            
            st.markdown("""
            Generate alternative hypotheses by testing different suspects, weapons, and locations.
            This helps explore "what-if" scenarios and alternative explanations.
            """)
            
            if st.button("🔮 Generate All Hypotheses", key="gen_hyp_btn"):
                with st.spinner("Analyzing alternatives..."):
                    try:
                        from HypothesesGenerator import AdvancedHypothesisGenerator, format_hypothesis_report
                        
                        model = st.session_state.model
                        graph_builder = st.session_state.graph_builder
                        
                        hyp_gen = AdvancedHypothesisGenerator(model, graph_builder)
                        report = hyp_gen.generate_comprehensive_report(scene_df)

                        # print("rohit  2")
                        
                        # Store report
                        st.session_state.hypothesis_report = report
                        
                        # Display formatted report
                        formatted = format_hypothesis_report(report)
                        st.code(formatted, language=None)
                        
                        # Interactive sections
                        st.subheader("📊 Interactive Analysis")
                        
                        # Alternative suspects visualization
                        if report['alternative_suspects']:
                            st.markdown("### Alternative Suspect Analysis")
                            alt_data = pd.DataFrame([
                                {
                                    'Suspect': alt['suspect'],
                                    'Predicted Crime': alt['predicted_crime'],
                                    'Confidence': alt['confidence']
                                }
                                for alt in report['alternative_suspects']
                            ])
                            
                            fig, ax = plt.subplots(figsize=(10, 5))
                            colors = plt.cm.RdYlGn(alt_data['Confidence'])
                            bars = ax.barh(alt_data['Suspect'], alt_data['Confidence'], color=colors)
                            ax.set_xlabel('Confidence')
                            ax.set_title('Alternative Suspect Likelihood')
                            ax.set_xlim(0, 1)
                            
                            for bar, conf in zip(bars, alt_data['Confidence']):
                                ax.text(conf + 0.02, bar.get_y() + bar.get_height()/2,
                                       f'{conf*100:.1f}%', va='center')
                            
                            st.pyplot(fig)
                        
                        # Weapon comparison
                        if report['weapon_scenarios']:
                            st.markdown("### Weapon/Object Scenario Comparison")
                            weapon_data = pd.DataFrame([
                                {
                                    'Object': wp['object'],
                                    'Predicted Crime': wp['predicted_crime'],
                                    'Confidence': wp['confidence']
                                }
                                for wp in report['weapon_scenarios']
                            ])
                            st.dataframe(weapon_data)
                        
                        # Download report
                        st.download_button(
                            label="📥 Download Full Report",
                            data=formatted,
                            file_name="hypothesis_report.txt",
                            mime="text/plain"
                        )
                        
                    except ImportError:
                        st.error("Hypothesis generator not found. Please ensure hypothesis_generator.py is in the same directory.")
                    except Exception as e:
                        st.error(f"Error generating hypotheses: {e}")
            
            # Quick hypothesis tests
            st.markdown("---")
            st.subheader("🎲 Quick Tests")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Test Different Suspects"):
                    try:
                        from HypothesesGenerator import AdvancedHypothesisGenerator
                        hyp_gen = AdvancedHypothesisGenerator(st.session_state.model, st.session_state.graph_builder)
                        alts = hyp_gen.generate_alternative_suspects(scene_df)
                        
                        for alt in alts:
                            st.write(f"**{alt['suspect']}**: {alt['predicted_crime']} ({alt['confidence']*100:.0f}%)")
                    except Exception as e:
                        st.error(f"Error: {e}")
            
            with col2:
                if st.button("Test Different Weapons"):
                    try:
                        from HypothesesGenerator import AdvancedHypothesisGenerator
                        hyp_gen = AdvancedHypothesisGenerator(st.session_state.model, st.session_state.graph_builder)
                        weapons = hyp_gen.generate_weapon_hypotheses(scene_df)
                        
                        for wp in weapons:
                            st.write(f"**{wp['object']}**: {wp['predicted_crime']} ({wp['confidence']*100:.0f}%)")
                    except Exception as e:
                        st.error(f"Error: {e}")
    
    # ==================== TAB 4: GRAPH VIEW ====================
    with tab4:
        st.header("Crime Scene Graph")
        
        if not st.session_state.model_loaded:
            st.warning("⚠️ Please load the model first")
        elif 'current_scene' not in st.session_state:
            st.info("📝 Please enter a crime scene in the Input tab first")
        elif 'top_crime' not in st.session_state:
            st.info("🎯 Please run prediction first")
        else:
            scene_df = st.session_state.current_scene
            prediction = st.session_state.top_crime
            confidence = st.session_state.top_conf
            
            st.subheader("Scene Entities")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**👤 Persons**")
                persons = list(pd.concat([scene_df['suspect'], scene_df['victim']]).unique())
                for p in persons:
                    st.write(f"• {p}")
            
            with col2:
                st.markdown("**🔧 Objects**")
                for o in scene_df['object'].unique():
                    st.write(f"• {o}")
            
            with col3:
                st.markdown("**📍 Locations**")
                for l in scene_df['location'].unique():
                    st.write(f"• {l}")
            
            st.subheader("Graph Visualization")
            fig = create_graph_figure(scene_df, prediction, confidence)
            st.pyplot(fig)
            
            # Legend
            st.markdown("""
            **Legend:**
            - 🔴 Red nodes: Persons (suspect/victim)
            - 🔵 Teal nodes: Objects/Weapons
            - 🟢 Green nodes: Locations
            - Edges show relationships (actions, uses, at)
            """)
    
    # ==================== TAB 5: EXPLAINER ====================
        # ==================== TAB 5: EXPLAINER ====================
    with tab5:
        st.header("🔍 Model Explainability")

        if not st.session_state.model_loaded:
            st.warning("⚠️ Please load the model first")
        elif 'current_scene' not in st.session_state:
            st.info("📝 Please enter a crime scene first")
        elif 'top_crime' not in st.session_state or 'top_conf' not in st.session_state:
            st.info("🎯 Please run a prediction first (see Prediction tab)")
        else:
            from Explainer2 import explain_crime_prediction

            scene_df   = st.session_state.current_scene
            prediction = st.session_state.top_crime
            confidence = st.session_state.top_conf
            preds_dict = st.session_state.get("predictions", None)

            st.markdown("""
            Understand **why** the model made its prediction by seeing which
            entities (suspects, objects, locations) and actions are most central
            in the scene graph.
            """)

            if st.button("🔍 Explain Prediction", key="explain_btn"):
                with st.spinner("Building explanation view..."):
                    fig, expl_text = explain_crime_prediction(
                        scene_df,
                        prediction,
                        confidence,
                        preds_dict,
                    )

                st.subheader("Graph-Based Explanation")
                st.pyplot(fig)

                st.subheader("Narrative Summary")
                st.text(expl_text)

    # ==================== TAB 6: ABOUT ====================
    with tab6:
        st.header("About CrimeLens")
        
        st.markdown("""
        ### What is CrimeLens?
        
        CrimeLens is a research prototype that uses **Graph Neural Networks (GNNs)** to analyze 
        crime scene descriptions and predict crime types.
        
        ### How it works
        
        1. **Input Processing**: Natural language descriptions are parsed to extract entities 
           (persons, objects, locations) and actions.
        
        2. **Graph Construction**: A graph is built where events are nodes and relationships 
           are captured through node features and edges.
        
        3. **GNN Inference**: A trained Graph Convolutional Network analyzes the graph structure 
           and predicts the most likely crime type.
        
        ### Model Architecture
        
        - **Graph Builder**: Events as nodes with action/object/location embeddings
        - **GNN**: 3-layer GCN with residual connections
        - **Features**: Learned embeddings + one-hot action encoding
        - **Output**: Crime type classification (homicide, robbery, assault, burglary, kidnapping)
        
        ### Technologies Used
        
        - PyTorch & PyTorch Geometric
        - NetworkX for graph visualization
        - Streamlit for UI
        - scikit-learn for baselines
        
        ### Disclaimer
        
        This is a research prototype for educational purposes. It should not be used for 
        actual criminal investigations.
        """)
        
        st.markdown("---")

if __name__ == "__main__":
    main()