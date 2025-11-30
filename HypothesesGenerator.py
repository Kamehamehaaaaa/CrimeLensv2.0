"""
Advanced Hypothesis Generator for CrimeLens
Generates multiple plausible scenarios and alternative suspects
"""

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import itertools


class AdvancedHypothesisGenerator:
    """
    Generate multiple crime hypotheses by:
    1. Testing alternative suspects
    2. Testing alternative objects/weapons
    3. Testing alternative locations
    4. Generating "what-if" scenarios
    """
    
    def __init__(self, model, graph_builder):
        self.model = model
        self.graph_builder = graph_builder
        
    @torch.no_grad()
    def generate_base_hypothesis(self, scene_df: pd.DataFrame) -> Dict:
        """Generate base prediction for the original scene."""
        self.model.eval()
        graph = self.graph_builder.build_graph(scene_df)
        output = self.model(graph)
        
        # Fix: Use .detach().numpy() instead of .numpy()
        crime_probs = F.softmax(output['crime_logits'], dim=1).squeeze().detach().numpy()
        
        crime_types = self.graph_builder.crime_encoder.classes_
        
        # Get persons from scene
        persons = list(pd.concat([scene_df['suspect'], scene_df['victim']]).unique())
        
        # Simple heuristic: original suspect gets highest score, others get lower scores
        suspect_scores = {}
        original_suspect = scene_df['suspect'].iloc[0]
        
        for i, person in enumerate(persons):
            if person == original_suspect:
                suspect_scores[person] = 0.85  # High confidence for original suspect
            else:
                suspect_scores[person] = 0.15 / max(len(persons) - 1, 1)  # Distribute remaining
        
        return {
            'crime_predictions': {ct: float(crime_probs[i]) for i, ct in enumerate(crime_types)},
            'suspect_scores': suspect_scores,
            'top_crime': crime_types[np.argmax(crime_probs)],
            'confidence': float(crime_probs.max()),
            'graph': graph,
            'original_scene': scene_df.copy()
        }
    
    @torch.no_grad()
    def generate_alternative_suspects(self, scene_df: pd.DataFrame, 
                                      num_alternatives: int = 3) -> List[Dict]:
        """
        Generate hypotheses with different suspects.
        Tests each person in the scene as the potential suspect and compares predictions.
        """
        self.model.eval()
        alternatives = []
        original_suspect = scene_df['suspect'].iloc[0]
        original_victim = scene_df['victim'].iloc[0]
        
        # Get all unique persons in the scene (both suspects and victims)
        all_persons = list(pd.concat([scene_df['suspect'], scene_df['victim']]).unique())
        
        # Get baseline prediction with original suspect
        baseline_graph = self.graph_builder.build_graph(scene_df)
        baseline_output = self.model(baseline_graph)
        # Fix: Add .detach()
        baseline_probs = F.softmax(baseline_output['crime_logits'], dim=1).squeeze().detach().numpy()
        
        # Generate alternatives by swapping suspect
        for alt_suspect in all_persons:
            if alt_suspect == original_suspect:
                continue
            
            # Create modified scene with this person as suspect
            alt_scene = scene_df.copy()
            alt_scene['suspect'] = alt_suspect
            
            # If this person was the victim, swap them
            if alt_suspect == original_victim:
                alt_scene['victim'] = original_suspect
            
            try:
                # Get prediction for this alternative
                graph = self.graph_builder.build_graph(alt_scene)
                output = self.model(graph)
                # Fix: Add .detach()
                crime_probs = F.softmax(output['crime_logits'], dim=1).squeeze().detach().numpy()
                
                crime_types = self.graph_builder.crime_encoder.classes_
                top_crime = crime_types[np.argmax(crime_probs)]
                confidence = float(crime_probs.max())
                
                # Calculate how different this prediction is from baseline
                prob_diff = np.abs(crime_probs - baseline_probs).mean()
                
                alternatives.append({
                    'suspect': alt_suspect,
                    'predicted_crime': top_crime,
                    'confidence': confidence,
                    'difference_from_original': prob_diff,
                    'crime_probs': {ct: float(crime_probs[i]) for i, ct in enumerate(crime_types)},
                    'scenario': f"If {alt_suspect} was the perpetrator"
                })
            except Exception as e:
                print(f"Error testing {alt_suspect}: {e}")
                continue
        
        # Sort by how different the prediction is (more interesting alternatives first)
        alternatives.sort(key=lambda x: x['difference_from_original'], reverse=True)
        return alternatives[:num_alternatives]
    
    @torch.no_grad()
    def generate_weapon_hypotheses(self, scene_df: pd.DataFrame) -> List[Dict]:
        """
        Generate hypotheses with different weapons/objects.
        Shows how crime type prediction changes with different weapons.
        """
        self.model.eval()
        alternatives = []
        original_object = scene_df['object'].iloc[0]
        
        # Test different weapon types
        test_objects = ['knife', 'gun', 'bat', 'rope', 'poison']
        available_objects = self.graph_builder.object_encoder.classes_
        test_objects = [obj for obj in test_objects if obj in available_objects]
        
        for alt_object in test_objects:
            if alt_object == original_object:
                continue
            
            # Create modified scene
            alt_scene = scene_df.copy()
            alt_scene['object'] = alt_object
            
            try:
                graph = self.graph_builder.build_graph(alt_scene)
                output = self.model(graph)
                # Fix: Add .detach()
                crime_probs = F.softmax(output['crime_logits'], dim=1).squeeze().detach().numpy()
                
                crime_types = self.graph_builder.crime_encoder.classes_
                top_crime = crime_types[np.argmax(crime_probs)]
                confidence = float(crime_probs.max())
                
                alternatives.append({
                    'object': alt_object,
                    'predicted_crime': top_crime,
                    'confidence': confidence,
                    'crime_probs': {ct: float(crime_probs[i]) for i, ct in enumerate(crime_types)},
                    'scenario': f"If {alt_object} was used instead of {original_object}"
                })
            except Exception as e:
                print(f"Error testing {alt_object}: {e}")
                continue
        
        alternatives.sort(key=lambda x: x['confidence'], reverse=True)
        return alternatives
    
    @torch.no_grad()
    def generate_location_hypotheses(self, scene_df: pd.DataFrame) -> List[Dict]:
        """
        Generate hypotheses for different locations.
        Shows how location affects crime type prediction.
        """
        self.model.eval()
        alternatives = []
        original_location = scene_df['location'].iloc[0]
        
        # Test different location types
        test_locations = ['home', 'street', 'park', 'bar', 'parking lot']
        available_locations = self.graph_builder.location_encoder.classes_
        test_locations = [loc for loc in test_locations if loc in available_locations]
        
        for alt_location in test_locations:
            if alt_location == original_location:
                continue
            
            alt_scene = scene_df.copy()
            alt_scene['location'] = alt_location
            
            try:
                graph = self.graph_builder.build_graph(alt_scene)
                output = self.model(graph)
                # Fix: Add .detach()
                crime_probs = F.softmax(output['crime_logits'], dim=1).squeeze().detach().numpy()
                
                crime_types = self.graph_builder.crime_encoder.classes_
                top_crime = crime_types[np.argmax(crime_probs)]
                confidence = float(crime_probs.max())
                
                alternatives.append({
                    'location': alt_location,
                    'predicted_crime': top_crime,
                    'confidence': confidence,
                    'scenario': f"If crime occurred at {alt_location} instead of {original_location}"
                })
            except Exception as e:
                print(f"Error testing {alt_location}: {e}")
                continue
        
        return alternatives
    
    @torch.no_grad()
    def generate_counterfactuals(self, scene_df: pd.DataFrame) -> List[Dict]:
        """
        Generate "what-if" counterfactual scenarios.
        Example: "What if the weapon was different AND location changed?"
        """
        self.model.eval()
        counterfactuals = []
        
        # Scenario 1: No weapon
        if 'weapon' not in scene_df['object'].iloc[0].lower():
            alt_scene = scene_df.copy()
            # Try to use 'fist' or first available object
            available_objects = self.graph_builder.object_encoder.classes_
            fallback_obj = 'fist' if 'fist' in available_objects else available_objects[0]
            alt_scene['object'] = fallback_obj
            
            try:
                graph = self.graph_builder.build_graph(alt_scene)
                output = self.model(graph)
                # Fix: Add .detach()
                crime_probs = F.softmax(output['crime_logits'], dim=1).squeeze().detach().numpy()
                crime_types = self.graph_builder.crime_encoder.classes_
                
                counterfactuals.append({
                    'scenario': 'Unarmed assault (no weapon)',
                    'predicted_crime': crime_types[np.argmax(crime_probs)],
                    'confidence': float(crime_probs.max()),
                    'description': 'If the perpetrator had not used a weapon'
                })
            except Exception as e:
                print(f"Error in counterfactual 1: {e}")
        
        # Scenario 2: Public vs Private location
        original_loc = scene_df['location'].iloc[0]
        public_locs = ['street', 'park', 'parking lot']
        private_locs = ['home', 'apartment', 'bedroom']
        
        is_public = any(loc in original_loc.lower() for loc in public_locs)
        test_loc = 'home' if is_public else 'street'
        
        if test_loc in self.graph_builder.location_encoder.classes_:
            alt_scene = scene_df.copy()
            alt_scene['location'] = test_loc
            try:
                graph = self.graph_builder.build_graph(alt_scene)
                output = self.model(graph)
                # Fix: Add .detach()
                crime_probs = F.softmax(output['crime_logits'], dim=1).squeeze().detach().numpy()
                crime_types = self.graph_builder.crime_encoder.classes_
                
                counterfactuals.append({
                    'scenario': f'{"Public" if not is_public else "Private"} location scenario',
                    'predicted_crime': crime_types[np.argmax(crime_probs)],
                    'confidence': float(crime_probs.max()),
                    'description': f'If crime occurred in a {"public" if not is_public else "private"} place'
                })
            except Exception as e:
                print(f"Error in counterfactual 2: {e}")
        
        return counterfactuals
    
    def generate_comprehensive_report(self, scene_df: pd.DataFrame) -> Dict:
        """
        Generate a comprehensive hypothesis report with all analyses.
        """
        print("Generating hypotheses...")
        
        # Base prediction
        base = self.generate_base_hypothesis(scene_df)
        
        # Alternative suspects
        alt_suspects = self.generate_alternative_suspects(scene_df)
        
        # Weapon hypotheses
        weapon_hyp = self.generate_weapon_hypotheses(scene_df)
        
        # Location hypotheses
        location_hyp = self.generate_location_hypotheses(scene_df)
        
        # Counterfactuals
        counterfactuals = self.generate_counterfactuals(scene_df)
        
        return {
            'base_prediction': base,
            'alternative_suspects': alt_suspects,
            'weapon_scenarios': weapon_hyp,
            'location_scenarios': location_hyp,
            'counterfactuals': counterfactuals,
            'original_scene': scene_df
        }


def format_hypothesis_report(report: Dict) -> str:
    """Format hypothesis report as human-readable text."""
    base = report['base_prediction']
    
    text = f"""
CRIME HYPOTHESIS REPORT
{'='*70}

ORIGINAL SCENE ANALYSIS
Predicted Crime Type: {base['top_crime']} ({base['confidence']*100:.1f}% confidence)

Top Crime Probabilities:
"""
    
    # Sort and show top 3 crime predictions
    sorted_crimes = sorted(base['crime_predictions'].items(), key=lambda x: -x[1])[:3]
    for crime, prob in sorted_crimes:
        bar = "█" * int(prob * 20)
        text += f"  {crime:20s} {prob*100:5.1f}% {bar}\n"
    
    # Alternative suspects
    if report['alternative_suspects']:
        text += f"\n{'='*70}\n"
        text += "ALTERNATIVE SUSPECT HYPOTHESES\n\n"
        for i, alt in enumerate(report['alternative_suspects'], 1):
            text += f"{i}. {alt['scenario']}\n"
            text += f"   → Predicted: {alt['predicted_crime']} ({alt['confidence']*100:.1f}% confidence)\n\n"
    
    # Weapon scenarios
    if report['weapon_scenarios']:
        text += f"{'='*70}\n"
        text += "WEAPON/OBJECT ANALYSIS\n\n"
        for i, wp in enumerate(report['weapon_scenarios'][:3], 1):
            text += f"{i}. {wp['scenario']}\n"
            text += f"   → Predicted: {wp['predicted_crime']} ({wp['confidence']*100:.1f}% confidence)\n\n"
    
    # Counterfactuals
    if report['counterfactuals']:
        text += f"{'='*70}\n"
        text += "WHAT-IF SCENARIOS\n\n"
        for i, cf in enumerate(report['counterfactuals'], 1):
            text += f"{i}. {cf['scenario']}\n"
            text += f"   {cf['description']}\n"
            text += f"   → Predicted: {cf['predicted_crime']} ({cf['confidence']*100:.1f}% confidence)\n\n"
    
    return text

# """
# Advanced Hypothesis Generator for CrimeLens
# Generates multiple plausible scenarios and alternative suspects
# """

# import torch
# import torch.nn.functional as F
# import pandas as pd
# import numpy as np
# from typing import List, Dict, Tuple, Optional
# import itertools


# class AdvancedHypothesisGenerator:
#     """
#     Generate multiple crime hypotheses by:
#     1. Testing alternative suspects
#     2. Testing alternative objects/weapons
#     3. Testing alternative locations
#     4. Generating "what-if" scenarios
#     """
    
#     def __init__(self, model, graph_builder):
#         self.model = model
#         self.graph_builder = graph_builder
        
#     @torch.no_grad()
#     def generate_base_hypothesis(self, scene_df: pd.DataFrame) -> Dict:
#         """Generate base prediction for the original scene."""
#         self.model.eval()
#         graph = self.graph_builder.build_graph(scene_df)
#         output = self.model(graph)
        
#         crime_probs = F.softmax(output['crime_logits'], dim=1).squeeze().numpy()
#         crime_types = self.graph_builder.crime_encoder.classes_
        
#         # For suspect scores, we need to create a heuristic since the model only outputs one score
#         # We'll rank suspects based on their role and graph properties
#         persons = list(pd.concat([scene_df['suspect'], scene_df['victim']]).unique())
        
#         # Simple heuristic: original suspect gets highest score, others get lower scores
#         suspect_scores = {}
#         original_suspect = scene_df['suspect'].iloc[0]
        
#         for i, person in enumerate(persons):
#             if person == original_suspect:
#                 suspect_scores[person] = 0.85  # High confidence for original suspect
#             else:
#                 suspect_scores[person] = 0.15 / (len(persons) - 1)  # Distribute remaining among others
        
#         return {
#             'crime_predictions': {ct: float(crime_probs[i]) for i, ct in enumerate(crime_types)},
#             'suspect_scores': suspect_scores,
#             'top_crime': crime_types[np.argmax(crime_probs)],
#             'confidence': float(crime_probs.max()),
#             'graph': graph,
#             'original_scene': scene_df.copy()
#         }
    
#     def generate_alternative_suspects(self, scene_df: pd.DataFrame, 
#                                       num_alternatives: int = 3) -> List[Dict]:
#         """
#         Generate hypotheses with different suspects.
        
#         Tests each person in the scene as the potential suspect and compares predictions.
#         """
#         alternatives = []
#         original_suspect = scene_df['suspect'].iloc[0]
#         original_victim = scene_df['victim'].iloc[0]
        
#         # Get all unique persons in the scene (both suspects and victims)
#         all_persons = list(pd.concat([scene_df['suspect'], scene_df['victim']]).unique())
        
#         # Get baseline prediction with original suspect
#         baseline_graph = self.graph_builder.build_graph(scene_df)
#         baseline_output = self.model(baseline_graph)
#         baseline_probs = F.softmax(baseline_output['crime_logits'], dim=1).squeeze().numpy()
        
#         # Generate alternatives by swapping suspect
#         for alt_suspect in all_persons:
#             if alt_suspect == original_suspect:
#                 continue
            
#             # Create modified scene with this person as suspect
#             alt_scene = scene_df.copy()
#             alt_scene['suspect'] = alt_suspect
            
#             # If this person was the victim, swap them
#             if alt_suspect == original_victim:
#                 alt_scene['victim'] = original_suspect
            
#             try:
#                 # Get prediction for this alternative
#                 graph = self.graph_builder.build_graph(alt_scene)
#                 output = self.model(graph)
#                 crime_probs = F.softmax(output['crime_logits'], dim=1).squeeze().numpy()
                
#                 crime_types = self.graph_builder.crime_encoder.classes_
#                 top_crime = crime_types[np.argmax(crime_probs)]
#                 confidence = float(crime_probs.max())
                
#                 # Calculate how different this prediction is from baseline
#                 prob_diff = np.abs(crime_probs - baseline_probs).mean()
                
#                 alternatives.append({
#                     'suspect': alt_suspect,
#                     'predicted_crime': top_crime,
#                     'confidence': confidence,
#                     'difference_from_original': prob_diff,
#                     'crime_probs': {ct: float(crime_probs[i]) for i, ct in enumerate(crime_types)},
#                     'scenario': f"If {alt_suspect} was the perpetrator"
#                 })
#             except Exception as e:
#                 print(f"Error testing {alt_suspect}: {e}")
#                 continue
        
#         # Sort by how different the prediction is (more interesting alternatives first)
#         alternatives.sort(key=lambda x: x['difference_from_original'], reverse=True)
#         return alternatives[:num_alternatives]
    
#     def generate_weapon_hypotheses(self, scene_df: pd.DataFrame) -> List[Dict]:
#         """
#         Generate hypotheses with different weapons/objects.
        
#         Shows how crime type prediction changes with different weapons.
#         """
#         alternatives = []
#         original_object = scene_df['object'].iloc[0]
        
#         # Test different weapon types
#         test_objects = ['knife', 'gun', 'bat', 'rope', 'poison']
#         available_objects = self.graph_builder.object_encoder.classes_
#         test_objects = [obj for obj in test_objects if obj in available_objects]
        
#         for alt_object in test_objects:
#             if alt_object == original_object:
#                 continue
            
#             # Create modified scene
#             alt_scene = scene_df.copy()
#             alt_scene['object'] = alt_object
            
#             try:
#                 graph = self.graph_builder.build_graph(alt_scene)
#                 output = self.model(graph)
#                 crime_probs = F.softmax(output['crime_logits'], dim=1).squeeze().numpy()
                
#                 crime_types = self.graph_builder.crime_encoder.classes_
#                 top_crime = crime_types[np.argmax(crime_probs)]
#                 confidence = float(crime_probs.max())
                
#                 alternatives.append({
#                     'object': alt_object,
#                     'predicted_crime': top_crime,
#                     'confidence': confidence,
#                     'crime_probs': {ct: float(crime_probs[i]) for i, ct in enumerate(crime_types)},
#                     'scenario': f"If {alt_object} was used instead of {original_object}"
#                 })
#             except:
#                 continue
        
#         alternatives.sort(key=lambda x: x['confidence'], reverse=True)
#         return alternatives
    
#     def generate_location_hypotheses(self, scene_df: pd.DataFrame) -> List[Dict]:
#         """
#         Generate hypotheses for different locations.
        
#         Shows how location affects crime type prediction.
#         """
#         alternatives = []
#         original_location = scene_df['location'].iloc[0]
        
#         # Test different location types
#         test_locations = ['home', 'street', 'park', 'bar', 'parking lot']
#         available_locations = self.graph_builder.location_encoder.classes_
#         test_locations = [loc for loc in test_locations if loc in available_locations]
        
#         for alt_location in test_locations:
#             if alt_location == original_location:
#                 continue
            
#             alt_scene = scene_df.copy()
#             alt_scene['location'] = alt_location
            
#             try:
#                 graph = self.graph_builder.build_graph(alt_scene)
#                 output = self.model(graph)
#                 crime_probs = F.softmax(output['crime_logits'], dim=1).squeeze().numpy()
                
#                 crime_types = self.graph_builder.crime_encoder.classes_
#                 top_crime = crime_types[np.argmax(crime_probs)]
#                 confidence = float(crime_probs.max())
                
#                 alternatives.append({
#                     'location': alt_location,
#                     'predicted_crime': top_crime,
#                     'confidence': confidence,
#                     'scenario': f"If crime occurred at {alt_location} instead of {original_location}"
#                 })
#             except:
#                 continue
        
#         return alternatives
    
#     def generate_counterfactuals(self, scene_df: pd.DataFrame) -> List[Dict]:
#         """
#         Generate "what-if" counterfactual scenarios.
        
#         Example: "What if the weapon was different AND location changed?"
#         """
#         counterfactuals = []
        
#         # Scenario 1: No weapon
#         if 'weapon' not in scene_df['object'].iloc[0].lower():
#             alt_scene = scene_df.copy()
#             alt_scene['object'] = 'fist'  # Unarmed
#             try:
#                 graph = self.graph_builder.build_graph(alt_scene)
#                 output = self.model(graph)
#                 crime_probs = F.softmax(output['crime_logits'], dim=1).squeeze().numpy()
#                 crime_types = self.graph_builder.crime_encoder.classes_
                
#                 counterfactuals.append({
#                     'scenario': 'Unarmed assault (no weapon)',
#                     'predicted_crime': crime_types[np.argmax(crime_probs)],
#                     'confidence': float(crime_probs.max()),
#                     'description': 'If the perpetrator had not used a weapon'
#                 })
#             except:
#                 pass
        
#         # Scenario 2: Public vs Private location
#         original_loc = scene_df['location'].iloc[0]
#         public_locs = ['street', 'park', 'parking lot']
#         private_locs = ['home', 'apartment', 'bedroom']
        
#         is_public = any(loc in original_loc.lower() for loc in public_locs)
#         test_loc = 'home' if is_public else 'street'
        
#         if test_loc in self.graph_builder.location_encoder.classes_:
#             alt_scene = scene_df.copy()
#             alt_scene['location'] = test_loc
#             try:
#                 graph = self.graph_builder.build_graph(alt_scene)
#                 output = self.model(graph)
#                 crime_probs = F.softmax(output['crime_logits'], dim=1).squeeze().numpy()
#                 crime_types = self.graph_builder.crime_encoder.classes_
                
#                 counterfactuals.append({
#                     'scenario': f'{"Public" if not is_public else "Private"} location scenario',
#                     'predicted_crime': crime_types[np.argmax(crime_probs)],
#                     'confidence': float(crime_probs.max()),
#                     'description': f'If crime occurred in a {"public" if not is_public else "private"} place'
#                 })
#             except:
#                 pass
        
#         return counterfactuals
    
#     def generate_comprehensive_report(self, scene_df: pd.DataFrame) -> Dict:
#         """
#         Generate a comprehensive hypothesis report with all analyses.
#         """
#         print("Generating hypotheses...")
        
#         # Base prediction
#         base = self.generate_base_hypothesis(scene_df)
        
#         # Alternative suspects
#         alt_suspects = self.generate_alternative_suspects(scene_df)
        
#         # Weapon hypotheses
#         weapon_hyp = self.generate_weapon_hypotheses(scene_df)
        
#         # Location hypotheses
#         location_hyp = self.generate_location_hypotheses(scene_df)
        
#         # Counterfactuals
#         counterfactuals = self.generate_counterfactuals(scene_df)
        
#         return {
#             'base_prediction': base,
#             'alternative_suspects': alt_suspects,
#             'weapon_scenarios': weapon_hyp,
#             'location_scenarios': location_hyp,
#             'counterfactuals': counterfactuals,
#             'original_scene': scene_df
#         }


# def format_hypothesis_report(report: Dict) -> str:
#     """Format hypothesis report as human-readable text."""
#     base = report['base_prediction']
    
#     text = f"""
# CRIME HYPOTHESIS REPORT
# {'='*70}

# ORIGINAL SCENE ANALYSIS
# Predicted Crime Type: {base['top_crime']} ({base['confidence']*100:.1f}% confidence)

# Top Crime Probabilities:
# """
    
#     # Sort and show top 3 crime predictions
#     sorted_crimes = sorted(base['crime_predictions'].items(), key=lambda x: -x[1])[:3]
#     for crime, prob in sorted_crimes:
#         bar = "█" * int(prob * 20)
#         text += f"  {crime:20s} {prob*100:5.1f}% {bar}\n"
    
#     # Alternative suspects
#     if report['alternative_suspects']:
#         text += f"\n{'='*70}\n"
#         text += "ALTERNATIVE SUSPECT HYPOTHESES\n\n"
#         for i, alt in enumerate(report['alternative_suspects'], 1):
#             text += f"{i}. {alt['scenario']}\n"
#             text += f"   → Predicted: {alt['predicted_crime']} ({alt['confidence']*100:.1f}% confidence)\n\n"
    
#     # Weapon scenarios
#     if report['weapon_scenarios']:
#         text += f"{'='*70}\n"
#         text += "WEAPON/OBJECT ANALYSIS\n\n"
#         for i, wp in enumerate(report['weapon_scenarios'][:3], 1):
#             text += f"{i}. {wp['scenario']}\n"
#             text += f"   → Predicted: {wp['predicted_crime']} ({wp['confidence']*100:.1f}% confidence)\n\n"
    
#     # Counterfactuals
#     if report['counterfactuals']:
#         text += f"{'='*70}\n"
#         text += "WHAT-IF SCENARIOS\n\n"
#         for i, cf in enumerate(report['counterfactuals'], 1):
#             text += f"{i}. {cf['scenario']}\n"
#             text += f"   {cf['description']}\n"
#             text += f"   → Predicted: {cf['predicted_crime']} ({cf['confidence']*100:.1f}% confidence)\n\n"
    
#     return text

# """
# Advanced Hypothesis Generator for CrimeLens
# Generates multiple plausible scenarios and alternative suspects
# """

# import torch
# import torch.nn.functional as F
# import pandas as pd
# import numpy as np
# from typing import List, Dict, Tuple, Optional
# import itertools


# class AdvancedHypothesisGenerator:
#     """
#     Generate multiple crime hypotheses by:
#     1. Testing alternative suspects
#     2. Testing alternative objects/weapons
#     3. Testing alternative locations
#     4. Generating "what-if" scenarios
#     """
    
#     def __init__(self, model, graph_builder):
#         self.model = model
#         self.graph_builder = graph_builder
        
#     @torch.no_grad()
#     def generate_base_hypothesis(self, scene_df: pd.DataFrame) -> Dict:
#         """Generate base prediction for the original scene."""
#         self.model.eval()
#         graph = self.graph_builder.build_graph(scene_df)
#         output = self.model(graph)
        
#         crime_probs = F.softmax(output['crime_logits'], dim=1).squeeze().numpy()
#         suspect_scores = torch.sigmoid(output['suspect_scores']).numpy()
        
#         crime_types = self.graph_builder.crime_encoder.classes_
#         persons = graph.metadata['persons']
#         print("rohit 5", crime_probs, crime_types, suspect_scores,persons)
        
#         return {
#             'crime_predictions': {ct: float(crime_probs[i]) for i, ct in enumerate(crime_types)},
#             'suspect_scores': {p: float(suspect_scores[i]) for i, p in enumerate(persons)},
#             'top_crime': crime_types[np.argmax(crime_probs)],
#             'confidence': float(crime_probs.max()),
#             'graph': graph,
#             'original_scene': scene_df.copy()
#         }
    
#     def generate_alternative_suspects(self, scene_df: pd.DataFrame, 
#                                       num_alternatives: int = 3) -> List[Dict]:
#         """
#         Generate hypotheses with different suspects.
        
#         Returns ranked list of alternative suspects with their predicted crimes.
#         """
#         alternatives = []
#         original_suspect = scene_df['suspect'].iloc[0]
        
#         # Get all unique persons in the scene
#         all_persons = list(pd.concat([scene_df['suspect'], scene_df['victim']]).unique())
        
#         # Generate alternatives
#         for alt_suspect in all_persons:
#             if alt_suspect == original_suspect:
#                 continue
            
#             # Create modified scene
#             alt_scene = scene_df.copy()
#             alt_scene['suspect'] = alt_suspect
            
#             try:
#                 # Get prediction
#                 graph = self.graph_builder.build_graph(alt_scene)
#                 output = self.model(graph)
#                 crime_probs = F.softmax(output['crime_logits'], dim=1).squeeze().numpy()
                
#                 crime_types = self.graph_builder.crime_encoder.classes_
#                 top_crime = crime_types[np.argmax(crime_probs)]
#                 confidence = float(crime_probs.max())
                
#                 alternatives.append({
#                     'suspect': alt_suspect,
#                     'predicted_crime': top_crime,
#                     'confidence': confidence,
#                     'crime_probs': {ct: float(crime_probs[i]) for i, ct in enumerate(crime_types)},
#                     'scenario': f"If {alt_suspect} was the perpetrator instead of {original_suspect}"
#                 })
#             except:
#                 continue
        
#         # Sort by confidence
#         alternatives.sort(key=lambda x: x['confidence'], reverse=True)
#         return alternatives[:num_alternatives]
    
#     def generate_weapon_hypotheses(self, scene_df: pd.DataFrame) -> List[Dict]:
#         """
#         Generate hypotheses with different weapons/objects.
        
#         Shows how crime type prediction changes with different weapons.
#         """
#         alternatives = []
#         original_object = scene_df['object'].iloc[0]
        
#         # Test different weapon types
#         test_objects = ['knife', 'gun', 'bat', 'rope', 'poison']
#         available_objects = self.graph_builder.object_encoder.classes_
#         test_objects = [obj for obj in test_objects if obj in available_objects]
        
#         for alt_object in test_objects:
#             if alt_object == original_object:
#                 continue
            
#             # Create modified scene
#             alt_scene = scene_df.copy()
#             alt_scene['object'] = alt_object
            
#             try:
#                 graph = self.graph_builder.build_graph(alt_scene)
#                 output = self.model(graph)
#                 crime_probs = F.softmax(output['crime_logits'], dim=1).squeeze().numpy()
                
#                 crime_types = self.graph_builder.crime_encoder.classes_
#                 top_crime = crime_types[np.argmax(crime_probs)]
#                 confidence = float(crime_probs.max())
                
#                 alternatives.append({
#                     'object': alt_object,
#                     'predicted_crime': top_crime,
#                     'confidence': confidence,
#                     'crime_probs': {ct: float(crime_probs[i]) for i, ct in enumerate(crime_types)},
#                     'scenario': f"If {alt_object} was used instead of {original_object}"
#                 })
#             except:
#                 continue
        
#         alternatives.sort(key=lambda x: x['confidence'], reverse=True)
#         return alternatives
    
#     def generate_location_hypotheses(self, scene_df: pd.DataFrame) -> List[Dict]:
#         """
#         Generate hypotheses for different locations.
        
#         Shows how location affects crime type prediction.
#         """
#         alternatives = []
#         original_location = scene_df['location'].iloc[0]
        
#         # Test different location types
#         test_locations = ['home', 'street', 'park', 'bar', 'parking lot']
#         available_locations = self.graph_builder.location_encoder.classes_
#         test_locations = [loc for loc in test_locations if loc in available_locations]
        
#         for alt_location in test_locations:
#             if alt_location == original_location:
#                 continue
            
#             alt_scene = scene_df.copy()
#             alt_scene['location'] = alt_location
            
#             try:
#                 graph = self.graph_builder.build_graph(alt_scene)
#                 output = self.model(graph)
#                 crime_probs = F.softmax(output['crime_logits'], dim=1).squeeze().numpy()
                
#                 crime_types = self.graph_builder.crime_encoder.classes_
#                 top_crime = crime_types[np.argmax(crime_probs)]
#                 confidence = float(crime_probs.max())
                
#                 alternatives.append({
#                     'location': alt_location,
#                     'predicted_crime': top_crime,
#                     'confidence': confidence,
#                     'scenario': f"If crime occurred at {alt_location} instead of {original_location}"
#                 })
#             except:
#                 continue
        
#         return alternatives
    
#     def generate_counterfactuals(self, scene_df: pd.DataFrame) -> List[Dict]:
#         """
#         Generate "what-if" counterfactual scenarios.
        
#         Example: "What if the weapon was different AND location changed?"
#         """
#         counterfactuals = []
        
#         # Scenario 1: No weapon
#         if 'weapon' not in scene_df['object'].iloc[0].lower():
#             alt_scene = scene_df.copy()
#             alt_scene['object'] = 'fist'  # Unarmed
#             try:
#                 graph = self.graph_builder.build_graph(alt_scene)
#                 output = self.model(graph)
#                 crime_probs = F.softmax(output['crime_logits'], dim=1).squeeze().numpy()
#                 crime_types = self.graph_builder.crime_encoder.classes_
                
#                 counterfactuals.append({
#                     'scenario': 'Unarmed assault (no weapon)',
#                     'predicted_crime': crime_types[np.argmax(crime_probs)],
#                     'confidence': float(crime_probs.max()),
#                     'description': 'If the perpetrator had not used a weapon'
#                 })
#             except:
#                 pass
        
#         # Scenario 2: Public vs Private location
#         original_loc = scene_df['location'].iloc[0]
#         public_locs = ['street', 'park', 'parking lot']
#         private_locs = ['home', 'apartment', 'bedroom']
        
#         is_public = any(loc in original_loc.lower() for loc in public_locs)
#         test_loc = 'home' if is_public else 'street'
        
#         if test_loc in self.graph_builder.location_encoder.classes_:
#             alt_scene = scene_df.copy()
#             alt_scene['location'] = test_loc
#             try:
#                 graph = self.graph_builder.build_graph(alt_scene)
#                 output = self.model(graph)
#                 crime_probs = F.softmax(output['crime_logits'], dim=1).squeeze().numpy()
#                 crime_types = self.graph_builder.crime_encoder.classes_
                
#                 counterfactuals.append({
#                     'scenario': f'{"Public" if not is_public else "Private"} location scenario',
#                     'predicted_crime': crime_types[np.argmax(crime_probs)],
#                     'confidence': float(crime_probs.max()),
#                     'description': f'If crime occurred in a {"public" if not is_public else "private"} place'
#                 })
#             except:
#                 pass
        
#         return counterfactuals
    
#     def generate_comprehensive_report(self, scene_df: pd.DataFrame) -> Dict:
#         """
#         Generate a comprehensive hypothesis report with all analyses.
#         """
#         print("Generating hypotheses...")
        
#         # Base prediction
#         base = self.generate_base_hypothesis(scene_df)
#         print("rohit  3")
        
#         # Alternative suspects
#         alt_suspects = self.generate_alternative_suspects(scene_df)
#         print("rohit  4")
        
#         # Weapon hypotheses
#         weapon_hyp = self.generate_weapon_hypotheses(scene_df)
#         print("rohit  5")

#         # Location hypotheses
#         location_hyp = self.generate_location_hypotheses(scene_df)
#         print("rohit  6")
        
#         # Counterfactuals
#         counterfactuals = self.generate_counterfactuals(scene_df)
#         print("rohit  7")
        
#         return {
#             'base_prediction': base,
#             'alternative_suspects': alt_suspects,
#             'weapon_scenarios': weapon_hyp,
#             'location_scenarios': location_hyp,
#             'counterfactuals': counterfactuals,
#             'original_scene': scene_df
#         }


# def format_hypothesis_report(report: Dict) -> str:
#     """Format hypothesis report as human-readable text."""
#     base = report['base_prediction']
    
#     text = f"""
# CRIME HYPOTHESIS REPORT
# {'='*70}

# ORIGINAL SCENE ANALYSIS
# Predicted Crime Type: {base['top_crime']} ({base['confidence']*100:.1f}% confidence)

# Top Crime Probabilities:
# """
    
#     # Sort and show top 3 crime predictions
#     sorted_crimes = sorted(base['crime_predictions'].items(), key=lambda x: -x[1])[:3]
#     for crime, prob in sorted_crimes:
#         bar = "█" * int(prob * 20)
#         text += f"  {crime:20s} {prob*100:5.1f}% {bar}\n"
    
#     # Alternative suspects
#     if report['alternative_suspects']:
#         text += f"\n{'='*70}\n"
#         text += "ALTERNATIVE SUSPECT HYPOTHESES\n\n"
#         for i, alt in enumerate(report['alternative_suspects'], 1):
#             text += f"{i}. {alt['scenario']}\n"
#             text += f"   → Predicted: {alt['predicted_crime']} ({alt['confidence']*100:.1f}% confidence)\n\n"
    
#     # Weapon scenarios
#     if report['weapon_scenarios']:
#         text += f"{'='*70}\n"
#         text += "WEAPON/OBJECT ANALYSIS\n\n"
#         for i, wp in enumerate(report['weapon_scenarios'][:3], 1):
#             text += f"{i}. {wp['scenario']}\n"
#             text += f"   → Predicted: {wp['predicted_crime']} ({wp['confidence']*100:.1f}% confidence)\n\n"
    
#     # Counterfactuals
#     if report['counterfactuals']:
#         text += f"{'='*70}\n"
#         text += "WHAT-IF SCENARIOS\n\n"
#         for i, cf in enumerate(report['counterfactuals'], 1):
#             text += f"{i}. {cf['scenario']}\n"
#             text += f"   {cf['description']}\n"
#             text += f"   → Predicted: {cf['predicted_crime']} ({cf['confidence']*100:.1f}% confidence)\n\n"
    
#     return text