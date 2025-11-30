# NERExtraction.py

"""
NER-based entity extraction for CrimeLens with strict mapping
to known model labels (no unseen labels for LabelEncoders).
"""

import re
from typing import List, Optional, Dict, Set

import spacy
import pandas as pd


class NEREntityExtractor:
    """
    spaCy-based extractor that ALWAYS maps actions/objects/locations
    back to known labels from the trained encoders.
    """

    def __init__(
        self,
        known_actions: Optional[Set[str]] = None,
        known_objects: Optional[Set[str]] = None,
        known_locations: Optional[Set[str]] = None,
    ):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            import subprocess

            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")

        self.known_actions = set(known_actions or [])
        self.known_objects = set(known_objects or [])
        self.known_locations = set(known_locations or [])

        # Crime verbs / object keywords only help detection,
        # final labels are always mapped to known_* sets.
        self.crime_verbs = {
            "stab", "stabbed", "shot", "shoot", "attacked", "assaulted",
            "robbed", "stole", "threatened", "hit", "punched", "kicked",
            "beat", "struck", "broke", "entered", "fled", "escaped", "ran",
        }

        self.object_keywords = {
            "knife", "gun", "pistol", "rifle", "blade", "dagger", "bat",
            "wallet", "purse", "phone", "cash", "money", "bag", "rope",
        }

    # --------- Common helpers ---------

    def _map_to_known(self, value: str, known_set: Set[str], default_if_empty: str) -> str:
        """
        Map a raw string to the closest known label.
        Guarantees: returned label ∈ known_set (if known_set non-empty),
        otherwise returns default_if_empty.
        """
        if not known_set:
            return default_if_empty

        v = (value or "").lower().strip()
        if not v:
            return sorted(known_set)[0]

        # 1) exact (case-insensitive)
        for k in known_set:
            if k.lower() == v:
                return k

        # 2) substring / fuzzy-ish
        for k in known_set:
            kl = k.lower()
            if v in kl or kl in v:
                return k

        # 3) last resort: first label, deterministic
        return sorted(known_set)[0]

    # --------- Extraction primitives ---------

    def _extract_persons(self, doc) -> List[str]:
        persons = []
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                name = ent.text.strip()
                if name and name.lower() not in {"he", "she", "they", "him", "her", "them", "i", "you"}:
                    persons.append(name)
        # fallback: capitalized proper nouns
        if not persons:
            for tok in doc:
                if tok.pos_ == "PROPN" and tok.text[0].isupper():
                    skip = {"The", "A", "An", "This", "That", "When", "Where"}
                    if tok.text not in skip:
                        persons.append(tok.text)
        return list(dict.fromkeys(persons))  # dedupe, preserve order

    def _extract_locations_raw(self, doc) -> List[str]:
        locs = []
        text_lower = doc.text.lower()

        # spaCy location entities
        for ent in doc.ents:
            if ent.label_ in {"GPE", "LOC", "FAC"}:
                locs.append(ent.text.lower())

        # pattern-based
        patterns = [
            r"\bat (?:the )?([a-zA-Z_]+)",
            r"\bin (?:the )?([a-zA-Z_]+)",
            r"\bon (?:the )?([a-zA-Z_]+)",
        ]
        for pat in patterns:
            locs.extend(re.findall(pat, text_lower))

        # known locations from training
        for kl in self.known_locations:
            if kl.lower() in text_lower:
                locs.append(kl)

        return list(dict.fromkeys(locs)) or ["unknown"]

    def _extract_objects_raw(self, doc) -> List[str]:
        objs = []
        text_lower = doc.text.lower()

        # keyword match
        for kw in self.object_keywords:
            if kw in text_lower:
                objs.append(kw)

        # known objects from training
        for ko in self.known_objects:
            if ko.lower() in text_lower:
                objs.append(ko)

        # dependency-based direct objects
        for tok in doc:
            if tok.dep_ in {"dobj", "pobj"} and tok.pos_ == "NOUN":
                objs.append(tok.text.lower())

        # "with a/an X"
        for m in re.findall(r"with (?:a |an |the )?([a-zA-Z_]+)", text_lower):
            objs.append(m)

        return list(dict.fromkeys(objs)) or ["unknown"]

    def _extract_actions_raw(self, doc) -> List[str]:
        acts = []
        text_lower = doc.text.lower()

        # from known_actions
        for ka in self.known_actions:
            if ka.lower() in text_lower:
                acts.append(ka)

        # crime verbs
        for v in self.crime_verbs:
            if v in text_lower:
                acts.append(v)

        # past tense verbs
        for tok in doc:
            if tok.pos_ == "VERB" and tok.tag_ in {"VBD", "VBN"}:
                lemma = tok.lemma_.lower()
                if lemma not in {"be", "have", "do"}:
                    acts.append(tok.text.lower())

        return list(dict.fromkeys(acts)) or ["unknown"]

    def _assign_roles(self, persons: List[str], doc, action: str):
        if len(persons) >= 2:
            # rough heuristic: first name before verb = suspect, next = victim
            action_idx = None
            for i, tok in enumerate(doc):
                if action.lower() in tok.text.lower():
                    action_idx = i
                    break

            if action_idx is not None:
                person_pos = []
                for p in persons:
                    for i, tok in enumerate(doc):
                        if p.lower() in tok.text.lower():
                            person_pos.append((p, i))
                            break
                if len(person_pos) >= 2:
                    person_pos.sort(key=lambda x: x[1])
                    return person_pos[0][0], person_pos[1][0]

            # fallback
            return persons[0], persons[1]

        if len(persons) == 1:
            return persons[0], "Unknown Victim"

        return "Unknown Suspect", "Unknown Victim"

    # --------- Public API ---------

    def parse_sentence(self, sentence: str, default_crime_type: str = "unknown") -> Optional[pd.DataFrame]:
        """
        Parse a single sentence into a **model-safe** row:
        action/object/location are mapped into known encoder labels.
        """
        doc = self.nlp(sentence)

        persons = self._extract_persons(doc)
        raw_locations = self._extract_locations_raw(doc)
        raw_objects = self._extract_objects_raw(doc)
        raw_actions = self._extract_actions_raw(doc)

        # pick first candidate
        raw_action = raw_actions[0]
        raw_object = raw_objects[0]
        raw_location = raw_locations[0]

        # map to known labels (critical!)
        action = self._map_to_known(
            raw_action, self.known_actions, default_if_empty=raw_action
        )
        obj = self._map_to_known(
            raw_object, self.known_objects, default_if_empty=raw_object
        )
        loc = self._map_to_known(
            raw_location, self.known_locations, default_if_empty=raw_location
        )

        # assign roles
        suspect, victim = self._assign_roles(persons, doc, action)

        data = {
            "scene_id": [1],
            "event_id": [1],
            "crime_type": [default_crime_type],
            "suspect": [suspect],
            "victim": [victim],
            "object": [obj],
            "location": [loc],
            "action": [action],
        }
        return pd.DataFrame(data)

    def parse_multiple_sentences(
        self, text: str, default_crime_type: str = "unknown"
    ) -> Optional[pd.DataFrame]:
        sentences = [s.strip() for s in text.split("\n") if s.strip()]
        events = []

        for i, s in enumerate(sentences):
            parsed = self.parse_sentence(s, default_crime_type)
            if parsed is not None:
                parsed["event_id"] = i + 1
                events.append(parsed)

        if not events:
            return None

        df = pd.concat(events, ignore_index=True)
        df["scene_id"] = 1
        return df


class ImprovedCrimeSentenceParser:
    """
    Drop-in parser used by the Streamlit app:
    wraps NEREntityExtractor and passes known label sets from graph_builder.
    """

    def __init__(self, graph_builder):
        self.graph_builder = graph_builder

        self.use_ner = True

        known_actions = set(
            getattr(graph_builder.action_encoder, "classes_", [])
        )
        known_objects = set(
            getattr(graph_builder.object_encoder, "classes_", [])
        )
        known_locations = set(
            getattr(graph_builder.location_encoder, "classes_", [])
        )

        self.extractor = NEREntityExtractor(
            known_actions=known_actions,
            known_objects=known_objects,
            known_locations=known_locations,
        )

        self.default_crime = (
            graph_builder.crime_encoder.classes_[0]
            if hasattr(graph_builder, "crime_encoder")
            else "unknown"
        )

    def parse(self, sentence: str) -> Optional[pd.DataFrame]:
        return self.extractor.parse_sentence(sentence, self.default_crime)

    def parse_multiple(self, sentences: List[str]) -> Optional[pd.DataFrame]:
        text = "\n".join(sentences)
        return self.extractor.parse_multiple_sentences(text, self.default_crime)