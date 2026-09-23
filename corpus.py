"""Toy document corpus and labeled queries for similarity search."""
from __future__ import annotations

from typing import Dict, List

DOCUMENTS: List[Dict[str, str]] = [
    {"id": "a1", "topic": "animals", "text": "The cat sat on the mat and chased the mouse around the house."},
    {"id": "a2", "topic": "animals", "text": "Dogs and cats are popular pets that live with families."},
    {"id": "a3", "topic": "animals", "text": "Birds fly in the sky while fish swim in the deep ocean water."},
    {"id": "a4", "topic": "animals", "text": "Ducks swim in the pond and geese migrate across the lakes."},
    {"id": "w1", "topic": "weather", "text": "The sun rises in the east and sets in the west every day."},
    {"id": "w2", "topic": "weather", "text": "Rain clouds gather and storms bring thunder lightning and wind."},
    {"id": "w3", "topic": "weather", "text": "The moon shines at night under a clear starry sky."},
    {"id": "w4", "topic": "weather", "text": "Snow falls in winter while summer heat warms the land."},
    {"id": "p1", "topic": "programming", "text": "Python and Java are popular programming languages for software."},
    {"id": "p2", "topic": "programming", "text": "Neural networks and deep learning train models on large datasets."},
    {"id": "p3", "topic": "programming", "text": "Embeddings map words to vectors that capture semantic similarity."},
    {"id": "p4", "topic": "programming", "text": "HTML and CSS build web pages while JavaScript adds interactivity."},
    {"id": "r1", "topic": "royalty", "text": "The king and queen rule the kingdom from the royal castle."},
    {"id": "r2", "topic": "royalty", "text": "The prince and princess visit the palace for a grand ceremony."},
    {"id": "r3", "topic": "royalty", "text": "Knights protect the crown and serve the royal court with honor."},
    {"id": "s1", "topic": "science", "text": "Students study math and science in school with dedicated teachers."},
    {"id": "s2", "topic": "science", "text": "Physics explains gravity force energy and motion of particles."},
    {"id": "s3", "topic": "science", "text": "Chemistry mixes atoms and molecules to form new compounds."},
    {"id": "s4", "topic": "science", "text": "Biology studies living cells plants animals and ecosystems."},
]

LABELED_QUERIES: List[Dict[str, object]] = [
    {"query": "cat dog pets animals", "expected_topic": "animals"},
    {"query": "rain storm clouds weather", "expected_topic": "weather"},
    {"query": "python java programming code", "expected_topic": "programming"},
    {"query": "king queen castle royalty", "expected_topic": "royalty"},
    {"query": "math physics science school", "expected_topic": "science"},
    {"query": "neural networks embeddings vectors", "expected_topic": "programming"},
    {"query": "moon stars night sky", "expected_topic": "weather"},
    {"query": "birds fish swim fly", "expected_topic": "animals"},
]
