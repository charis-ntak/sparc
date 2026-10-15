"""SPARC: Shapley Pareto Robust Concept attribution with fuzzy logic.

A post hoc explainability method for tabular black box models. Features are
clustered into fuzzy concepts by their correlation profiles, the Shapley
credit of a prediction is assigned to concepts, the attribution of every
concept is estimated over bootstrap resamples and several model classes and
summarized as a trapezoidal fuzzy number with a sign confidence, a
disagreement index and linguistic labels, and the reported explanation is
selected from a Pareto front that trades off fidelity loss, complexity and
instability.

Author: Charis Ntakolia, Hellenic Air Force Academy.
"""

from sparc import plots  # noqa: E402
from sparc.attribution import AttributionEngine, AttributionResult, aggregate_to_concepts, default_model_classes, redistribute_to_features, shapley_values
from sparc.concept_shapley import concept_shapley_values, fuzzy_removal_weights
from sparc.concepts import ConceptFormer, ConceptPartition, apriori_membership, fuzzy_c_means
from sparc.explainer import Explainer, Explanation
from sparc.fuzzification import (
    IntervalType2FuzzyNumber,
    LinguisticLabelSet,
    TrapezoidalFuzzyNumber,
    fuzzify,
)
from sparc.selection import ExplanationConfiguration, ParetoSelector, knee_point, pareto_mask

# names of the study code base (pfca), kept for one release
PFCAExplainer = Explainer
PFCAExplanation = Explanation

__version__ = "0.1.1"
__author__ = "Charis Ntakolia"

__all__ = [
    "AttributionEngine",
    "AttributionResult",
    "ConceptFormer",
    "ConceptPartition",
    "Explainer",
    "Explanation",
    "ExplanationConfiguration",
    "IntervalType2FuzzyNumber",
    "LinguisticLabelSet",
    "ParetoSelector",
    "TrapezoidalFuzzyNumber",
    "aggregate_to_concepts",
    "apriori_membership",
    "concept_shapley_values",
    "default_model_classes",
    "fuzzify",
    "fuzzy_c_means",
    "fuzzy_removal_weights",
    "knee_point",
    "pareto_mask",
    "plots",
    "redistribute_to_features",
    "shapley_values",
]
