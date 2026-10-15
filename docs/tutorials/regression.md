# Regression with a known concept structure

The script of this tutorial is `examples/regression_synthetic.py`; every figure below is produced by it.

The problem has three latent factors, each measured by three features that correlate at 0.8 within the block, and a nonlinear response. The true concepts are the three blocks, so the recovered partition, the stability of the ranking and the calibration of the intervals can all be checked.

## Data and explainer

```python
--8<-- "examples/regression_synthetic.py:data"
```

```python
--8<-- "examples/regression_synthetic.py:fit"
```

The explainer prints its summary and the sentence form of the first instance. The knee explanation here has three concepts, one per block, and retains two of them.

## The explanation at a glance

```python
--8<-- "examples/regression_synthetic.py:plots"
```

### Which concepts matter, over all instances and for one

The global bar shows the mean absolute attribution of every concept with the support of its fuzzy global importance; the asterisk marks the concepts the knee retains. The beeswarm shows the same per instance, coloured by the concept score of the instance.

![bar](../assets/tutorial_regression/bar.png)
![beeswarm](../assets/tutorial_regression/beeswarm.png)

### One instance, with its uncertainty

The waterfall walks from the base value to the model output concept by concept. The interval bars add what a point estimate hides: the core and the support of every attribution, its sign confidence and the disagreement between model classes. The fuzzy attribution plot draws the membership functions themselves; where the model classes disagree beyond the threshold the attribution is of type 2 and the band between the model classes is shown.

![waterfall](../assets/tutorial_regression/waterfall.png)
![interval bars](../assets/tutorial_regression/interval_bars.png)
![fuzzy attribution](../assets/tutorial_regression/fuzzy_attribution.png)

### Words, and where they come from

The linguistic profile is the compatibility of every label with every concept's fuzzy number; the best label is what `describe` reports. The disagreement plot shows the resamples of every model class separately, which is where the disagreement index comes from.

![linguistic profile](../assets/tutorial_regression/linguistic_profile.png)
![disagreement](../assets/tutorial_regression/disagreement.png)

### The concepts

The membership heat map and the concept graph show the fuzzy partition: three blocks, recovered from the correlation profiles alone.

![membership](../assets/tutorial_regression/membership.png)
![concept graph](../assets/tutorial_regression/concept_graph.png)

### The front

The knee is a default, not a limit. The front plot shows every evaluated configuration, the Pareto front, the knee and the fidelity optimal member; the navigator shows the same instance explained at three members of the front, from the sparsest to the most faithful. The selection membership tells how often the front retains every concept and every feature, a robustness that no single explanation can show.

![front](../assets/tutorial_regression/front.png)
![front navigator](../assets/tutorial_regression/front_navigator.png)
![selection membership](../assets/tutorial_regression/selection_membership.png)

### The familiar views

Heat map of instances by concepts, dependence of a concept's attribution on its score, force plot and decision plot of the whole sample.

![heatmap](../assets/tutorial_regression/heatmap.png)
![dependence](../assets/tutorial_regression/dependence.png)
![force](../assets/tutorial_regression/force.png)
![decision](../assets/tutorial_regression/decision.png)

## Are the intervals honest?

A replicate attribution, the same instances explained by an explainer fitted on another split of the data, should fall inside the support about 90 percent of the time and inside the core about 50 percent of the time.

```python
--8<-- "examples/regression_synthetic.py:coverage"
```

![coverage](../assets/tutorial_regression/coverage.png)

## Tables and checks

```python
--8<-- "examples/regression_synthetic.py:tables"
```

```python
--8<-- "examples/regression_synthetic.py:truth"
```

The last two lines verify two things that hold by construction: the concept values plus the base value reproduce the model output of every instance, and the hardened partition is compared with the true blocks by the adjusted Rand index.
