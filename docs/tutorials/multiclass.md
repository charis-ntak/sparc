# More than two classes

The script of this tutorial is `examples/multiclass_wine.py`.

SPARC explains a scalar model output. For a classifier that output is the predicted probability of one class, so a problem with several classes is explained one class at a time, with `target_class` naming the class. The wine recognition data of scikit-learn (178 instances, 13 features, 3 cultivars) serves as the example.

```python
--8<-- "examples/multiclass_wine.py:data"
```

```python
--8<-- "examples/multiclass_wine.py:fit"
```

Every explainer trains the same kind of pool on the same data and differs only in the class whose probability is explained; the concept partitions are formed from the features alone, so the three explanations share the same candidate partitions and can be read side by side.

```python
--8<-- "examples/multiclass_wine.py:plots"
```

## Concept importance per class

![bar by class](../assets/tutorial_multiclass/bar_by_class.png)

## One instance, three class probabilities

The same wine, explained for every class. The concepts that raise the probability of its class lower the probabilities of the others, and the sign confidence tells how reliably.

![instance by class](../assets/tutorial_multiclass/instance_by_class.png)

## The partition and the labels

![membership](../assets/tutorial_multiclass/membership.png)
![linguistic profile](../assets/tutorial_multiclass/linguistic_profile.png)
![heatmap](../assets/tutorial_multiclass/heatmap.png)

!!! note "Gradient boosting with several classes"
    A scikit-learn gradient boosting classifier with more than two classes is explained with the permutation estimator rather than the exact tree estimator, which is slower. Random forests and histogram gradient boosting are explained exactly for any number of classes.
