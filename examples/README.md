# Examples

Three scripts, one per kind of problem, that fit SPARC on a built in dataset and write every figure of the corresponding tutorial of the documentation site to `docs/assets/`. They are the source of the tutorials: the documentation embeds their code sections and their figures, so the figures a reader sees are the ones the code produces.

| Script | Problem | Shows |
|---|---|---|
| `regression_synthetic.py` | regression with a known concept structure (`sparc.synthetic`) | every plot, the coverage of a replicate, the recovery of the true blocks, the efficiency check |
| `classification_breast_cancer.py` | binary classification with ten known feature groups (scikit-learn breast cancer data) | known groups as a candidate partition, the explained class, the HTML card |
| `multiclass_wine.py` | three classes (scikit-learn wine data) | one explainer per class, the same instance explained for every class |

```
python examples/regression_synthetic.py
python examples/classification_breast_cancer.py
python examples/multiclass_wine.py
```

Each takes a few minutes on a laptop; an optional argument sets the output directory.
