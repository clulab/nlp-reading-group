# Mihai's Summary of EMNLP/CoNLL 2018

Disclaimer: this is obviously a small subset of the awesome papers/talks at EMNLP and CoNLL. I selected them subjectively, based solely on my own research interests. The papers are listed chronologically.

The EMNLP proceedings are available here: https://aclanthology.coli.uni-saarland.de/events/emnlp-2018. 
The CoNLL proceedings are here: https://aclanthology.coli.uni-saarland.de/events/conll-2018

## CoNLL DAY 1

### Adversarially Regularising Neural NLI Models to Integrate Logical Background Knowledge
**URL**: https://aclanthology.coli.uni-saarland.de/papers/K18-1007/k18-1007

**Abstract**: Adversarial examples are inputs to machine learning models designed to cause the model to make a mistake. They are useful for understanding the shortcomings of machine learning models, interpreting their results, and for regularisation. In NLP, however, most example generation strategies produce input text by using known, pre-specified semantic transformations, requiring significant manual effort and in-depth understanding of the problem and domain. In this paper, we investigate the problem of automatically generating adversarial examples that violate a set of given First-Order Logic constraints in Natural Language Inference (NLI). We reduce the problem of identifying such adversarial examples to a combinatorial optimisation problem, by maximising a quantity measuring the degree of violation of such constraints and by using a language model for generating linguistically-plausible examples. Furthermore, we propose a method for adversarially regularising neural NLI models for incorporating background knowledge. Our results show that, while the proposed method does not always improve results on the SNLI and MultiNLI datasets, it significantly and consistently increases the predictive accuracy on adversarially-crafted datasets – up to a 79.6% relative improvement – while drastically reducing the number of background knowledge violations. Furthermore, we show that adversarial examples transfer among model architectures, and that the proposed adversarial training procedure improves the robustness of NLI models to adversarial examples.

**Mihai's comments**: neat idea to infuse FOL rules into the training process for a NN for NLI. They add an "inconsistency loss" term to the regularizaton component of the loss function, which measure how much these rules are violated.



