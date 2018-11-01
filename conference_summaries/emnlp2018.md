# Mihai's Summary of EMNLP/CoNLL 2018

Disclaimer: this is obviously a small subset of the awesome papers/talks at EMNLP and CoNLL. I selected them subjectively, based solely on my own research interests. The papers are listed chronologically.

The EMNLP proceedings are available here: https://aclanthology.coli.uni-saarland.de/events/emnlp-2018. 
The CoNLL proceedings are here: https://aclanthology.coli.uni-saarland.de/events/conll-2018

## CoNLL DAY 1

### Adversarially Regularising Neural NLI Models to Integrate Logical Background Knowledge
**URL**: https://aclanthology.coli.uni-saarland.de/papers/K18-1007/k18-1007

**Abstract**: Adversarial examples are inputs to machine learning models designed to cause the model to make a mistake. They are useful for understanding the shortcomings of machine learning models, interpreting their results, and for regularisation. In NLP, however, most example generation strategies produce input text by using known, pre-specified semantic transformations, requiring significant manual effort and in-depth understanding of the problem and domain. In this paper, we investigate the problem of automatically generating adversarial examples that violate a set of given First-Order Logic constraints in Natural Language Inference (NLI). We reduce the problem of identifying such adversarial examples to a combinatorial optimisation problem, by maximising a quantity measuring the degree of violation of such constraints and by using a language model for generating linguistically-plausible examples. Furthermore, we propose a method for adversarially regularising neural NLI models for incorporating background knowledge. Our results show that, while the proposed method does not always improve results on the SNLI and MultiNLI datasets, it significantly and consistently increases the predictive accuracy on adversarially-crafted datasets – up to a 79.6% relative improvement – while drastically reducing the number of background knowledge violations. Furthermore, we show that adversarial examples transfer among model architectures, and that the proposed adversarial training procedure improves the robustness of NLI models to adversarial examples.

**Mihai's comments**: neat idea to infuse FOL rules into the training process for a NN for NLI. They add an "inconsistency loss" term to the regularizaton component of the loss function, which measure how much these rules are violated.

## BlackboxNLP Workshop

Proceedings for this workshop are here: http://aclweb.org/anthology/W18-5400

### Interpretable Structure Induction Via Sparse Attention

**URL**: http://aclweb.org/anthology/W18-5450

**Mihai's comments**: forces NNs to be more interpretable by forcing them to yield sparse probabilities, which can be traced back to the most relevant parts of the input.

### Understanding Convolutional Neural Networks for Text Classification

**Abstract**: We present an analysis into the inner workings of Convolutional Neural Networks (CNNs) for processing text. CNNs used for computer vision can be interpreted by projecting filters into image space, but for discrete sequence inputs CNNs remain a mystery. We aim to understand the method by which the networks process and classify text. We examine common hypotheses to this problem: that filters, accompanied by global max-pooling, serve as ngram detectors. We show that filters may capture several different semantic classes of ngrams by using different activation patterns, and that global max-pooling induces behavior which separates important ngrams from the rest. Finally, we show practical use cases derived from our findings in the form of model interpretability (explaining a trained model by deriving a concrete identity for each filter, bridging the gap between visualization tools in vision tasks and NLP) and prediction interpretability (explaining predictions).

**Mihai's comments**: Identifies the relevant n-grams behind a CNN. Intuitively, the informative n-grams are selected based on their correlation with the corresponding convolutional filter.


