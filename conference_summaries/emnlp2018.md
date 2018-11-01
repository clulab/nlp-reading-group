# Mihai's Summary of EMNLP/CoNLL 2018

Disclaimer: this is obviously a small subset of the awesome papers/talks at EMNLP and CoNLL. I selected them subjectively, based solely on my own research interests. The papers are listed chronologically.

The EMNLP proceedings are available here: https://aclanthology.coli.uni-saarland.de/events/emnlp-2018. 
The CoNLL proceedings are here: https://aclanthology.coli.uni-saarland.de/events/conll-2018

# CoNLL

### Adversarially Regularising Neural NLI Models to Integrate Logical Background Knowledge
**URL**: https://aclanthology.coli.uni-saarland.de/papers/K18-1007/k18-1007

**Abstract**: Adversarial examples are inputs to machine learning models designed to cause the model to make a mistake. They are useful for understanding the shortcomings of machine learning models, interpreting their results, and for regularisation. In NLP, however, most example generation strategies produce input text by using known, pre-specified semantic transformations, requiring significant manual effort and in-depth understanding of the problem and domain. In this paper, we investigate the problem of automatically generating adversarial examples that violate a set of given First-Order Logic constraints in Natural Language Inference (NLI). We reduce the problem of identifying such adversarial examples to a combinatorial optimisation problem, by maximising a quantity measuring the degree of violation of such constraints and by using a language model for generating linguistically-plausible examples. Furthermore, we propose a method for adversarially regularising neural NLI models for incorporating background knowledge. Our results show that, while the proposed method does not always improve results on the SNLI and MultiNLI datasets, it significantly and consistently increases the predictive accuracy on adversarially-crafted datasets – up to a 79.6% relative improvement – while drastically reducing the number of background knowledge violations. Furthermore, we show that adversarial examples transfer among model architectures, and that the proposed adversarial training procedure improves the robustness of NLI models to adversarial examples.

**Mihai's comments**: neat idea to infuse FOL rules into the training process for a NN for NLI. They add an "inconsistency loss" term to the regularizaton component of the loss function, which measure how much these rules are violated.

# BlackboxNLP Workshop

Proceedings for this workshop are here: http://aclweb.org/anthology/W18-5400

### Interpretable Structure Induction Via Sparse Attention

**URL**: http://aclweb.org/anthology/W18-5450

**Mihai's comments**: forces NNs to be more interpretable by forcing them to yield sparse probabilities, which can be traced back to the most relevant parts of the input.

### Understanding Convolutional Neural Networks for Text Classification

**URL**: https://arxiv.org/pdf/1809.08037.pdf

**Abstract**: We present an analysis into the inner workings of Convolutional Neural Networks (CNNs) for processing text. CNNs used for computer vision can be interpreted by projecting filters into image space, but for discrete sequence inputs CNNs remain a mystery. We aim to understand the method by which the networks process and classify text. We examine common hypotheses to this problem: that filters, accompanied by global max-pooling, serve as ngram detectors. We show that filters may capture several different semantic classes of ngrams by using different activation patterns, and that global max-pooling induces behavior which separates important ngrams from the rest. Finally, we show practical use cases derived from our findings in the form of model interpretability (explaining a trained model by deriving a concrete identity for each filter, bridging the gap between visualization tools in vision tasks and NLP) and prediction interpretability (explaining predictions).

**Mihai's comments**: Identifies the relevant n-grams behind a CNN. Intuitively, the informative n-grams are selected based on their correlation with the corresponding convolutional filter.

### Rule induction for global explanation of trained models

**URL**: https://arxiv.org/abs/1808.09744

**Abstract**: Understanding the behavior of a trained network and finding explanations for its outputs is important for improving the network's performance and generalization ability, and for ensuring trust in automated systems. Several approaches have previously been proposed to identify and visualize the most important features by analyzing a trained network. However, the relations between different features and classes are lost in most cases. We propose a technique to induce sets of if-then-else rules that capture these relations to globally explain the predictions of a network. We first calculate the importance of the features in the trained network. We then weigh the original inputs with these feature importance scores, simplify the transformed input space, and finally fit a rule induction model to explain the model predictions. We find that the output rule-sets can explain the predictions of a neural network trained for 4-class text classification from the 20 newsgroups dataset to a macro-averaged F-score of 0.80.

**Mihai's comments**: Similar idea to our "Snap to Grid" paper. But they have  the extra step at the end where they induce a rule model from the important features.

### How much should you ask? On the question structure in QA systems

**URL**: https://arxiv.org/pdf/1809.03734.pdf

**Abstract**: Datasets that boosted state-of-the-art solutions for Question Answering (QA) systems prove that it is possible to ask questions in natural language manner. However, users are still used to query-like systems where they type in key- words to search for answer. In this study we validate which parts of questions are essential for obtaining valid answer. In order to conc- lude that, we take advantage of LIME - a fra- mework that explains prediction by local ap- proximation. We find that grammar and na- tural language is disregarded by QA. State- of-the-art model can answer properly even if ’asked’ only with a few words with high co- efficients calculated with LIME. According to our knowledge, it is the first time that QA mo- del is being explained by LIME.

**Mihai's comments**: reduces the text in SQuAD questions to 1 or just a few words, and DrQA still works...

###  Does it care what you asked? Understanding Importance of Verbs in Deep Learning QA System

**URL**: https://arxiv.org/abs/1809.03740

**Abstract**: In this paper we present the results of an investigation of the importance of verbs in a deep learning QA system trained on SQuAD dataset. We show that main verbs in questions carry little influence on the decisions made by the system - in over 90% of researched cases swapping verbs for their antonyms did not change system decision. We track this phenomenon down to the insides of the net, analyzing the mechanism of self-attention and values contained in hidden layers of RNN. Finally, we recognize the characteristics of the SQuAD dataset as the source of the problem. Our work refers to the recently popular topic of adversarial examples in NLP, combined with investigating deep net structure.

**Mihai's comments**: messing with verbs in SQuAD questions does not change the answer...

### Firearms and Tigers are Dangerous, Kitchen Knives and Zebras are Not: Testing whether Word Embeddings Can Tell

**URL**: https://arxiv.org/abs/1809.01375

**Abstract**: This paper presents an approach for investigating the nature of semantic information captured by word embeddings. We propose a method that extends an existing human-elicited semantic property dataset with gold negative examples using crowd judgments. Our experimental approach tests the ability of supervised classifiers to identify semantic features in word embedding vectors and com- pares this to a feature-identification method based on full vector cosine similarity. The idea behind this method is that properties identified by classifiers, but not through full vector comparison are captured by embeddings. Properties that cannot be identified by either method are not. Our results provide an initial indication that semantic properties relevant for the way entities interact (e.g. dangerous) are captured, while perceptual information (e.g. colors) is not represented. We conclude that, though preliminary, these results show that our method is suitable for identifying which properties are captured by embeddings.

**Mihai's comments**: investigates which properties are actually captured by simple word embeddings. Buzz word: "diagnostic classifier"



