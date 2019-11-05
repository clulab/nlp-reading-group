# Mihai's Summary of EMNLP 2019

Disclaimer: this is obviously a small subset of the awesome papers/talks at EMNLP and the associated workshops. I selected them subjectively, based solely on my own research interests, and on what I have attended. The papers are listed chronologically.

The proceedings for EMNLP and all the workshops are available here: https://www.aclweb.org/anthology/events/emnlp-2019/

# Tutorials

### Graph Neural Networks for NLP
**URL**: https://github.com/svjan5/GNNs-for-NLP (includes code and slides)

**Mihai's comments**: if your data is a graph, you need this.

# Workshop papers

### Syntax-aware Multi-task Graph Convolutional Networks for Biomedical Relation Extraction
**URL**: https://www.aclweb.org/anthology/D19-6204/

**Abstract**: In this paper we tackle two unique challenges in biomedical relation extraction. The first challenge is that the contextual information between two entity mentions often involves sophisticated syntactic structures. We propose a novel graph convolutional networks model that incorporates dependency parsing and contextualized embedding to effectively capture comprehensive contextual information. The second challenge is that most of the benchmark data sets for this task are quite imbalanced because more than 80\% mention pairs are negative instances (i.e., no relations). We propose a multi-task learning framework to jointly model relation identification and classification tasks to propagate supervision signals from each other and apply a focal loss to focus training on ambiguous mention pairs. By applying these two strategies, experiments show that our model achieves state-of-the-art F-score on the 2013 drug-drug interaction extraction task.

**Mihai's comments**: a nice example of graph convolutional NNs (GCN). Also, a good dataset for relation extraction. Relevant to people working on relation/event extraction.

### Layerwise Relevance Visualization in Convolutional Text Graph Classifiers
**URL**: https://www.aclweb.org/anthology/D19-5308/

**Abstract**: Representations in the hidden layers of Deep Neural Networks (DNN) are often hard to interpret since it is difficult to project them into an interpretable domain. Graph Convolutional Networks (GCN) allow this projection, but existing explainability methods do not exploit this fact, i.e. do not focus their explanations on intermediate states. In this work, we present a novel method that traces and visualizes features that contribute to a classification decision in the visible and hidden layers of a GCN. Our method exposes hidden cross-layer dynamics in the input graph structure. We experimentally demonstrate that it yields meaningful layerwise explanations for a GCN sentence classifier.

**Mihai's comments**: Layerwise relevance propagation for GCNs!

# EMNLP papers

### To Annotate or Not? Predicting Performance Drop under Domain Shift
**URL**: https://www.aclweb.org/anthology/D19-1222/
**Abstract**: Performance drop due to domain-shift is an endemic problem for NLP models in produc- tion. This problem creates an urge to con- tinuously annotate evaluation datasets to mea- sure the expected drop in the model perfor- mance which can be prohibitively expensive and slow. In this paper, we study the problem of predicting the performance drop of mod- ern NLP models under domain-shift, in the ab- sence of any target domain labels. We investi- gate three families of methods (H-divergence, reverse classification accuracy and confidence measures), show how they can be used to pre- dict the performance drop and study their ro- bustness to adversarial domain-shifts. Our re- sults on sentiment classification and sequence labelling show that our method is able to pre- dict performance drops with an error rate as low as 2.15% and 0.89% for sentiment analy- sis and POS tagging respectively.

**Mihai's comments**: Discusses several methods to detect what will happen to your model when it is evaluated out of domain, *without* any data in the target domain.

### Show Your Work: Improved Reporting of Experimental Results
**URL**: https://arxiv.org/abs/1909.03004

**Abstract**: Research in natural language processing pro- ceeds, in part, by demonstrating that new mod- els achieve superior performance (e.g., accu- racy) on held-out test data, compared to pre- vious results. In this paper, we demonstrate that test-set performance scores alone are in- sufficient for drawing accurate conclusions about which model performs best. We argue for reporting additional details, especially per- formance on validation data obtained during model development. We present a novel tech- nique for doing so: expected validation per- formance of the best-found model as a func- tion of computation budget (i.e., the number of hyperparameter search trials or the overall training time). Using our approach, we find multiple recent model comparisons where au- thors would have reached a different conclu- sion if they had used more (or less) compu- tation. Our approach also allows us to esti- mate the amount of computation required to obtain a given accuracy; applying it to sev- eral recently published results yields massive variation across papers, from hours to weeks. We conclude with a set of best practices for reporting experimental results which allow for robust future comparison, and provide code to allow researchers to use our technique.

**Mihai's comments**: performance should be normalized by computation effort invested in it. That is, report how much tuning you did.

### Attention is not not Explanation
**URL**: https://www.aclweb.org/anthology/D19-1002.pdf

**Abstract**: Attention mechanisms play a central role in
NLP systems, especially within recurrent neural network (RNN) models. Recently, there
has been increasing interest in whether or
not the intermediate representations offered by
these modules may be used to explain the reasoning for a model’s prediction, and consequently reach insights regarding the model’s
decision-making process. A recent paper
claims that ‘Attention is not Explanation’ (Jain
and Wallace, 2019). We challenge many of
the assumptions underlying this work, arguing that such a claim depends on one’s definition of explanation, and that testing it needs
to take into account all elements of the model.
We propose four alternative tests to determine
when/whether attention can be used as explanation: a simple uniform-weights baseline;
a variance calibration based on multiple random seed runs; a diagnostic framework using
frozen weights from pretrained models; and an
end-to-end adversarial attention training protocol. Each allows for meaningful interpretation of attention mechanisms in RNN models.
We show that even when reliable adversarial
distributions can be found, they don’t perform
well on the simple diagnostic, indicating that
prior work does not disprove the usefulness of
attention mechanisms for explainability.

**Mihai's comments**: this paper opens up (or keeps open) the discussion on if we can use attentition weights as explanations. The paper shows that attention weights are sometimes needed for good performance (and sometimes now...). Second, they show that sometimes attention weights are faightful, that is, they can't be easily manipulated in adversarial seetings (but sometimes they can...).

### Practical Obstacles to Deploying Active Learning
**URL**: https://www.aclweb.org/anthology/D19-1003.pdf

**Abstract**: Active learning (AL) is a widely-used training strategy for maximizing predictive performance subject to a fixed annotation budget. In
AL one iteratively selects training examples
for annotation, often those for which the current model is most uncertain (by some measure). The hope is that active sampling leads
to better performance than would be achieved
under independent and identically distributed
(i.i.d.) random samples. While AL has
shown promise in retrospective evaluations,
these studies often ignore practical obstacles
to its use. In this paper we show that while
AL may provide benefits when used with specific models and for particular domains, the
benefits of current approaches do not generalize reliably across models and tasks. This is
problematic because in practice one does not
have the opportunity to explore and compare
alternative AL strategies. Moreover, AL couples the training dataset with the model used
to guide its acquisition. We find that subsequently training a successor model with an
actively-acquired dataset does not consistently
outperform training on i.i.d. sampled data.
Our findings raise the question of whether the
downsides inherent to AL are worth the modest and inconsistent performance gains it tends
to afford.

**Mihai's comments**: AL heuristics do not always work better than random choice for classification and sequence tasks in NLP.

### Transfer Learning Between Related Tasks Using Expected Label Proportions
**URL**: https://www.aclweb.org/anthology/D19-1004.pdf

**Abstract**: Deep learning systems thrive on abundance of
labeled training data but such data is not always available, calling for alternative methods
of supervision. One such method is expectation regularization (XR) (Mann and McCallum, 2007), where models are trained based
on expected label proportions. We propose
a novel application of the XR framework for
transfer learning between related tasks, where
knowing the labels of task A provides an estimation of the label proportion of task B.
We then use a model trained for A to label
a large corpus, and use this corpus with an
XR loss to train a model for task B. To make
the XR framework applicable to large-scale
deep-learning setups, we propose a stochastic
batched approximation procedure. We demonstrate the approach on the task of Aspectbased Sentiment classification, where we effectively use a sentence-level sentiment predictor to train accurate aspect-based predictor.
The method improves upon fully supervised
neural system trained on aspect-level data, and
is also cumulative with LM-based pretraining, as we demonstrate by improving a BERTbased Aspect-based Sentiment model.

**Mihai's comments**: Distant supervision using *only* label proportions as supervision. Uses expectation regularization as the objective function. Also includes a transfer learning algorithm under this setup.

### A Little Annotation does a Lot of Good: A Study in Bootstrapping Low-resource Named Entity Recognizers
**URL**: https://www.aclweb.org/anthology/D19-1520.pdf

**Abstract**:Most state-of-the-art models for named entity recognition (NER) rely on the availability of large amounts of labeled data, making them challenging to extend to new, lowerresourced languages. However, there are now
several proposed approaches involving either
cross-lingual transfer learning, which learns
from other highly resourced languages, or active learning, which efficiently selects effective training data based on model predictions.
This paper poses the question: given this recent progress, and limited human annotation,
what is the most effective method for efficiently creating high-quality entity recognizers
in under-resourced languages? Based on extensive experimentation using both simulated
and real human annotation, we find a dualstrategy approach best, starting with a crosslingual transferred model, then performing targeted annotation of only uncertain entity spans
in the target language, minimizing annotator
effort. Results demonstrate that cross-lingual
transfer is a powerful tool when very little data
can be annotated, but an entity-targeted annotation strategy can achieve competitive accuracy quickly, with just one-tenth of training
data. 

**Mihai's comments**: Uses cross-language transfer to get annotated data from a high-resource language into a low-resource one. Then they use active learning to clean the data. They use a partial-CRF to train on sentences with partial annotations.

### Title
**URL**:
**Abstract**:
**Mihai's comments**: 

