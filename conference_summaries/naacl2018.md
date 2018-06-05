# Mihai's Summary of NAACL 2018

Disclaimer: this is obviously a small subset of the awesome papers/talks at NAACL. I selected them subjectively, based solely on my own research interests. The papers are listed chronologically.

The NAACL proceedings are available here: https://aclanthology.coli.uni-saarland.de/events/naacl-2018

## DAY 1

### Keynote by Charles Yang on how children learn 
*Mihai's comments:* key statement: for a rule to be good and acquired by children, the number of exceptions should be smaller than `N/ln(N)`, where `N` is the total number of cases where it applies. This has immediate applications to rule induction algorithms!

### Label-Aware Double Transfer Learning for Cross-Specialty Medical Named Entity Recognition
_Zhenghui Wang, Yanru Qu, Liheng Chen, Jian Shen, Weinan Zhang, Shaodian Zhang, Yimei Gao, Gen Gu, Ken Chen, and Yong Yu_

We study the problem of named entity recognition (NER) from electronic medical records, which is one of the most fundamental and critical problems for medical text mining. Medical records which are written by clinicians from different specialties usually contain quite different terminologies and writing styles. The difference of specialties and the cost of human annotation makes it particularly difficult to train a universal medical NER system. In this paper, we propose a label-aware double transfer learning framework (La-DTL) for cross-specialty NER, so that a medical NER system designed for one specialty could be conveniently applied to another one with minimal annotation efforts. The transferability is guaranteed by two components: (i) we propose label-aware MMD for feature representation transfer, and (ii) we perform parameter transfer with a theoretical upper bound which is also label aware. We conduct extensive experiments on 12 cross-specialty NER tasks. The experimental results demonstrate that La-DTL provides consistent accuracy improvement over strong baselines. Besides, the promising experimental results on non-medical NER scenarios indicate that La-DTL is potential to be seamlessly adapted to a wide range of NER tasks.

*Mihai's comments*: Discusses multitask training with a framework in which the usual sharing of parameters (in this case a BiLSTM) is constrained by encouraging the label distribution in the source and destination domains to be similar. Useful, in cases where such distributions are similar.

### Joint Bootstrapping Machines for High Confidence Relation Extraction
_Pankaj Gupta, Benjamin Roth, and Hinrich Schütze_

Semi-supervised bootstrapping techniques for relationship extraction from text iteratively expand a set of initial seed instances. Due to the lack of labeled data, a key challenge in bootstrapping is semantic drift: if a false positive instance is added during an iteration, then all following iterations are contaminated. We introduce BREX, a new bootstrapping method that protects against such contamination by highly effective confidence assessment. This is achieved by using entity and template seeds jointly (as opposed to just one as in previous work), by expanding entities and templates in parallel and in a mutually constraining fashion in each iteration and by introducing higherquality similarity measures for templates. Experimental results show that BREX achieves an F1 that is 0.13 (0.87 vs. 0.74) better than the state of the art for four relationships.

*Mihai's comments*: I missed this one. But it sounds cool and relevant to our IE work. Must read!

### Comparing Constraints for Taxonomic Organization
_Anne Cocos, Marianna Apidianaki, and Chris Callison-Burch_

Building a taxonomy from the ground up involves several sub-tasks: selecting terms to include, predicting semantic relations between terms, and selecting a subset of relational instances to keep, given constraints on the taxonomy graph. Methods for this final step – taxonomic organization – vary both in terms of the con- straints they impose, and whether they enable discovery of synonymous terms. It is hard to isolate the impact of these factors on the quality of the resulting taxonomy because organization methods are rarely compared directly. In this paper, we present a head-to-head comparison of six taxonomic organization algorithms that vary with respect to their structural and transitivity constraints, and treatment of synonymy. We find that while transitive algorithms out-perform their non-transitive counterparts, the top-performing transitive algo- rithm is prohibitively slow for taxonomies with as few as 50 entities. We propose a simple modification to a non-transitive optimum branching algorithm to explicitly incorporate synonymy, resulting in a method that is substantially faster than the best transitive algorithm while giving complementary performance.

*Mihai's comments*: An analysis of taxonomy acquisition algorithms. This is a great starting point for any work on taxonomy acquisition. 

### Multiple Instance Learning Networks for Fine-Grained Sentiment Analysis (TACL)
_Stefanos Angelidis and Mirella Lapata_

We consider the task of fine-grained sentiment analysis from the perspective of multiple instance learning (MIL). Our neural model is trained on document sentiment labels, and learns to predict the sentiment of text segments, i.e. sentences or elementary discourse units (EDUs), without segment-level supervision. We intro- duce an attention-based polarity scoring method for identifying positive and negative text snippets and a new dataset which we call SPOT (as shorthand for Segment-level POlariTy annotations) for evaluating MIL-style sentiment models like ours. Experimental results demonstrate superior performance against multiple baselines, whereas a judgement elicitation study shows that EDU-level opinion extraction produces more informative summaries than sentence-based alternatives.

*Mihai's comments*: simple and elegant method using attention for handling multi-instance learning in distant supervision

### Self-Training for Jointly Learning to Ask and Answer Questions
_Mrinmaya Sachan and Eric Xing_

Building curious machines that can answer as well as ask questions is an important challenge for AI. The two tasks of question answering and question generation are usually tackled separately in the NLP literature. At the same time, both require significant amounts of supervised data which is hard to obtain in many do- mains. To alleviate these issues, we propose a self-training method for jointly learning to ask as well as answer questions, leveraging unlabeled text along with labeled question answer pairs for learning. We evaluate our approach on four benchmark datasets: SQUAD, MS MARCO, WikiQA and TrecQA, and show significant im- provements over a number of established baselines on both question answering and question generation tasks. We also achieved new state-of-the-art results on two competitive answer sentence selection tasks: WikiQA and TrecQA.

*Mihai's comments*: interesting work on how to generate artificial questions and answers, and how to control for their quality

### The Web as a Knowledge-base for Answering Complex Questions
_Alon Talmor and Jonathan Berant_

Answering complex questions is a time-consuming activity for humans that requires reasoning and integration of information. Recent work on reading comprehension made headway in answering simple questions, but tackling complex questions is still an ongoing research challenge. Conversely, semantic parsers have been successful at handling compositionality, but only when the information resides in a target knowledge-base. In this paper, we present a novel framework for answering broad and complex questions, assuming answering simple questions is possible using a search engine and a reading comprehension model. We propose to decom- pose complex questions into a sequence of simple questions, and compute the final answer from the sequence of answers. To illustrate the viability of our approach, we create a new dataset of complex questions, Com- plexWebQuestions, and present a model that decomposes questions and interacts with the web to compute an answer. We empirically demonstrate that question decomposition improves performance from 20.8 precision1 to 27.5 precision1 on this new dataset.

*Mihai's comments*: They discuss a new dataset and approach to answer complex questions that can be decomposed into a combination of simpler questions. These questions were artificially generated from a KB. I wonder how close they are to naturally-occuring questions that a person would ask?

### Strong Baselines for Simple Question Answering over Knowledge Graphs with and without Neural Networks
_Salman Mohammed, Peng Shi, and Jimmy Lin_

We examine the problem of question answering over knowledge graphs, focusing on simple questions that can be answered by the lookup of a single fact. Adopting a straightforward decomposition of the problem into entity detection, entity linking, relation prediction, and evidence combination, we explore simple yet strong baselines. On the popular SimpleQuestions dataset, we find that basic LSTMs and GRUs plus a few heuristics yield accuracies that approach the state of the art, and techniques that do not use neural networks also perform reasonably well. These results show that gains from sophisticated deep learning techniques proposed in the literature are quite modest and that some previous models exhibit unnecessary complexity.

*Mihai's comments*: What the title says. Good to see robust baselines being published!

### Questionable Answers in Question Answering Research: Reproducibility and Variability of Published Results (TACL)
_Matt Crane_

Based on theoretical reasoning it has been suggested that the reliability of findings published in the scientific literature decreases with the popularity of a research field” (Pfeiffer and Hoffmann, 2009). As we know, deep learning is very popular and the ability to reproduce results is an important part of science. There is growing concern within the deep learning community about the reproducibility of results that are presented. In this paper we present a number of controllable, yet unreported, effects that can substantially change the effective- ness of a sample model, and thusly the reproducibility of those results. Through these environmental effects we show that the commonly held belief that distribution of source code is all that is needed for reproducibility is not enough. Source code without a reproducible environment does not mean anything at all. In addition the range of results produced from these effects can be larger than the majority of incremental improvement reported.

*Mihai's comments*: explains why DL results vary so much, for the same approach. Explanations range from differences in NN library versions to different hardware.

## DAY 2

### Keynote talk by Mari Ostendorf on the journey to build a "chitchat" conversational agent for the Alexa Challenge
*Mihai's comments*: seq2seq models don't work for this task. 

## Neural Models of Factuality
_Rachel Rudinger, Aaron Steven White, and Benjamin Van Durme_

We present two neural models for event factuality prediction, which yield significant performance gains over
previous models on three event factuality datasets: FactBank, UW, and MEANTIME. We also present a sub- stantial expansion of the It Happened portion of the Universal Decompositional Semantics dataset, yielding the largest event factuality dataset to date. We report model results on this extended factuality dataset as well.

*Mihai's comments*: this is NOT fake news detection, but a simpler task to detect if an event happened or not. For example: if negated did not happen; if future tense it did not happen. Such a component should be included in most IE systems...

### FEVER: a Large-scale Dataset for Fact Extraction and VERification
_James Thorne, Andreas Vlachos, Christos Christodoulopoulos, and Arpit Mittal_

In this paper we introduce a new publicly available dataset for verification against textual sources, FEVER: Fact Extraction and VERification. It consists of 185,445 claims generated by altering sentences extracted from Wikipedia and subsequently verified without knowledge of the sentence they were derived from. The claims are classified as Supported, Refuted or NotEnoughInfo by annotators achieving 0.6841 in Fleiss kappa. For the first two classes, the annotators also recorded the sentence(s) forming the necessary evidence for their judgment. To characterize the challenge of the dataset presented, we develop a pipeline approach and compare it to suitably designed oracles. The best accuracy we achieve on labeling a claim accompanied by the correct evidence is 31.87%, while if we ignore the evidence we achieve 50.91%. Thus we believe that FEVER is a challenging testbed that will help stimulate progress on claim verification against textual sources.

*Mihai's comments*: Cool and important task

### A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference
This paper introduces the Multi-Genre Natural Language Inference (MultiNLI) corpus, a dataset designed for use in the development and evaluation of machine learning models for sentence understanding. At 433k examples, this resource is one of the largest corpora available for natural language inference (a.k.a. recog- nizing textual entailment), improving upon available resources in both its coverage and difficulty. MultiNLI accomplishes this by offering data from ten distinct genres of written and spoken English, making it possi- ble to evaluate systems on nearly the full complexity of the language, while supplying an explicit setting for evaluating cross-genre domain adaptation. In addition, an evaluation using existing machine learning models designed for the Stanford NLI corpus shows that it represents a substantially more difficult task than does that corpus, despite the two showing similar levels of inter-annotator agreement.

*Mihai's comments*: A newer, multi-genre version of SNLI. Might be useful for multi-hop QA.

### Keep Your Bearings: Lightly-Supervised Information Extraction with Ladder Networks That Avoids Semantic Drift
_Ajay Nagesh and Mihai Surdeanu_

We propose a novel approach to semi- supervised learning for information extraction that uses ladder networks (Rasmus et al., 2015). In particular, we focus on the task of named entity classification, defined as identifying the correct label (e.g., person or organization name) of an entity mention in a given context. Our approach is simple, efficient and has the benefit of being robust to semantic drift, a dominant problem in most semi- supervised learning systems. We empirically demonstrate the superior performance of our system compared to the state-of-the-art on two standard datasets for named entity classification. We obtain between 62% and 200% improvement over the state-of-art baseline on these two datasets.

*Mihai's comments*: self promotion. One-shot learning is better than iterative bootstrapping.

## DAY 3



### A Neural Layered Model for Nested Named Entity Recognition
_Meizhi Ju, Makoto Miwa, Sophia Ananiadou_

Entity mentions embedded in longer entity mentions are referred to as nested entities. Most named entity recognition (NER) systems deal only with the flat entities and ignore the inner nested ones, which fails to capture finer-grained semantic information in underlying texts. To address this issue, we propose a novel neural model to identify nested entities by dynamically stacking flat NER layers. Each flat NER layer is based on the state-of-the-art flat NER model that captures sequential context representation with bidirectional Long Short-Term Memory (LSTM) layer and feeds it to the cascaded CRF layer. Our model merges the output of the LSTM layer in the current flat NER layer to build new representation for detected entities and subsequently feeds them into the next flat NER layer. This allows our model to extract outer entities by taking full advantage of information encoded in their corresponding inner entities, in an inside-to-outside way. Our model dynamically stacks the flat NER layers until no outer entities are extracted. Extensive evaluation shows that our dynamic model outperforms state-of-the-art feature-based systems on nested NER, achieving 74.7% and 72.2% on GENIA and ACE2005 datasets, respectively, in terms of F-score.

*Mihai’s comments*: This is a beautifully simple take on nested NER: just recursively apply the same LSTM on the output of the previous one. Very easy to implement, good results.

### KBGAN: Adversarial Learning for Knowledge Graph Embeddings
_Liwei Cai and William Yang Wang_

We introduce KBGAN, an adversarial learning framework to improve the performances of a wide range of existing knowledge graph embedding models. Because knowledge graphs typically only contain positive facts, sampling useful negative training examples is a nontrivial task. Replacing the head or tail entity of a fact with a uniformly randomly selected entity is a conventional method for generating negative facts, but the major- ity of the generated negative facts can be easily discriminated from positive facts, and will contribute little towards the training. Inspired by generative adversarial networks (GANs), we use one knowledge graph em- bedding model as a negative sample generator to assist the training of our desired model, which acts as the discriminator in GANs. This framework is independent of the concrete form of generator and discriminator, and therefore can utilize a wide variety of knowledge graph embedding models as its building blocks. In ex- periments, we adversarially train two translation-based models, TRANSE and TRANSD, each with assistance from one of the two probability-based models, DISTMULT and COMPLEX. We evaluate the performances of KBGAN on the link prediction task, using three knowledge base completion datasets: FB15k-237, WN18 and WN18RR. Experimental results show that adversarial training substantially improves the performances of target embedding models under various settings.

*Mihai’s comments*: A GAN approach to choosing appropriate negative examples for bilinear relation extraction models. Cool. But how does this handle false negatives?

### Learning Joint Semantic Parsers from Disjoint Data
_Hao Peng, Sam Thomson, Swabha Swayamdipta, and Noah A. Smith_

We present a new approach to learning a semantic parser from multiple datasets, even when the target semantic formalisms are drastically different and the underlying corpora do not overlap. We handle such “disjoint” data by treating annotations for unobserved formalisms as latent structured variables. Building on state-of-the- art baselines, we show improvements both in frame-semantic parsing and semantic dependency parsing by modeling them jointly.

*Mihai’s comments*: multi-task learning for multiple semantic formalisms (PropBank + FrameNet). Instead of the usual sharing of parameters, they focus on joint decoding for the two tasks.

### Variational Knowledge Graph Reasoning
_Wenhu Chen, Wenhan Xiong, Xifeng Yan, and William Yang Wang_

Inferring missing links in knowledge graphs (KG) has attracted a lot of attention from the research community. In this paper, we tackle a practical query answering task involving predicting the relation of a given entity pair. We frame this prediction problem as an inference problem in a probabilistic graphical model and aim at resolving it from a variational inference perspective. In order to model the relation between the query entity pair, we assume that there exists an underlying latent variable (paths connecting two nodes) in the KG, which carries the equivalent semantics of their relations. However, due to the intractability of connections in large KGs, we propose to use variation inference to maximize the evidence lower bound. More specifically, our framework (Diva) is composed of three modules, i.e. a posterior approximator, a prior (path finder), and a likelihood (path reasoner). By using variational inference, we are able to incorporate them closely into a unified architecture and jointly optimize them to perform KG reasoning. With active interactions among these sub-modules, DIVA is better at handling noise and coping with more complex reasoning scenarios. In order to evaluate our method, we conduct the experiment of the link prediction task on multiple datasets and achieve state-of-the-art performances on both datasets.

*Mihai’s comments*: a variational inference take on the KB completion task for *compositional* approaches such as Cohen’s PRA. 

### Linguistic Cues to Deception and Perceived Deception in Interview Dialogues
_Sarah Ita Levitan, Angel Maredia, and Julia Hirschberg_

We explore deception detection in interview dialogues. We analyze a set of linguistic features in both truthful and deceptive responses to interview questions. We also study the perception of deception, identifying characteristics of statements that are perceived as truthful or deceptive by interviewers. Our analysis show significant differences between truthful and deceptive question responses, as well as variations in deception patterns across gender and native language. This analysis motivated our selection of features for machine learning experiments aimed at classifying globally deceptive speech. Our best classification performance is 72.74% F1-Score (about 17% better than human performance), which is achieved using a combination of linguistic features and individual traits.

*Mihai’s comments*: Several interesting linguistic cues that indicate deception. This might be applicable to the identification of fake news?

### Non-Projective Dependency Parsing with Non-Local Transitions
_Daniel Fernández-González and Carlos Gómez-Rodríguez_

We present a novel transition system, based on the Covington non-projective parser, introducing non-local
transitions that can directly create arcs involving nodes to the left of the current focus positions. This avoids the need for long sequences of No-Arcs transitions to create long-distance arcs, thus alleviating error propa- gation. The resulting parser outperforms the original version and achieves the best accuracy on the Stanford Dependencies conversion of the Penn Treebank among greedy transition-based parsers.

*Mihai’s comments*: a variant of the Covington algorithm, which is allowed to attach tokens at distance k from the current focus. Works slightly better than the latest Stanford parser. This indeed reduces the number of transition actions needed, but increases the action space…

### Deep Contextualized Word Representations
_Matthew Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark, Kenton Lee, and Luke Zettlemoyer_

We introduce a new type of deep contextualized word representation that models both (1) complex characteristics of word use (e.g., syntax and semantics), and (2) how these uses vary across linguistic contexts (i.e., to model polysemy). Our word vectors are learned functions of the internal states of a deep bidirectional language model (biLM), which is pre-trained on a large text corpus. We show that these representations can be easily added to existing models and significantly improve the state of the art across six challenging NLP problems, including question answering, textual entailment and sentiment analysis. We also present an analysis showing that exposing the deep internals of the pre-trained network is crucial, allowing downstream models to mix different types of semi-supervision signals.

*Mihai’s comments*: ELMO embeddings. Use them.

### Learning to Map Context-Dependent Sentences to Executable Formal Queries
_Alane Suhr, Srinivasan Iyer, and Yoav Artzi_

We propose a context-dependent model to map utterances within an interaction to executable formal queries.
To incorporate interaction history, the model maintains an interaction-level encoder that updates after each turn, and can copy sub-sequences of previously predicted queries during generation. Our approach combines implicit and explicit modeling of references between utterances. We evaluate our model on the ATIS flight planning interactions, and demonstrate the benefits of modeling context and explicit references.

*Mihai’s comments*: Start here if you are interested in seq2seq models for dialog. However, I wonder about the generality of these approaches on more complicated dialog tasks (see Mari Ostendorf’s invited talk).




