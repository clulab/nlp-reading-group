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
