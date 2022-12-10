# Mihai's Summary of EMNLP 2022

### GREENER: Graph Neural Networks for News Media Profiling
**URL**: https://arxiv.org/abs/2211.05533

**Abstract**: We study the problem of profiling news media on the Web with respect to their factuality of reporting and bias. This is an important but under-studied problem related to disinformation and "fake news" detection, but it addresses the issue at a coarser granularity compared to looking at an individual article or an individual claim. This is useful as it allows to profile entire media outlets in advance. Unlike previous work, which has focused primarily on text (e.g.,~on the text of the articles published by the target website, or on the textual description in their social media profiles or in Wikipedia), here our main focus is on modeling the similarity between media outlets based on the overlap of their audience. This is motivated by homophily considerations, i.e.,~the tendency of people to have connections to people with similar interests, which we extend to media, hypothesizing that similar types of media would be read by similar kinds of users. In particular, we propose GREENER (GRaph nEural nEtwork for News mEdia pRofiling), a model that builds a graph of inter-media connections based on their audience overlap, and then uses graph neural networks to represent each medium. We find that such representations are quite useful for predicting the factuality and the bias of news media outlets, yielding improvements over state-of-the-art results reported on two datasets. When augmented with conventionally used representations obtained from news articles, Twitter, YouTube, Facebook, and Wikipedia, prediction accuracy is found to improve by 2.5-27 macro-F1 points for the two tasks.

**Mihai's comments**: Modeling fact verification as a graph-based tsk helps for fact verification

### UnifiedSKG: Unifying and Multi-Tasking Structured Knowledge Grounding with Text-to-Text Language Models
**Mihai's comments**: An unified architecture for 6 tasks that work on top of knowledge bases. Each task has a unique prefix, but they share the same LM underneath. This improves performance over models fine-tuned individually. GPT-3 + Codex and few-shot learning performs worse than fine-tuning a LM. But careful prompt design brings GPT-3 at the same level as a fine-tuned T5.

### Reasoning Like Program Executors
**Mihai's comments**: Use the output of programs to pre-train LMs. That is, they: (a) generate natural language forms of programs (e.g., describing math operations in NL), and (b) pre-train LMs to mimic the output of these programs when exposed to the NL representations. This improves results for math programs, logic programs, SQL. 
(This is relevant for Zhengzhong.)

### DocInfer: Document-level Natural Language Inference using Optimal Evidence Selection
**Mihai's comments**: Addresses document-level NLI, where both premises and hypotheses are entire documents. The key observation is that they use an attention mechanism to select paragraphs from the premise that are most relevant to the hypothesis. Then they use LSTM + REINFORCE to select the best evidence sentences from these paragraphs. Interestingly, they use 4 rewards functions. 
(This is relevant to Alice.)

### Infinite SCAN: An Infinite Model of Diachronic Semantic Change
**Mihai's comments**: They model word senses using topic models. They do it over time to track word meaning changes in time. They estimate the number of senses per word using Dirichlet processes. 

### Measuring Context-Word Biases in Lexical Semantic Datasets
**Mihai's comments**: Discusses a series of tasks to evaluate and analyze word sense disambiguation in context.

### Unobserved Local Structures Make Compositional Generalization Hard
**Mihai's comments**: NLP models struggle to generalize for compositionality (in the context of semantic parsing). Their key observation is that models do not generalize on test instances that contain local structures that do not appear in training data.
(Relevant to Sushma and Haris)

### True Few-Shot Learning With Prompts - A Real-World Perspective
**Mihai's comments**: Prompts are brittle, especially in real-world tasks. They do an ensemble of prompts with a meta-classifier on top to produce the final label.
(Relevant to Mahdi and Zheng)

### Generate, Annotate, and Learn: NLP with Synthetic Text
**Mihai's comments**: They generate unlabeled data *in-domain* using a nearest neighbor search. Then they self train on this data using a teacher-student model. Performs better on GLUE than many other SSL approaches.

### Structural Persistence in Language Models: Priming as a Window into Abstract Language Representations
**Mihai's comments**: They find that LMs do respond to structural priming fairly similar to humans. 
(Relevant to Sushma and Haris)

### ProoFVer: Natural Logic Theorem Proving for Fact Verification
**URL:** https://arxiv.org/abs/2108.11357

**Mihai's comments**: A logic prover for fact verification using natural logic. Very cool paper! They generate training data for natural logic by chunking and aligning fragments of premise and hypothesis using a series of linguistic heuristics. They use Wiki Data to insert world knowledge in the system. The performance on FEVER is very good.
(Very relevant to Sushma and Haris)

### Investigating Reasons for Disagreement in Natural Language Inference
**Mihai's comments**: Disagreements in NLI annotations are not necessarily bad. They create a taxonomy that captures systematic disagreements in annotations in MNLI. Multilabel classification can better capture different interpretations.

### Diff-Explainer: Differentiable Convex Optimization for Explainable Multi-hop Inference
**Mihai's comments**: A fully differentiable approach for multi-hop QA that is (partially) inspired by Vikas's AutoROCC! They approximate TupleILP using SDP. The results show minor but consistent improvements over AutoROCC on ARC and World Tree.
(Relevant to Pere J, Vikas)

### Meta-Learning the Difference: Preparing Large Language Models for Efficient Adaptation
**Mihai's comments**: This paper addresses the domain adaptation of LMs during fine-tuning. They propose several improvements for meta-learning, MAML in particular, which make it overfit less. They should some small improvements for summarization over pre-training + fine-tuning.
(Relevant to Shahriar)

### Learning Fair Representations via Rate-Distortion Maximization
**Mihai's comments**: They introduce gender balance in LMs without a drop in accuracy for downstream tasks.

### COCO-DR: Combating the Distribution Shift in Zero-Shot Dense Retrieval with Contrastive and Distributionally Robust Learning
**Mihai's comments**: Improves dense retrieval by continuining pre-training in the target domain. I did not read this paper yet.
(Relevant to Mihai)

### 
**Mihai's comments**:







