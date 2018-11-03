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

This workshop is fantastic! Imo, many of the papers here are nicer than many conference publications.
Proceedings for this workshop are here: http://aclweb.org/anthology/W18-5400.
(Zheng: you should read all these papers)

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


# EMNLP Day 1

### Reasoning about Actions and State Changes by Injecting Commonsense Knowledge

**URL**: http://aclweb.org/anthology/D18-1006

**Abstract**: Comprehending procedural text, e.g., a para- graph describing photosynthesis, requires modeling actions and the state changes they produce, so that questions about entities at dif- ferent timepoints can be answered. Although several recent systems have shown impressive progress in this task, their predictions can be globally inconsistent or highly improbable. In this paper, we show how the predicted effects of actions in the context of a paragraph can be improved in two ways: (1) by incorporat- ing global, commonsense constraints (e.g., a non-existent entity cannot be destroyed), and (2) by biasing reading with preferences from large-scale corpora (e.g., trees rarely move). Unlike earlier methods, we treat the problem as a neural structured prediction task, allow- ing hard and soft constraints to steer the model away from unlikely predictions. We show that the new model significantly outperforms ear- lier systems on a benchmark dataset for proce- dural text comprehension (+8% relative gain), and that it also avoids some of the nonsensical predictions that earlier systems make.

**Mihai's comments**: QA for process questions, i.e., where the answer is a sequence of actions. Treated as structured prediction, where the search space is pruned with commonsense knowledge, which, in turn, is extracted from large corpora. The approach (Figure 3) is akin to memory networks over the whole process. (Zhengzhong: read this)

### Collecting Diverse Natural Language Inference Problems for Sentence Representation 

**URL**: http://aclweb.org/anthology/D18-1007

**Abstract**: We present a large scale collection of diverse natural language inference (NLI) datasets that help provide insight into how well a sentence representation captures distinct types of rea- soning. The collection results from recasting 13 existing datasets from 7 semantic phenom- ena into a common NLI structure, resulting in over half a million labeled context-hypothesis pairs in total. We refer to our collection as the DNC: Diverse Natural Language Inference Collection. The DNC is available online at http://www.decomp.net, and will grow over time as additional resources are recast and added from novel sources.

**Mihai's comments**: A diverse dataset for NLI. Read and use this if you work on NLI.

### Phrase-Indexed Question Answering: A New Challenge for Scalable Document Comprehension

**URL**: http://aclweb.org/anthology/D18-1052

**Abstract**: 
We formalize a new modular variant of current question answering tasks by enforcing com- plete independence of the document encoder from the question encoder. This formulation addresses a key challenge in machine compre- hension by requiring a standalone representa- tion of the document discourse. It addition- ally leads to a significant scalability advantage since the encoding of the answer candidate phrases in the document can be pre-computed and indexed offline for efficient retrieval. We experiment with baseline models for the new task, which achieve a reasonable accuracy but significantly underperform unconstrained QA models. We invite the QA research commu- nity to engage in Phrase-Indexed Question An- swering (PIQA, pika) for closing the gap. The leaderboard is at: nlp.cs.washington. edu/piqa

**Mihai's comments**: indexes phrases (just NPs and NEs?) rather than documents. Then generate an encoding of each phrase, and retrieve good answers using nearest neighbors to the question vector. This is a neat paper, even if the performance is not great yet. (Vikas: read this.)

### Ranking Paragraphs for Improving Answer Recall in Open-Domain Question Answering

**URL**: http://aclweb.org/anthology/D18-1053

**Abstract**: Recently, open-domain question answering (QA) has been combined with machine com- prehension models to find answers in a large knowledge source. As open-domain QA re- quires retrieving relevant documents from text corpora to answer questions, its performance largely depends on the performance of doc- ument retrievers. However, since traditional information retrieval systems are not effective in obtaining documents with a high probabil- ity of containing answers, they lower the per- formance of QA systems. Simply extracting more documents increases the number of ir- relevant documents, which also degrades the performance of QA systems. In this paper, we introduce Paragraph Ranker which ranks para- graphs of retrieved documents for a higher an- swer recall with less noise. We show that rank- ing paragraphs and aggregating answers us- ing Paragraph Ranker improves performance of open-domain QA pipeline on the four open- domain QA datasets by 7.8% on average.

**Mihai's comments**: trains a paragraph ranker (PR) jointly with the QA system. Limitations: PR is supervised; focuses on simple factoid questions only (how well does it work on complex QA?); focuses on boosting recall during PR (should we focus on F1)? (Vikas: read this)

### Adaptive Document Retrieval for Deep Question Answering

**URL**: http://aclweb.org/anthology/D18-1055

**Abstract**: State-of-the-art systems in deep question an- swering proceed as follows: (1) an initial document retrieval selects relevant documents, which (2) are then processed by a neural net- work in order to extract the final answer. Yet the exact interplay between both compo- nents is poorly understood, especially con- cerning the number of candidate documents that should be retrieved. We show that choos- ing a static number of documents – as used in prior research – suffers from a noise- information trade-off and yields suboptimal results. As a remedy, we propose an adaptive document retrieval model. This learns the opti- mal candidate number for document retrieval, conditional on the size of the corpus and the query. We report extensive experimental re- sults showing that our adaptive approach out- performs state-of-the-art methods on multiple benchmark datasets, as well as in the context of corpora with variable sizes.

**Mihai's comments**: makes the number of documents retrieved for QA dependent on the confidence we have in the IR system for this query (the more confident, the fewer documents). Nice, but it uses the IR system as a blackbox... (Vikas: read this)

### A Deep Neural Network Sentence Level Classification Method with Context Information

**URL**: http://aclweb.org/anthology/D18-1107

**Abstract**: In the sentence classification task, context formed from sentences adjacent to the sen- tence being classified can provide important information for classification. This context is, however, often ignored. Where methods do make use of context, only small amounts are considered, making it difficult to scale. We present a new method for sentence classifica- tion, Context-LSTM-CNN, that makes use of potentially large contexts. The method also utilizes long-range dependencies within the sentence being classified, using an LSTM, and short-span features, using a stacked CNN. Our experiments demonstrate that this approach consistently improves over previous methods on two different datasets.

**Mihai's comments**: An efficient way to encode large contexts (e.g., for sentence classification, the entire document around the sentence). Efficient encoding with FOFE. Read this if you need large context for classification.

### RESIDE: Improving Distantly-Supervised Neural Relation Extraction using Side Information

**URL**: http://aclweb.org/anthology/D18-1157

**Abstract**: Distantly-supervised Relation Extraction (RE) methods train an extractor by automatically aligning relation instances in a Knowledge Base (KB) with unstructured text. In addi- tion to relation instances, KBs often contain other relevant side information, such as aliases of relations (e.g., founded and co-founded are aliases for the relation founderOfCompany). RE models usually ignore such readily avail- able side information. In this paper, we pro- pose RESIDE, a distantly-supervised neural relation extraction method which utilizes ad- ditional side information from KBs for im- proved relation extraction. It uses entity type and relation alias information for imposing soft constraints while predicting relations. RE- SIDE employs Graph Convolution Networks (GCN) to encode syntactic information from text and improves performance even when limited side information is available. Through extensive experiments on benchmark datasets, we demonstrate RESIDE’s effectiveness. We have made RESIDE’s source code available to encourage reproducible research.

**Mihai's comments**: New state-of-the-art for distantly supervised RE. Uses relation aliases from the KB + an architecture that concatenates PCNN with a graph convolutional network (GCN) (Fan: read this)

# EMNLP Day 2

### QuAC : Question Answering in Context

**URL**: http://aclweb.org/anthology/D18-1241

**Abstract**: We present QuAC, a dataset for Question
Answering in Context that contains 14K information-seeking QA dialogs (100K ques- tions in total). The dialogs involve two crowd workers: (1) a student who poses a sequence of freeform questions to learn as much as pos- sible about a hidden Wikipedia text, and (2) a teacher who answers the questions by pro- viding short excerpts from the text. QuAC in- troduces challenges not found in existing ma- chine comprehension datasets: its questions are often more open-ended, unanswerable, or only meaningful within the dialog context, as we show in a detailed qualitative evaluation. We also report results for a number of ref- erence models, including a recently state-of- the-art reading comprehension architecture ex- tended to model dialog context. Our best model underperforms humans by 20 F1, sug- gesting that there is significant room for fu- ture work on this data. Dataset, baseline, and leaderboard available at http://quac.ai.

**Mihai's comments**: New dataset for QA in dialog. (Fan: read this)

### Learning Scalar Adjective Intensity from Paraphrases

**URL**: http://aclweb.org/anthology/D18-1202

**Abstract**: Adjectives like warm, hot, and scalding all de- scribe temperature but differ in intensity. Un- derstanding these differences between adjec- tives is a necessary part of reasoning about nat- ural language. We propose a new paraphrase- based method to automatically learn the rela- tive intensity relation that holds between a pair of scalar adjectives. Our approach analyzes over 36k adjectival pairs from the Paraphrase Database under the assumption that, for exam- ple, paraphrase pair really hot ↔ scalding sug- gests that hot < scalding. We show that com- bining this paraphrase evidence with existing, complementary pattern- and lexicon-based ap- proaches improves the quality of systems for automatically ordering sets of scalar adjectives and inferring the polarity of indirect answers to yes/no questions.

**Mihai's comments**: learns ranking of adjectives from paraphrases. (Mithun: read this)

### MemoReader: Large-Scale Reading Comprehension through Neural Memory Controller

**URL**: http://aclweb.org/anthology/D18-1237

**Abstract**: Machine reading comprehension helps ma- chines learn to utilize most of the human knowledge written in the form of text. Existing approaches made a significant progress com- parable to human-level performance, but they are still limited in understanding, up to a few paragraphs, failing to properly comprehend lengthy document. In this paper, we propose a novel deep neural network architecture to han- dle a long-range dependency in RC tasks. In detail, our method has two novel aspects: (1) an advanced memory-augmented architecture and (2) an expanded gated recurrent unit with dense connections that mitigate potential in- formation distortion occurring in the memory. Our proposed architecture is widely applicable to other models. We have performed exten- sive experiments with well-known benchmark datasets such as TriviaQA, QUASAR-T, and SQuAD. The experimental results demonstrate that the proposed method outperforms existing methods, especially for lengthy documents.

**Mihai's comments**: Extension of memory networks for reading comprehension. (Zhengzhong: read this)

### Cross-Pair Text Representations for Answer Sentence Selection

**URL**: http://aclweb.org/anthology/D18-1240

**Abstract**: High-level semantics tasks, e.g., paraphras- ing, textual entailment or question answer- ing, involve modeling of text pairs. Before the emergence of neural networks, this has been mostly performed using intra-pair fea- tures, which incorporate similarity scores or rewrite rules computed between the members within the same pair. In this paper, we com- pute scalar products between vectors repre- senting similarity between members of differ- ent pairs, in place of simply using a single vector for each pair. This allows us to obtain a representation specific to any pair of pairs, which delivers the state of the art in answer sentence selection. Most importantly, our ap- proach can outperform much more complex algorithms based on neural networks.

**Mihai's comments**: Kernels continue to work very well for QA

### N-ary Relation Extraction using Graph State LSTM

**URL**: http://aclweb.org/anthology/D18-1246

**Abstract**: Cross-sentence n-ary relation extraction de- tects relations among n entities across multi- ple sentences. Typical methods formulate an input as a document graph, integrating vari- ous intra-sentential and inter-sentential depen- dencies. The current state-of-the-art method splits the input graph into two DAGs, adopt- ing a DAG-structured LSTM for each. Though being able to model rich linguistic knowledge by leveraging graph edges, important infor- mation can be lost in the splitting procedure. We propose a graph-state LSTM model, which uses a parallel state to model each word, recur- rently enriching state values via message pass- ing. Compared with DAG LSTMs, our graph LSTM keeps the original graph structure, and speeds up computation by allowing more par- allelization. On a standard benchmark, our model shows the best result in the literature.

**Mihai's comments**: A very nice take on graph LSTMs, which follows each word in a sentence over time. Application to n-ary relation extraction.

### Large-scale Exploration of Neural Relation Classification Architectures

**URL**: http://aclweb.org/anthology/D18-1250

**Abstract**: Experimental performance on the task of rela- tion classification has generally improved us- ing deep neural network architectures. One major drawback of reported studies is that individual models have been evaluated on a very narrow range of datasets, raising ques- tions about the adaptability of the architec- tures, while making comparisons between ap- proaches difficult. In this work, we present a systematic large-scale analysis of neural rela- tion classification architectures on six bench- mark datasets with widely varying characteris- tics. We propose a novel multi-channel LSTM model combined with a CNN that takes ad- vantage of all currently popular linguistic and architectural features. Our ‘Man for All Sea- sons’ approach achieves state-of-the-art per- formance on two datasets. More importantly, in our view, the model allowed us to obtain direct insights into the continued challenges faced by neural language models on this task. Example data and source code are available at: https://github.com/aidantee/ MASS.

**Mihai's comments**: What the title says. What works: fancy word embeddings that combine characters, WordNet info, POS tags, and Fastext; the usual: position embeddings, biLSTMs, CNNs. 

### Can a Suit of Armor Conduct Electricity? A New Dataset for Open Book Question Answering

**URL**: http://aclweb.org/anthology/D18-1260

**Abstract**: We present a new kind of question answering dataset, OpenBookQA, modeled after open book exams for assessing human understand- ing of a subject. The open book that comes with our questions is a set of 1326 elementary level science facts. Roughly 6000 questions probe an understanding of these facts and their application to novel situations. This requires combining an open book fact (e.g., metals con- duct electricity) with broad common knowl- edge (e.g., a suit of armor is made of metal) ob- tained from other sources. While existing QA datasets over documents or knowledge bases, being generally self-contained, focus on lin- guistic understanding, OpenBookQA probes a deeper understanding of both the topic—in the context of common knowledge—and the lan- guage it is expressed in. Human performance on OpenBookQA is close to 92%, but many state-of-the-art pre-trained QA methods per- form surprisingly poorly, worse than several simple neural baselines we develop. Our or- acle experiments designed to circumvent the knowledge retrieval bottleneck demonstrate the value of both the open book and additional facts. We leave it as a challenge to solve the retrieval problem in this multi-hop setting and to close the large gap to human performance.

**Mihai's comments**: A nice dataset for language inference over free text. (Zhengzhong: read this)

### Title

**URL**: 

**Abstract**: 

**Mihai's comments**: 


