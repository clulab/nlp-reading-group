# Enrique's summary of NAACL-HLT 2022

## Papers

### An Enhanced Span-based Decomposition Method for Few-Shot Sequence Labeling.
Few-Shot Sequence Labeling (FSSL) is a canonical paradigm for the tagging models, e.g., named entity recognition and slot filling, to generalize on an emerging, resource-scarce domain. Recently, the metric-based meta-learning framework has been recognized as a promising approach for FSSL. However, most prior works assign a label to each token based on the token-level similarities, which ignores the integrality of named entities or slots. To this end, in this paper, we propose ESD, an Enhanced Span-based Decomposition method for FSSL. ESD formulates FSSL as a span-level matching problem between test query and supporting instances. Specifically, ESD decomposes the span matching problem into a series of span-level procedures, mainly including enhanced span representation, class prototype aggregation and span conflicts resolution. Extensive experiments show that ESD achieves the new state-of-the-art results on two popular FSSL benchmarks, FewNERD and SNIPS, and is proven to be more robust in the noisy and nested tagging scenarios.

*Link:* https://aclanthology.org/2022.naacl-main.369/

---

### Modeling Multi-Granularity Hierarchical Features for Relation Extraction
Relation extraction is a key task in Natural Language Processing (NLP), which aims to extract relations between entity pairs from given texts. Recently, relation extraction (RE) has achieved remarkable progress with the development of deep neural networks. Most existing research focuses on constructing explicit structured features using external knowledge such as knowledge graph and dependency tree. In this paper, we propose a novel method to extract multi-granularity features based solely on the original input sentences. We show that effective structured features can be attained even without external knowledge. Three kinds of features based on the input sentences are fully exploited, which are in entity mention level, segment level, and sentence level. All the three are jointly and hierarchically modeled. We evaluate our method on three public benchmarks: SemEval 2010 Task 8, Tacred, and Tacred Revisited. To verify the effectiveness, we apply our method to different encoders such as LSTM and BERT. Experimental results show that our method significantly outperforms existing state-of-the-art models that even use external knowledge. Extensive analyses demonstrate that the performance of our model is contributed by the capture of multi-granularity features and the model of their hierarchical structure.

*Link*: https://aclanthology.org/2022.naacl-main.375/

---

### Contrastive Representation Learning for Cross-Document Coreference Resolution of Events and Entities
Identifying related entities and events within and across documents is fundamental to natural language understanding. We present an approach to entity and event coreference resolution utilizing contrastive representation learning. Earlier state-of-the-art methods have formulated this problem as a binary classification problem and leveraged large transformers in a cross-encoder architecture to achieve their results. For large collections of documents and corresponding set of n mentions, the necessity of performing n2 transformer computations in these earlier approaches can be computationally intensive. We show that it is possible to reduce this burden by applying contrastive learning techniques that only require n transformer computations at inference time. Our method achieves state-of-the-art results on a number of key metrics on the ECB+ corpus and is competitive on others.

*Link*: https://aclanthology.org/2022.naacl-main.267/

___

### Sparse Distillation: Speeding Up Text Classification by Using Bigger Student Models
Distilling state-of-the-art transformer models into lightweight student models is an effective way to reduce computation cost at inference time. The student models are typically compact transformers with fewer parameters, while expensive operations such as self-attention persist. Therefore, the improved inference speed may still be unsatisfactory for real-time or high-volume use cases. In this paper, we aim to further push the limit of inference speed by distilling teacher models into bigger, sparser student models – bigger in that they scale up to billions of parameters; sparser in that most of the model parameters are n-gram embeddings. Our experiments on six single-sentence text classification tasks show that these student models retain 97% of the RoBERTa-Large teacher performance on average, and meanwhile achieve up to 600x speed-up on both GPUs and CPUs at inference time. Further investigation reveals that our pipeline is also helpful for sentence-pair classification tasks, and in domain generalization settings.

*Enrique's note*: This paper proposes counter-intuitively to train student models with more parameters than the teacher and achieve speed-ups an order of magnitude greater than those achieved by smaller students

*Link*: https://aclanthology.org/2022.naacl-main.169/

___

### FNet: Mixing Tokens with Fourier Transforms
We show that Transformer encoder architectures can be sped up, with limited accuracy costs, by replacing the self-attention sublayers with simple linear transformations that “mix” input tokens. Most surprisingly, we find that replacing the self-attention sublayer in a Transformer encoder with a standard, unparameterized Fourier Transform achieves 92-97% of the accuracy of BERT counterparts on the GLUE benchmark, but trains 80% faster on GPUs and 70% faster on TPUs at standard 512 input lengths. At longer input lengths, our FNet model is significantly faster: when compared to the “efficient Transformers” on the Long Range Arena benchmark, FNet matches the accuracy of the most accurate models, while outpacing the fastest models across all sequence lengths on GPUs (and across relatively shorter lengths on TPUs). Finally, FNet has a light memory footprint and is particularly efficient at smaller model sizes; for a fixed speed and accuracy budget, small FNet models outperform Transformer counterparts.

*Enrique's note*: This paper looks interesting because it replaces self-attention with a FFN, which doesn't have parameters to optimize and has $O(N log N)$ complexity insteas of $N^2$

*Link*: https://aclanthology.org/2022.naacl-main.319/

---

### When Does Syntax Mediate Neural Language Model Performance?
Recent causal probing literature reveals when language models and syntactic probes use similar representations. Such techniques may yield “false negative” causality results: models may use representations of syntax, but probes may have learned to use redundant encodings of the same syntactic information. We demonstrate that models do encode syntactic information redundantly and introduce a new probe design that guides probes to consider all syntactic information present in embeddings. Using these probes, we find evidence for the use of syntax in models where prior methods did not, allowing us to boost model performance by injecting syntactic information into representations.

*Link*: https://aclanthology.org/2022.naacl-main.394/

___

### Linguistic Frameworks Go Toe-to-Toe at Neuro-Symbolic Language Modeling.
We examine the extent to which, in principle, different syntactic and semantic graph representations can complement and improve neural language modeling. Specifically, by conditioning on a subgraph encapsulating the locally relevant sentence history, can a model make better next-word predictions than a pretrained sequential language model alone? With an ensemble setup consisting of GPT-2 and ground-truth graphs from one of 7 different formalisms, we find that the graph information indeed improves perplexity and other metrics. Moreover, this architecture provides a new way to compare different frameworks of linguistic representation. In our oracle graph setup, training and evaluating on English WSJ, semantic constituency structures prove most useful to language modeling performance—outpacing syntactic constituency structures as well as syntactic and semantic dependency structures.

*Link*: https://aclanthology.org/2022.naacl-main.325/

---

### Using Paraphrases to Study Properties of Contextual Embeddings
We use paraphrases as a unique source of data to analyze contextualized embeddings, with a particular focus on BERT. Because paraphrases naturally encode consistent word and phrase semantics, they provide a unique lens for investigating properties of embeddings. Using the Paraphrase Database’s alignments, we study words within paraphrases as well as phrase representations. We find that contextual embeddings effectively handle polysemous words, but give synonyms surprisingly different representations in many cases. We confirm previous findings that BERT is sensitive to word order, but find slightly different patterns than prior work in terms of the level of contextualization across BERT’s layers.

*Link*: https://aclanthology.org/2022.naacl-main.338/

---

### Necessity and Sufficiency for Explaining Text Classifiers: A Case Study in Hate Speech Detection
We present a novel feature attribution method for explaining text classifiers, and analyze it in the context of hate speech detection. Although feature attribution models usually provide a single importance score for each token, we instead provide two complementary and theoretically-grounded scores – necessity and sufficiency – resulting in more informative explanations. We propose a transparent method that calculates these values by generating explicit perturbations of the input text, allowing the importance scores themselves to be explainable. We employ our method to explain the predictions of different hate speech detection models on the same set of curated examples from a test suite, and show that different values of necessity and sufficiency for identity terms correspond to different kinds of false positive errors, exposing sources of classifier bias against marginalized groups.

*Link*: https://aclanthology.org/2022.naacl-main.192/

---

### Interactive Query-Assisted Summarization via Deep Reinforcement Learning
Interactive summarization is a task that facilitates user-guided exploration of information within a document set. While one would like to employ state of the art neural models to improve the quality of interactive summarization, many such technologies cannot ingest the full document set or cannot operate at sufficient speed for interactivity. To that end, we propose two novel deep reinforcement learning models for the task that address, respectively, the subtask of summarizing salient information that adheres to user queries, and the subtask of listing suggested queries to assist users throughout their exploration. In particular, our models allow encoding the interactive session state and history to refrain from redundancy. Together, these models compose a state of the art solution that addresses all of the task requirements. We compare our solution to a recent interactive summarization system, and show through an experimental study involving real users that our models are able to improve informativeness while preserving positive user experience.

*Link*: https://aclanthology.org/2022.naacl-main.184/

---

### GPL: Generative Pseudo Labeling for Unsupervised Domain Adaptation of Dense Retrieval
Dense retrieval approaches can overcome the lexical gap and lead to significantly improved search results. However, they require large amounts of training data which is not available for most domains. As shown in previous work (Thakur et al., 2021b), the performance of dense retrievers severely degrades under a domain shift. This limits the usage of dense retrieval approaches to only a few domains with large training datasets. In this paper, we propose the novel unsupervised domain adaptation method Generative Pseudo Labeling (GPL), which combines a query generator with pseudo labeling from a cross-encoder. On six representative domain-specialized datasets, we find the proposed GPL can outperform an out-of-the-box state-of-the-art dense retrieval approach by up to 9.3 points nDCG@10. GPL requires less (unlabeled) data from the target domain and is more robust in its training than previous methods. We further investigate the role of six recent pre-training methods in the scenario of domain adaptation for retrieval tasks, where only three could yield improved results. The best approach, TSDAE (Wang et al., 2021) can be combined with GPL, yielding another average improvement of 1.4 points nDCG@10 across the six tasks. The code and the models are available at https://github.com/UKPLab/gpl.

*Link*: https://aclanthology.org/2022.naacl-main.168/

--- 

###  Boosted Dense Retriever
We propose DrBoost, a dense retrieval ensemble inspired by boosting. DrBoost is trained in stages: each component model is learned sequentially and specialized by focusing only on retrieval mistakes made by the current ensemble. The final representation is the concatenation of the output vectors of all the component models, making it a drop-in replacement for standard dense retrievers at test time. DrBoost enjoys several advantages compared to standard dense retrieval models. It produces representations which are 4x more compact, while delivering comparable retrieval results. It also performs surprisingly well under approximate search with coarse quantization, reducing latency and bandwidth needs by another 4x. In practice, this can make the difference between serving indices from disk versus from memory, paving the way for much cheaper deployments.

*Enrique's comment*: This talk was really neat. It promises very good performance after reducing the index size

*Link*: https://aclanthology.org/2022.naacl-main.226/

---

### Multi-Vector Models with Textual Guidance for Fine-Grained Scientific Document Similarity
We present a new scientific document similarity model based on matching fine-grained aspects of texts. To train our model, we exploit a naturally-occurring source of supervision: sentences in the full-text of papers that cite multiple papers together (co-citations). Such co-citations not only reflect close paper relatedness, but also provide textual descriptions of how the co-cited papers are related. This novel form of textual supervision is used for learning to match aspects across papers. We develop multi-vector representations where vectors correspond to sentence-level aspects of documents, and present two methods for aspect matching: (1) A fast method that only matches single aspects, and (2) a method that makes sparse multiple matches with an Optimal Transport mechanism that computes an Earth Mover’s Distance between aspects. Our approach improves performance on document similarity tasks in four datasets. Further, our fast single-match method achieves competitive results, paving the way for applying fine-grained similarity to large scientific corpora.

*Link*: https://aclanthology.org/2022.naacl-main.331/

---

### A Structured Span Selector
Many natural language processing tasks, e.g., coreference resolution and semantic role labeling, require selecting text spans and making decisions about them. A typical approach to such tasks is to score all possible spans and greedily select spans for task-specific downstream processing. This approach, however, does not incorporate any inductive bias about what sort of spans ought to be selected, e.g., that selected spans tend to be syntactic constituents. In this paper, we propose a novel grammar-based structured span selection model which learns to make use of the partial span-level annotation provided for such problems. Compared to previous approaches, our approach gets rid of the heuristic greedy span selection scheme, allowing us to model the downstream task on an optimal set of spans. We evaluate our model on two popular span prediction tasks: coreference resolution and semantic role labeling; and show improvements on both.


*Link*: https://aclanthology.org/2022.naacl-main.189/

---

### Learning to Retrieve Passages without Supervision
Dense retrievers for open-domain question answering (ODQA) have been shown to achieve impressive performance by training on large datasets of question-passage pairs. In this work we ask whether this dependence on labeled data can be reduced via unsupervised pretraining that is geared towards ODQA. We show this is in fact possible, via a novel pretraining scheme designed for retrieval. Our “recurring span retrieval” approach uses recurring spans across passages in a document to create pseudo examples for contrastive learning. Our pretraining scheme directly controls for term overlap across pseudo queries and relevant passages, thus allowing to model both lexical and semantic relations between them. The resulting model, named Spider, performs surprisingly well without any labeled training examples on a wide range of ODQA datasets. Specifically, it significantly outperforms all other pretrained baselines in a zero-shot setting, and is competitive with BM25, a strong sparse baseline. Moreover, a hybrid retriever over Spider and BM25 improves over both, and is often competitive with DPR models, which are trained on tens of thousands of examples. Last, notable gains are observed when using Spider as an initialization for supervised training.

*Link*: https://aclanthology.org/2022.naacl-main.193/

---

### Entity Linking via Explicit Mention-Mention Coreference Modeling
Learning representations of entity mentions is a core component of modern entity linking systems for both candidate generation and making linking predictions. In this paper, we present and empirically analyze a novel training approach for learning mention and entity representations that is based on building minimum spanning arborescences (i.e., directed spanning trees) over mentions and entities across documents to explicitly model mention coreference relationships. We demonstrate the efficacy of our approach by showing significant improvements in both candidate generation recall and linking accuracy on the Zero-Shot Entity Linking dataset and MedMentions, the largest publicly available biomedical dataset. In addition, we show that our improvements in candidate generation yield higher quality re-ranking models downstream, setting a new SOTA result in linking accuracy on MedMentions. Finally, we demonstrate that our improved mention representations are also effective for the discovery of new entities via cross-document coreference.

*Link*: https://aclanthology.org/2022.naacl-main.343/

---

### MultiSpanQA: A Dataset for Multi-Span Question Answering
Most existing reading comprehension datasets focus on single-span answers, which can be extracted as a single contiguous span from a given text passage. Multi-span questions, i.e., questions whose answer is a series of multiple discontiguous spans in the text, are common real life but are less studied. In this paper, we present MultiSpanQA, a new dataset that focuses on multi-span questions. Raw questions and contexts are extracted from the Natural Questions dataset. After multi-span re-annotation, MultiSpanQA consists of over a total of 6,000 multi-span questions in the basic version, and over 19,000 examples with unanswerable questions, and questions with single-, and multi-span answers in the expanded version. We introduce new metrics for the purposes of multi-span question answering evaluation, and establish several baselines using advanced models. Finally, we propose a new model which beats all baselines and achieves state-of-the-art on our dataset.

*Link*: https://aclanthology.org/2022.naacl-main.90/

---

### Modularized Transfer Learning with Multiple Knowledge Graphs for Zero-shot Commonsense Reasoning
Commonsense reasoning systems should be able to generalize to diverse reasoning cases. However, most state-of-the-art approaches depend on expensive data annotations and overfit to a specific benchmark without learning how to perform general semantic reasoning. To overcome these drawbacks, zero-shot QA systems have shown promise as a robust learning scheme by transforming a commonsense knowledge graph (KG) into synthetic QA-form samples for model training. Considering the increasing type of different commonsense KGs, this paper aims to extend the zero-shot transfer learning scenario into multiple-source settings, where different KGs can be utilized synergetically. Towards this goal, we propose to mitigate the loss of knowledge from the interference among the different knowledge sources, by developing a modular variant of the knowledge aggregation as a new zero-shot commonsense reasoning framework. Results on five commonsense reasoning benchmarks demonstrate the efficacy of our framework, improving the performance with multiple KGs.

*Link*: https://aclanthology.org/2022.naacl-main.163/

---

### ♫ MuSiQue: Multihop Questions via Single-hop Question Composition
Multihop reasoning remains an elusive goal as existing multihop benchmarks are known to be largely solvable via shortcuts. Can we create a question answering (QA) dataset that, by construction, requires proper multihop reasoning? To this end, we introduce a bottom–up approach that systematically selects composable pairs of single-hop questions that are connected, that is, where one reasoning step critically relies on information from another. This bottom–up methodology lets us explore a vast space of questions and add stringent filters as well as other mechanisms targeting connected reasoning. It provides fine-grained control over the construction process and the properties of the resulting k-hop questions. We use this methodology to create MuSiQue-Ans, a new multihop QA dataset with 25K 2–4 hop questions. Relative to existing datasets, MuSiQue-Ans is more difficult overall (3× increase in human–machine gap), and harder to cheat via disconnected reasoning (e.g., a single-hop model has a 30-point drop in F1). We further add unanswerable contrast questions to produce a more stringent dataset, MuSiQue-Full. We hope our datasets will help the NLP community develop models that perform genuine multihop reasoning.

*Link*: https://aclanthology.org/2022.tacl-1.31/
