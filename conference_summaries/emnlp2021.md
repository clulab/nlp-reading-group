# Mihai's Summary of EMNLP 2021

### ContractNLI: A Dataset for Document-level Natural Language Inference for Contracts
**URL**: https://arxiv.org/pdf/2110.01799.pdf

**Abstract**: Reviewing contracts is a time-consuming pro- cedure that incurs large expenses to companies and social inequality to those who cannot af- ford it. In this work, we propose document- level natural language inference (NLI) for con- tracts, a novel, real-world application of NLI that addresses such problems. In this task, a system is given a set of hypotheses (such as “Some obligations of Agreement may survive termination.”) and a contract, and it is asked to classify whether each hypothesis is entailed by, contradicting to or not mentioned by (neu- tral to) the contract as well as identifying ev- idence for the decision as spans in the con- tract. We annotated and release the largest cor- pus to date consisting of 607 annotated con- tracts. We then show that existing models fail badly on our task and introduce a strong baseline, which (1) models evidence identifi- cation as multi-label classification over spans instead of trying to predict start and end to- kens, and (2) employs more sophisticated con- text segmentation for dealing with long docu- ments. We also show that linguistic character- istics of contracts, such as negations by excep- tions, are contributing to the difficulty of this task and that there is much room for improve- ment.

### Ido Dagan's Keynote
**Mihai's comments**: 
* See "Controled Crowdsourcing" from his ACL 2020 paper (Rot or Roit et al.) as a way to scale up user studies. Relevant for Marco and Becky's rule synthesis project.
* Cross-document LM (paper in EMNLP Findings 2021): may be useful for coreference resolution (Enfa).


### Distilling Linguistic Context for Language Model Compression
**URL**: https://arxiv.org/abs/2109.08359

**Abstract**: A computationally expensive and memory intensive neural network lies behind the recent success of language representation learning. Knowledge distillation, a major technique for deploying such a vast language model in resource-scarce environments, transfers the knowledge on individual word representations learned without restrictions. In this paper, inspired by the recent observations that language representations are relatively positioned and have more semantic knowledge as a whole, we present a new knowledge distillation objective for language representation learning that transfers the contextual knowledge via two types of relationships across representations: Word Relation and Layer Transforming Relation. Unlike other recent distillation techniques for the language models, our contextual distillation does not have any restrictions on architectural changes between teacher and student. We validate the effectiveness of our method on challenging benchmarks of language understanding tasks, not only in architectures of various sizes, but also in combination with DynaBERT, the recently proposed adaptive size pruning method.

**Mihai's comments**: Distills a network making sure that relations between words (e.g., meronymy) is preserved. That is, the consistency loss minimizes the difference between relation classification scores. See this paper also as a good review of distillation losses. Relevant for Mithun.

### Dynamic Knowledge Distillation for Pre-trained Language Models
**URL**: https://arxiv.org/abs/2109.11295

**Abstract**: Knowledge distillation~(KD) has been proved effective for compressing large-scale pre-trained language models. However, existing methods conduct KD statically, e.g., the student model aligns its output distribution to that of a selected teacher model on the pre-defined training dataset. In this paper, we explore whether a dynamic knowledge distillation that empowers the student to adjust the learning procedure according to its competency, regarding the student performance and learning efficiency. We explore the dynamical adjustments on three aspects: teacher model adoption, data selection, and KD objective adaptation. Experimental results show that (1) proper selection of teacher model can boost the performance of student model; (2) conducting KD with 10% informative instances achieves comparable performance while greatly accelerates the training; (3) the student performance can be boosted by adjusting the supervision contribution of different alignment objective. We find dynamic knowledge distillation is promising and provide discussions on potential future directions towards more efficient KD methods. 

**Mihai's comments**: Similar idea to Mithun and Sandeep's paper, but they adjust the learning process dynamically rather than adjusting the data (as Mithun and Sandeep do).

### Distantly Supervised Relation Extraction using Multi-Layer Revision Network and Confidence-based Multi-Instance Learning
**URL**: TBD

**Abstract**: TBD 

**Mihai's comments**: reduces DS noise at both word level and sentence level. The word-level noise is reduced using TransE, i.e., making sure we keep the words that maximize the similarity between e(head - mod) and e(relation). Sentences are filtered based on prediction confidence.

### Logic-level Evidence Retrieval and Graph-based Verification Network for Table-based Fact Verification
**URL**: https://arxiv.org/abs/2109.06480

**Abstract**: Table-based fact verification task aims to verify whether the given statement is supported by the given semi-structured table. Symbolic reasoning with logical operations plays a crucial role in this task. Existing methods leverage programs that contain rich logical information to enhance the verification process. However, due to the lack of fully supervised signals in the program generation process, spurious programs can be derived and employed, which leads to the inability of the model to catch helpful logical operations. To address the aforementioned problems, in this work, we formulate the table-based fact verification task as an evidence retrieval and reasoning framework, proposing the Logic-level Evidence Retrieval and Graph-based Verification network (LERGV). Specifically, we first retrieve logic-level program-like evidence from the given table and statement as supplementary evidence for the table. After that, we construct a logic-level graph to capture the logical relations between entities and functions in the retrieved evidence, and design a graph-based verification network to perform logic-level graph-based reasoning based on the constructed graph to classify the final entailment relation. Experimental results on the large-scale benchmark TABFACT show the effectiveness of the proposed approach.

**Mihai's comments**: presentation skipped due to technical issues. TODO: Read this!

### Condenser: a Pre-training Architecture for Dense Retrieval
**URL**: https://arxiv.org/abs/2104.08253

**Abstract**: Pre-trained Transformer language models (LM) have become go-to text representation encoders. Prior research fine-tunes deep LMs to encode text sequences such as sentences and passages into single dense vector representations for efficient text comparison and retrieval. However, dense encoders require a lot of data and sophisticated techniques to effectively train and suffer in low data situations. This paper finds a key reason is that standard LMs' internal attention structure is not ready-to-use for dense encoders, which needs to aggregate text information into the dense representation. We propose to pre-train towards dense encoder with a novel Transformer architecture, Condenser, where LM prediction CONditions on DENSE Representation. Our experiments show Condenser improves over standard LM by large margins on various text retrieval and similarity tasks.

**Mihai's comments**: Forces [CLS] to encode the information generated in the later layers. This encourages [CLS] to capture information about the whole text, which was shown to not happen with the original implementation. Works well on NQ, TQA, MS-MARCO. Relevant to Fan and Zhengzhong.

### Disentangling Representations of Text by Masking Transformers
**URL**: https://arxiv.org/abs/2104.07155

**Abstract**: Representations from large pretrained models such as BERT encode a range of features into monolithic vectors, affording strong predictive accuracy across a multitude of downstream tasks. In this paper we explore whether it is possible to learn disentangled representations by identifying existing subnetworks within pretrained models that encode distinct, complementary aspect representations. Concretely, we learn binary masks over transformer weights or hidden units to uncover subsets of features that correlate with a specific factor of variation; this eliminates the need to train a disentangled model from scratch for a particular task. We evaluate this method with respect to its ability to disentangle representations of sentiment from genre in movie reviews, "toxicity" from dialect in Tweets, and syntax from semantics. 
By combining masking with magnitude pruning we find that we can identify sparse subnetworks within BERT that strongly encode particular aspects (e.g., toxicity) while only weakly encoding others (e.g., race). Moreover, despite only learning masks, we find that disentanglement-via-masking performs as well as -- and often better than -- previously proposed methods based on variational autoencoders and adversarial training.

**Mihai's comments**: Learns a binary mask on top of BERT to mark important/non-important words. Relevant to Zheng.

### Label Verbalization and Entailment for Effective Zero and Few-Shot Relation Extraction
**URL**: https://arxiv.org/abs/2109.03659

**Abstract**: Relation extraction systems require large amounts of labeled examples which are costly to annotate. In this work we reformulate relation extraction as an entailment task, with simple, hand-made, verbalizations of relations produced in less than 15 min per relation. The system relies on a pretrained textual entailment engine which is run as-is (no training examples, zero-shot) or further fine-tuned on labeled examples (few-shot or fully trained). In our experiments on TACRED we attain 63% F1 zero-shot, 69% with 16 examples per relation (17% points better than the best supervised system on the same conditions), and only 4 points short to the state-of-the-art (which uses 20 times more training data). We also show that the performance can be improved significantly with larger entailment models, up to 12 points in zero-shot, allowing to report the best results to date on TACRED when fully trained. The analysis shows that our few-shot systems are specially effective when discriminating between relations, and that the performance difference in low data regimes comes mainly from identifying no-relation cases.

**Mihai's comments**: Formulates RE as NLI, and feeds it into the pre-training phase of the LM.

### RuleBERT: Teaching Soft Rules to Pre-Trained Language Models
**URL**: https://arxiv.org/abs/2109.13006

**Abstract**: While pre-trained language models (PLMs) are the go-to solution to tackle many natural language processing problems, they are still very limited in their ability to capture and to use common-sense knowledge. In fact, even if information is available in the form of approximate (soft) logical rules, it is not clear how to transfer it to a PLM in order to improve its performance for deductive reasoning tasks. Here, we aim to bridge this gap by teaching PLMs how to reason with soft Horn rules. We introduce a classification task where, given facts and soft rules, the PLM should return a prediction with a probability for a given hypothesis. We release the first dataset for this task, and we propose a revised loss function that enables the PLM to learn how to predict precise probabilities for the task. Our evaluation results show that the resulting fine-tuned models achieve very high performance, even on logical rules that were unseen at training. Moreover, we demonstrate that logical notions expressed by the rules are transferred to the fine-tuned model, yielding state-of-the-art results on external datasets.

**Mihai's comments**: Very cool work that shows how to fine-tune BERT on soft rules. Rules are verbalized to be inserted as training data. Rule weights are incorporated using a weighted binary cross entropy (where the weights correspond to rule confidences). Relevant to Zhengzhong.


### Learning Logic Rules for Document-Level Relation Extraction
**URL**: https://underline.io/lecture/38055-learning-logic-rules-for-document-level-relation-extraction

**Abstract**: Document-level relation extraction aims to identify relations between entities in a whole document. Prior efforts to capture long-range dependencies have relied heavily on implicitly powerful representations learned through (graph) neural networks, which makes the model less transparent. To tackle this challenge, in this paper, we propose LogiRE, a novel probabilistic model for document-level relation extraction by learning logic rules. LogiRE treats logic rules as latent variables and consists of two modules: a rule generator and a relation extractor. The rule generator is to generate logic rules potentially contributing to final predictions, and the relation extractor outputs final predictions based on the generated logic rules. Those two modules can be efficiently optimized with the expectation-maximization (EM) algorithm. By introducing logic rules into neural networks, LogiRE can explicitly capture long-range dependencies as well as enjoy better interpretation. Empirical results show that LogiRE significantly outperforms several strong baselines in terms of relation performance and logical consistency.

**Mihai's comments**: One more paper that inserts rules into the training of BERT. Unlike the previous paper, here rules are not known, and are modeled as latent variables in a document-wide RE task. This joint task is trained using a form of EM. The rule generation is tractable because rules are generated as sets of possible relations that hold between pairs of entities in a document (rather than any possible rule in the universe). Relevant to Zhengzhong.


### All Bark and No Bite: Rogue Dimensions in Transformer Language Models Obscure Representational Quality
**URL**: https://arxiv.org/abs/2109.04404

**Abstract**: Similarity measures are a vital tool for understanding how language models represent and process language. Standard representational similarity measures such as cosine similarity and Euclidean distance have been successfully used in static word embedding models to understand how words cluster in semantic space. Recently, these measures have been applied to embeddings from contextualized models such as BERT and GPT-2. In this work, we call into question the informativity of such measures for contextualized language models. We find that a small number of rogue dimensions, often just 1-3, dominate these measures. Moreover, we find a striking mismatch between the dimensions that dominate similarity measures and those which are important to the behavior of the model. We show that simple postprocessing techniques such as standardization are able to correct for rogue dimensions and reveal underlying representational quality. We argue that accounting for rogue dimensions is essential for any similarity-based analysis of contextual language models.

**Mihai's comments**: "Rogue" dimensions in contextualized embeddings dominate cosine similarity scores. But their impact on LM probability distributions is not that big. This is the most surprising finding at EMNLP so far for me... The authors also introduce postprocessing techniques to normalize dimensions, which reduces the impact of these rogue dimensions.


### Neuralizing Regular Expressions for Slot Filling
**URL**: https://faculty.sist.shanghaitech.edu.cn/faculty/tukw/emnlp21.pdf

**Abstract**: Neural models and symbolic rules such as reg- ular expressions have their respective merits and weaknesses. In this paper, we study the integration of the two approaches for the slot filling task by converting regular expressions into neural networks. Specifically, we first con- vert regular expressions into a special form of finite-state transducers, then unfold its approx- imate inference algorithm as a bidirectional re- current neural model that performs slot filling via sequence labeling. Experimental results show that our model has superior zero-shot and few-shot performance and stays competi- tive when there are sufficient training data.

**Mihai's comments**: Converts a regex (and its equivalent FST) into a neural network by using tensors to encode the transition scores. Trained using an algorithm similar to forward-backward. Relevant to OdinSynth (Robert, George, Marco).


### Case-based Reasoning for Natural Language Queries over Knowledge Bases
**URL**: https://arxiv.org/abs/2104.08762

**Abstract**: It is often challenging to solve a complex problem from scratch, but much easier if we can access other similar problems with their solutions -- a paradigm known as case-based reasoning (CBR). We propose a neuro-symbolic CBR approach (CBR-KBQA) for question answering over large knowledge bases. CBR-KBQA consists of a nonparametric memory that stores cases (question and logical forms) and a parametric model that can generate a logical form for a new question by retrieving cases that are relevant to it. On several KBQA datasets that contain complex questions, CBR-KBQA achieves competitive performance. For example, on the ComplexWebQuestions dataset, CBR-KBQA outperforms the current state of the art by 11\% on accuracy. Furthermore, we show that CBR-KBQA is capable of using new cases \emph{without} any further training: by incorporating a few human-labeled examples in the case memory, CBR-KBQA is able to successfully generate logical forms containing unseen KB entities as well as relations.

**Mihai's comments**: Generates SQL-like queries for question types unseen in training by mixing and matching parts of queries from training questions similar to the test uqestion.


### Progressive Adversarial Learning for Bootstrapping: A Case Study on Entity Set Expansion
**URL**: https://arxiv.org/abs/2109.12082

**Abstract**: Bootstrapping has become the mainstream method for entity set expansion. Conventional bootstrapping methods mostly define the expansion boundary using seed-based distance metrics, which heavily depend on the quality of selected seeds and are hard to be adjusted due to the extremely sparse supervision. In this paper, we propose BootstrapGAN, a new learning method for bootstrapping which jointly models the bootstrapping process and the boundary learning process in a GAN framework. Specifically, the expansion boundaries of different bootstrapping iterations are learned via different discriminator networks; the bootstrapping network is the generator to generate new positive entities, and the discriminator networks identify the expansion boundaries by trying to distinguish the generated entities from known positive entities. By iteratively performing the above adversarial learning, the generator and the discriminators can reinforce each other and be progressively refined along the whole bootstrapping process. Experiments show that BootstrapGAN achieves the new state-of-the-art entity set expansion performance.

**Mihai's comments**: A direct application of adversarial learning for bootstrapping, which outperforms traditional bootstrapping. Relevant to Robert, George, Mahdi.


### Modular Self-Supervision for Document-Level Relation Extraction
**URL**: https://arxiv.org/abs/2109.05362

**Abstract**: Extracting relations across large text spans has been relatively underexplored in NLP, but it is particularly important for high-value domains such as biomedicine, where obtaining high recall of the latest findings is crucial for practical applications. Compared to conventional information extraction confined to short text spans, document-level relation extraction faces additional challenges in both inference and learning. Given longer text spans, state-of-the-art neural architectures are less effective and task-specific self-supervision such as distant supervision becomes very noisy. In this paper, we propose decomposing document-level relation extraction into relation detection and argument resolution, taking inspiration from Davidsonian semantics. This enables us to incorporate explicit discourse modeling and leverage modular self-supervision for each sub-problem, which is less noise-prone and can be further refined end-to-end via variational EM. We conduct a thorough evaluation in biomedical machine reading for precision oncology, where cross-paragraph relation mentions are prevalent. Our method outperforms prior state of the art, such as multi-scale learning and graph neural networks, by over 20 absolute F1 points. The gain is particularly pronounced among the most challenging relation instances whose arguments never co-occur in a paragraph.

**Mihai's comments**: Did not read this paper, yet. But it seems like a simple, modular approach for long-distance RE. Relevant to Enrique.


### PermuteFormer: Efficient Relative Position Encoding for Long Sequences
**URL**: https://arxiv.org/abs/2109.02377

**Abstract**: A recent variation of Transformer, Performer, scales Transformer to longer sequences with a linear attention mechanism. However, it is not compatible with relative position encoding, which has advantages over absolute position encoding. In this paper, we discuss possible ways to add relative position encoding to Performer. Based on the analysis, we propose PermuteFormer, a Performer-based model with relative position encoding that scales linearly on long sequences. PermuteFormer applies position-dependent transformation on queries and keys to encode positional information into the attention module. This transformation is carefully crafted so that the final output of self-attention is not affected by absolute positions of tokens. PermuteFormer introduces negligible computational overhead by design that it runs as fast as Performer. We evaluate PermuteFormer on Long-Range Arena, a dataset for long sequences, as well as WikiText-103, a language modeling dataset. The experiments show that PermuteFormer uniformly improves the performance of Performer with almost no computational overhead and outperforms vanilla Transformer on most of the tasks.

**Mihai's comments**: Nice discussion on how to encode relative positions (between query and value) in transformers. See also Long-Range Arena, a dataset for long sequences. Relevant to Enrique.


### Back to Square One: Artifact Detection, Training and Commonsense Disentanglement in the Winograd Schema
**URL**: https://arxiv.org/abs/2104.08161

**Abstract**: The Winograd Schema (WS) has been proposed as a test for measuring commonsense capabilities of models. Recently, pre-trained language model-based approaches have boosted performance on some WS benchmarks but the source of improvement is still not clear. This paper suggests that the apparent progress on WS may not necessarily reflect progress in commonsense reasoning. To support this claim, we first show that the current evaluation method of WS is sub-optimal and propose a modification that uses twin sentences for evaluation. We also propose two new baselines that indicate the existence of artifacts in WS benchmarks. We then develop a method for evaluating WS-like sentences in a zero-shot setting to account for the commonsense reasoning abilities acquired during the pretraining and observe that popular language models perform randomly in this setting when using our more strict evaluation. We conclude that the observed progress is mostly due to the use of supervision in training WS models, which is not likely to successfully support all the required commonsense reasoning skills and knowledge.

**Mihai's comments**: Winograd schema results are inflated because: (a) they overfit on lexical artifacts, (b) the evaluation inflates results (it should give credit only when both variants are classified correctly), and (c) training on this data yields limited generalization.


### Few-Shot Named Entity Recognition: An Empirical Baseline Study
**URL**: https://arxiv.org/abs/2012.14978

**Abstract**: This paper presents a comprehensive study to efficiently build named entity recognition (NER) systems when a small number of in-domain labeled data is available. Based upon recent Transformer-based self-supervised pre-trained language models (PLMs), we investigate three orthogonal schemes to improve the model generalization ability for few-shot settings: (1) meta-learning to construct prototypes for different entity types, (2) supervised pre-training on noisy web data to extract entity-related generic representations and (3) self-training to leverage unlabeled in-domain data. Different combinations of these schemes are also considered. We perform extensive empirical comparisons on 10 public NER datasets with various proportions of labeled data, suggesting useful insights for future research. Our experiments show that (i) in the few-shot learning setting, the proposed NER schemes significantly improve or outperform the commonly used baseline, a PLM-based linear classifier fine-tuned on domain labels; (ii) We create new state-of-the-art results on both few-shot and training-free settings compared with existing methods. We will release our code and pre-trained models for reproducible research.

**Mihai's comments**: Nice discussion of baselines for few-shot NER. See also WiFine, a large dataset with many NE labels. Very relevant to George.


### How to Train BERT with an Academic Budget
**URL**: https://arxiv.org/abs/2104.07705

**Abstract**: While large language models a la BERT are used ubiquitously in NLP, pretraining them is considered a luxury that only a few well-funded industry labs can afford. How can one train such models with a more modest budget? We present a recipe for pretraining a masked language model in 24 hours using a single low-end deep learning server. We demonstrate that through a combination of software optimizations, design choices, and hyperparameter tuning, it is possible to produce models that are competitive with BERT-base on GLUE tasks at a fraction of the original pretraining cost.

**Mihai's comments**: Use BERT-large (converges faster) but short sequences (128 tokens, single sents). Just masked LM during training. Gradient accumulation to simulate large batches. Time-based learning rate schedule (not epoch based). 


### Rationales for Sequential Predictions
**URL**: https://aclanthology.org/2021.emnlp-main.804.pdf

**Abstract**: Sequence models are a critical component of modern NLP systems, but their predictions are difficult to explain. We consider model ex- planations though rationales, subsets of con- text that can explain individual model predic- tions. We find sequential rationales by solving a combinatorial optimization: the best ratio- nale is the smallest subset of input tokens that would predict the same output as the full se- quence. Enumerating all subsets is intractable, so we propose an efficient greedy algorithm to approximate this objective. The algorithm, which is called greedy rationalization, applies to any model. For this approach to be effec- tive, the model should form compatible condi- tional distributions when making predictions on incomplete subsets of the context. This condition can be enforced with a short fine- tuning step. We study greedy rationalization on language modeling and machine translation. Compared to existing baselines, greedy ratio- nalization is best at optimizing the sequential objective and provides the most faithful ratio- nales. On a new dataset of annotated sequen- tial rationales, greedy rationales are most simi- lar to human rationales.

**Mihai's comments**: Greedily generates explanations for LMs, by incrementally including context words as long as prediction probability increases. Very relevant to Zheng.



