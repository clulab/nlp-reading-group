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

**Mihai's comments**: One more paper that inserts rules into the training of BERT. Unlike the previous paper, here rules are not known, and are modeled as latent variables in a document-wide RE task. This joint task is trained using a form of EM.


### 
**URL**: 

**Abstract**: 

**Mihai's comments**: 


### 
**URL**: 

**Abstract**: 

**Mihai's comments**: 


### 
**URL**: 

**Abstract**: 

**Mihai's comments**: 


### 
**URL**: 

**Abstract**: 

**Mihai's comments**: 


### 
**URL**: 

**Abstract**: 

**Mihai's comments**: 



