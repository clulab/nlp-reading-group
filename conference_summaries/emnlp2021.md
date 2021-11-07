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



