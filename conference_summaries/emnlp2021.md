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

**Mihai's comments**: The in-person talk was skipped due to technical issues. read this!

### 
**URL**: 

**Abstract**: 

**Mihai's comments**: 



