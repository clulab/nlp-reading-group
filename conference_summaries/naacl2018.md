# Mihai's Summary of NAACL 2018

Disclaimer: this is obviously a small subset of the awesome papers/talks at NAACL. I selected them subjectively, based solely on my own research interests. 

The NAACL proceedings are available here: https://aclanthology.coli.uni-saarland.de/events/naacl-201

## DAY 1

### Keynote by Charles Yang on how children learn 
*Mihai's comments:* key statement: for a rule to be good and acquired by children, the number of exceptions should be smaller than N/ln(N), where N is the total number of cases where it applies. This has immediate applications to rule induction algorithms!

### Label-Aware Double Transfer Learning for Cross-Specialty Medical Named Entity Recognition
_Zhenghui Wang, Yanru Qu, Liheng Chen, Jian Shen, Weinan Zhang, Shaodian Zhang, Yimei Gao, Gen Gu, Ken Chen, and Yong Yu_

We study the problem of named entity recognition (NER) from electronic medical records, which is one of the most fundamental and critical problems for medical text mining. Medical records which are written by clinicians from different specialties usually contain quite different terminologies and writing styles. The difference of specialties and the cost of human annotation makes it particularly difficult to train a universal medical NER system. In this paper, we propose a label-aware double transfer learning framework (La-DTL) for cross-specialty NER, so that a medical NER system designed for one specialty could be conveniently applied to another one with minimal annotation efforts. The transferability is guaranteed by two components: (i) we propose label-aware MMD for feature representation transfer, and (ii) we perform parameter transfer with a theoretical upper bound which is also label aware. We conduct extensive experiments on 12 cross-specialty NER tasks. The experimental results demonstrate that La-DTL provides consistent accuracy improvement over strong baselines. Besides, the promising experimental results on non-medical NER scenarios indicate that La-DTL is potential to be seamlessly adapted to a wide range of NER tasks.

*Mihai's comments*: Discusses multitask training with a framework somewhat close to Marco’s idea!
