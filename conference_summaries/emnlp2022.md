# Mihai's Summary of EMNLP 2022

### GREENER: Graph Neural Networks for News Media Profiling
**URL**: https://arxiv.org/abs/2211.05533
**Abstract**: We study the problem of profiling news media on the Web with respect to their factuality of reporting and bias. This is an important but under-studied problem related to disinformation and "fake news" detection, but it addresses the issue at a coarser granularity compared to looking at an individual article or an individual claim. This is useful as it allows to profile entire media outlets in advance. Unlike previous work, which has focused primarily on text (e.g.,~on the text of the articles published by the target website, or on the textual description in their social media profiles or in Wikipedia), here our main focus is on modeling the similarity between media outlets based on the overlap of their audience. This is motivated by homophily considerations, i.e.,~the tendency of people to have connections to people with similar interests, which we extend to media, hypothesizing that similar types of media would be read by similar kinds of users. In particular, we propose GREENER (GRaph nEural nEtwork for News mEdia pRofiling), a model that builds a graph of inter-media connections based on their audience overlap, and then uses graph neural networks to represent each medium. We find that such representations are quite useful for predicting the factuality and the bias of news media outlets, yielding improvements over state-of-the-art results reported on two datasets. When augmented with conventionally used representations obtained from news articles, Twitter, YouTube, Facebook, and Wikipedia, prediction accuracy is found to improve by 2.5-27 macro-F1 points for the two tasks.

**Mihai's comments**: Modeling fact verification as a graph-based tsk helps for fact verification

### UnifiedSKG: Unifying and Multi-Tasking Structured Knowledge Grounding with Text-to-Text Language Models
**URL**: 
**Abstract**: 
**Mihai's comments**: An unified architecture for 6 tasks that work on top of knowledge bases. Each task has a unique prefix, but they share the same LM underneath. This improves performance over models fine-tuned individually. GPT-3 + Codex and few-shot learning performs worse than fine-tuning a LM.

### 
**URL**: 
**Abstract**: 
**Mihai's comments**: 
