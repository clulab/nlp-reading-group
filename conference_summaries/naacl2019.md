## Hopefully collaborative summary of talks we found interesting...

[**Practical Semantic Parsing for Spoken Language Understanding**](https://www.aclweb.org/anthology/N19-2003)

abstract: Executable semantic parsing is the task of con- verting natural language utterances into logi- cal forms that can be directly used as queries to get a response. We build a transfer learn- ing framework for executable semantic pars- ing. We show that the framework is effec- tive for Question Answering (Q&A) as well as for Spoken Language Understanding (SLU). We further investigate the case where a parser on a new domain can be learned by exploit- ing data on other domains, either via multi- task learning between the target domain and an auxiliary domain or via pre-training on the auxiliary domain and fine-tuning on the target domain. With either flavor of transfer learn- ing, we are able to improve performance on most domains; we experiment with public data sets such as Overnight and NLmaps as well as with commercial SLU data. The experiments carried out on data sets that are different in nature show how executable semantic parsing can unify different areas of NLP such as Q&A and SLU.

- Executable semantic parsing for QA and spoken language understanding
    - Could use Slot tagging (i.e., tag the slot that the word will fill) → but here they convert the seq to trees and use Transition based parser
- Exploit high resource domains for low-resource ones:
    - Datasets: Overnight, NLmaps -- low-resource datasets, and different action types have very different number of slots/slot types.
    - Pretraining on high-resource domain helps more consistently than multi-task setting!

[**Cross-Lingual Alignment of Contextual Word Embeddings, with Applications to Zero-shot Dependency Parsing**](https://www.aclweb.org/anthology/N19-1162)

abstract: We introduce a novel method for multilingual transfer that utilizes deep contextual embeddings, pretrained in an unsupervised fashion. While contextual embeddings have been shown to yield richer representations of meaning compared to their static counterparts, aligning them poses a challenge due to their dynamic nature. To this end, we construct context-independent variants of the original monolingual spaces and utilize their map- ping to derive an alignment for the context- dependent spaces. This mapping readily supports processing of a target language, improv- ing transfer by context-aware embeddings. Our experimental results demonstrate the effectiveness of this approach for zero-shot and few-shot learning of dependency parsing. Specifically, our method consistently outperforms the previous state-of-the-art on 6 tested languages, yielding an improvement of 6.8 LAS points on average.

Makes use of the "cloud" of embeddings for a token, (from the contextualized nature) and the "anchor" for the cloud (which allows you to make use of a bilingual dictionary for supervision).  Maintains the contextualized nature of the spaces.
