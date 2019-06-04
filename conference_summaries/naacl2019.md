## Hopefully collaborative summary of talks we found interesting...

[**Practical Semantic Parsing for Spoken Language Understanding**](https://www.aclweb.org/anthology/N19-2003)

**ABSTRACT**: Executable semantic parsing is the task of con- verting natural language utterances into logi- cal forms that can be directly used as queries to get a response. We build a transfer learn- ing framework for executable semantic pars- ing. We show that the framework is effec- tive for Question Answering (Q&A) as well as for Spoken Language Understanding (SLU). We further investigate the case where a parser on a new domain can be learned by exploit- ing data on other domains, either via multi- task learning between the target domain and an auxiliary domain or via pre-training on the auxiliary domain and fine-tuning on the target domain. With either flavor of transfer learn- ing, we are able to improve performance on most domains; we experiment with public data sets such as Overnight and NLmaps as well as with commercial SLU data. The experiments carried out on data sets that are different in nature show how executable semantic parsing can unify different areas of NLP such as Q&A and SLU.

- Executable semantic parsing for QA and spoken language understanding
    - Could use Slot tagging (i.e., tag the slot that the word will fill) → but here they convert the seq to trees and use Transition based parser
- Exploit high resource domains for low-resource ones:
    - Datasets: Overnight, NLmaps -- low-resource datasets, and different action types have very different number of slots/slot types.
    - Pretraining on high-resource domain helps more consistently than multi-task setting!

[**Cross-Lingual Alignment of Contextual Word Embeddings, with Applications to Zero-shot Dependency Parsing**](https://www.aclweb.org/anthology/N19-1162)

**ABSTRACT**: We introduce a novel method for multilingual transfer that utilizes deep contextual embeddings, pretrained in an unsupervised fashion. While contextual embeddings have been shown to yield richer representations of meaning compared to their static counterparts, aligning them poses a challenge due to their dynamic nature. To this end, we construct context-independent variants of the original monolingual spaces and utilize their map- ping to derive an alignment for the context- dependent spaces. This mapping readily supports processing of a target language, improv- ing transfer by context-aware embeddings. Our experimental results demonstrate the effectiveness of this approach for zero-shot and few-shot learning of dependency parsing. Specifically, our method consistently outperforms the previous state-of-the-art on 6 tested languages, yielding an improvement of 6.8 LAS points on average.

Makes use of the "cloud" of embeddings for a token, (from the contextualized nature) and the "anchor" for the cloud (which allows you to make use of a bilingual dictionary for supervision).  Maintains the contextualized nature of the spaces.

[**Inoculation by Fine-Tuning: A Method for Analyzing Challenge Datasets**](https://www.aclweb.org/anthology/N19-1225)

**ABSTRACT**: Several datasets have recently been constructed to expose brittleness in models trained on existing benchmarks. While model performance on these challenge datasets is significantly lower compared to the original benchmark, it is unclear what particular weaknesses they reveal. For example, a challenge dataset may be difficult because it targets phenomena that current models cannot capture, or because it simply exploits blind spots in a model’s specific training set. We introduce inoculation by fine-tuning, a new analysis method for studying challenge datasets by exposing models (the metaphorical patient) to a small amount of data from the challenge dataset (a metaphor- ical pathogen) and assessing how well they can adapt. We apply our method to analyze the NLI “stress tests” (Naik et al., 2018) and the Adversarial SQuAD dataset (Jia and Liang, 2017). We show that after slight exposure, some of these datasets are no longer challenging, while others remain difficult. Our results indicate that failures on challenge datasets may lead to very different conclusions about models, training datasets, and the challenge datasets themselves.

NOTES: Using fine-tuning to better understanding why NN models fail on challenge / adversarial datasets.  Basically, if fine-tuning (on a small number of challenge dataset examples) fixes performance, then the failing reveals a weakness of the data.  Else, if it doesn’t help, then it’s a model weakness. Otherwise, if after fine-tuning, challenge does better and base does worse, then it’s a distribution issue/artifacts -- e.g., fine-tuning teaches the model to ignore something that had signal in normal dataset, but not in challenge, like last sentence in SQuAD challenge set.  
Counter-argument: We’ll never have a perfect train set, so is it really a dataset fail?

