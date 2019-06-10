## Hopefully collaborative summary of talks we found interesting...

*FYI: Becky's favorite so far is "Attention is not Explanation"*

[**Practical Semantic Parsing for Spoken Language Understanding**](https://www.aclweb.org/anthology/N19-2003)

**ABSTRACT**: Executable semantic parsing is the task of con- verting natural language utterances into logi- cal forms that can be directly used as queries to get a response. We build a transfer learn- ing framework for executable semantic pars- ing. We show that the framework is effec- tive for Question Answering (Q&A) as well as for Spoken Language Understanding (SLU). We further investigate the case where a parser on a new domain can be learned by exploit- ing data on other domains, either via multi- task learning between the target domain and an auxiliary domain or via pre-training on the auxiliary domain and fine-tuning on the target domain. With either flavor of transfer learn- ing, we are able to improve performance on most domains; we experiment with public data sets such as Overnight and NLmaps as well as with commercial SLU data. The experiments carried out on data sets that are different in nature show how executable semantic parsing can unify different areas of NLP such as Q&A and SLU.

**NOTES**:
- Executable semantic parsing for QA and spoken language understanding
    - Could use Slot tagging (i.e., tag the slot that the word will fill) → but here they convert the seq to trees and use Transition based parser
- Exploit high resource domains for low-resource ones:
    - Datasets: Overnight, NLmaps -- low-resource datasets, and different action types have very different number of slots/slot types.
    - Pretraining on high-resource domain helps more consistently than multi-task setting!

[**Density Matching for Bilingual Word Embedding**](https://www.aclweb.org/anthology/N19-1161)
**ABSTRACT**: Recent approaches to cross-lingual word embedding have generally been based on linear transformations between the sets of embedding vectors in the two languages. In this paper, we propose an approach that instead expresses the two monolingual embedding spaces as probability densities defined by a Gaussian mixture model, and matches the two densities using a method called normalizing flow. The method requires no explicit supervision, and can be learned with only a seed dictionary of words that have identical strings. We argue that this formulation has several intuitively attractive properties, particularly with the respect to improving robustness and generalization to mappings between difficult language pairs or word pairs. On a benchmark data set of bilingual lexicon induction and cross-lingual word similarity, our approach can achieve competitive or superior performance compared to state-of-the-art published results, with particularly strong results being found on etymologically distant and/or morphologically rich languages.

**NOTES**: A really different approach to aligning two vector spaces (from different languages) based on aligning the probability distributions of the embedding spaces.  They said it better handles the inherent uncertainty in the embeddings themselves (i.e., if you train same language model with diff random restarts, you get diff nearest neighbors).
During the question period, he was asked about how this would work with asymmetrical embeddings, to which he replied thatdata asymmetry behind initial word embeddings is a serious problem, and aligning those spaces (with any technique) doesn’t work well, but there is maybe more potential with this approach...

[**Cross-Lingual Alignment of Contextual Word Embeddings, with Applications to Zero-shot Dependency Parsing**](https://www.aclweb.org/anthology/N19-1162)

**ABSTRACT**: We introduce a novel method for multilingual transfer that utilizes deep contextual embeddings, pretrained in an unsupervised fashion. While contextual embeddings have been shown to yield richer representations of meaning compared to their static counterparts, aligning them poses a challenge due to their dynamic nature. To this end, we construct context-independent variants of the original monolingual spaces and utilize their map- ping to derive an alignment for the context- dependent spaces. This mapping readily supports processing of a target language, improv- ing transfer by context-aware embeddings. Our experimental results demonstrate the effectiveness of this approach for zero-shot and few-shot learning of dependency parsing. Specifically, our method consistently outperforms the previous state-of-the-art on 6 tested languages, yielding an improvement of 6.8 LAS points on average.

**NOTES**: Makes use of the "cloud" of embeddings for a token, (from the contextualized nature) and the "anchor" for the cloud (which allows you to make use of a bilingual dictionary for supervision).  Maintains the contextualized nature of the spaces.

[**Inoculation by Fine-Tuning: A Method for Analyzing Challenge Datasets**](https://www.aclweb.org/anthology/N19-1225)

**ABSTRACT**: Several datasets have recently been constructed to expose brittleness in models trained on existing benchmarks. While model performance on these challenge datasets is significantly lower compared to the original benchmark, it is unclear what particular weaknesses they reveal. For example, a challenge dataset may be difficult because it targets phenomena that current models cannot capture, or because it simply exploits blind spots in a model’s specific training set. We introduce inoculation by fine-tuning, a new analysis method for studying challenge datasets by exposing models (the metaphorical patient) to a small amount of data from the challenge dataset (a metaphor- ical pathogen) and assessing how well they can adapt. We apply our method to analyze the NLI “stress tests” (Naik et al., 2018) and the Adversarial SQuAD dataset (Jia and Liang, 2017). We show that after slight exposure, some of these datasets are no longer challenging, while others remain difficult. Our results indicate that failures on challenge datasets may lead to very different conclusions about models, training datasets, and the challenge datasets themselves.

**NOTES**: Using fine-tuning to better understanding why NN models fail on challenge / adversarial datasets.  Basically, if fine-tuning (on a small number of challenge dataset examples) fixes performance, then the failing reveals a weakness of the data.  Else, if it doesn’t help, then it’s a model weakness. Otherwise, if after fine-tuning, challenge does better and base does worse, then it’s a distribution issue/artifacts -- e.g., fine-tuning teaches the model to ignore something that had signal in normal dataset, but not in challenge, like last sentence in SQuAD challenge set.  
Counter-argument: We’ll never have a perfect train set, so is it really a dataset fail?

[Attention is not Explanation](https://www.aclweb.org/anthology/N19-1357)

**ABSTRACT**: Attention mechanisms have seen wide adoption in neural NLP models. In addition to improving predictive performance, these are often touted as affording transparency: models equipped with attention provide a distribution over attended-to input units, and this is often presented (at least implicitly) as communicating the relative importance of inputs. However, it is unclear what relationship exists between attention weights and model outputs. In this work we perform extensive experiments across a variety of NLP tasks that aim to assess the degree to which attention weights provide meaningful “explanations" for predictions. We find that they largely do not. For example, learned attention weights are frequently uncorrelated with gradient-based mea- sures of feature importance, and one can identify very different attention distributions that nonetheless yield equivalent predictions. Our findings show that standard attention modules do not provide meaningful explanations and should not be treated as though they do. Code to reproduce all experiments is available at https://github.com/successar/AttentionExplanation.

**NOTES**: A really nice look at whether attention weights really correlate with what is most informative to the model: spoiler -- not super really.  You should watch the video for sure, it was a fun and rather *bold* presentation, and the final line was provocative -- re: the fact that attn weights (and so accordingly the heatmaps from them) are over the *hidden states* of the words, not the words themselves, “Just because things have the same first dimension doesn’t mean you can lay one over the other.”  A fun listen.


### Vikas - favorite papers

**The most task innovative paper**
[Question Answering as an Automatic Evaluation Metric for News Article Summarization](https://www.aclweb.org/anthology/N19-1395)

Proved that trained QA models can be used as evaluator for summarizations and this evaluater is more effective than any other summarization evaluation metric. The discussion with the first author blew my mind. For summarization evaluation metrics, you need labelled training data but not for QA based evaluation. The QA performance is directly proportional to the summarization quality. 


[Shifting the Baseline:Single Modality Performance on Visual Navigation & QA](https://www.aclweb.org/anthology/N19-1197)

**Fun short paper**

Both language and image information is important. The ablation studies done in the paper are very informative and we can learn a lot from here. 

[Star-Transformer](https://arxiv.org/pdf/1902.09113.pdf)

If anyone understood this paper, please explain me too. I missed Mihai when I was reading this poster, hahaha...


### Steve

[Rethinking Complex Neural Network Architectures for Document Classification](https://www.aclweb.org/anthology/N19-1408)

You don't need attention, hierarchical structure, and sequence generation if you use a good regularizer.

[A Structural Probe for Finding Syntax in Word Representations](https://www.aclweb.org/anthology/N19-1419)

Very cool way of looking for syntax in BERT and ELMo.

[Linguistic Knowledge and Transferability of Contextual Representations](https://www.aclweb.org/anthology/N19-1112)

Another paper probing contextual word embeddings.

[Neural Semi-Markov Conditional Random Fields for Robust Character-Based Part-of-Speech](https://www.aclweb.org/anthology/N19-1280)

An character-level approach that jointly tokenizes and tags.

[Detection of Abusive Language: the Problem of Biased Datasets](https://www.aclweb.org/anthology/N19-1060)

Analysis of different abusive language datasets. A must-read for anyone doing abusive language or related areas. Shows why the Kaggle dataset is so easy. And concludes that we really should not use the Waseem dataset or other datasets based on topic sampling.

[Discontinuous Constituency Parsing with a Stack-Free Transition System and a Dynamic Oracle: Transition system for constituency parsing](https://www.aclweb.org/anthology/N19-1018)

Transition-based parsing that replaces the stack with a set.

[Mitigating Uncertainty in Document Classification](https://www.aclweb.org/anthology/N19-1316)

Introduces a simple dropout-based approach for estimating uncertainty of predictions. Also includes a margin-based approach to learn a contextual embedding space that better separates the classes.

### ---------------Yiyun's List -----------------
###1.Understand the black box

[**Analysis Methods in Neural Language Processing: A survey**](https://arxiv.org/pdf/1812.08951.pdf)

This is a survey that investigates different methods to understand what neural nets learn.

[**The emergence of number and syntax units in LSTM models**](https://arxiv.org/pdf/1903.07435.pdf)

**Abstract**:Recent work has shown that LSTMs trained on a generic language modeling objective capture syntax-sensitive generalizations such as longdistance number agreement. We have however no mechanistic understanding of how they accomplish this remarkable feat. Some have conjectured it depends on heuristics that do not truly take hierarchical structure into account. We present here a detailed study of the inner mechanics of number tracking in LSTMs at the single neuron level. We discover that long distance number information is largely managed by two “number units”. Importantly, the behaviour of these units is partially controlled by other units independently shown to track syntactic structure. We conclude that LSTMs are, to some extent, implementing genuinely syntactic processing mechanisms, paving the way to a more general understanding of grammatical encoding in LSTMs.

**NOTES**:Some of the analysis methods I found cool such as using ablation tests of each unit to find out which one encodes the number-agreement and they indeed found one unit for singular and one for plural. They also traced the activation change of forgetting gate to see whether the syntactic information is kept when it is needed. 

[**Understanding Learning Dynamics of Language Models with SVCCA**](https://arxiv.org/abs/1811.00225)

**Abstract**: Research has shown that neural models implicitly encode linguistic features, but there has been no research showing how these encodings arise as the models are trained. We present the first study on the learning dynamics of neural language models, using a simple and flexible analysis method called Singular Vector Canonical Correlation Analysis (SVCCA), which enables us to compare learned representations across time and across models, without the need to evaluate directly on annotated data. We probe the evolution of syntactic, semantic, and topic representations and find that part-of-speech is learned earlier than topic; that recurrent layers become more similar to those of a tagger during training; and embedding layers less similar. Our results and methods could inform better learning algorithms for NLP models, possibly to incorporate linguistic information more effectively.

**NOTES**:One interesting idea is to show how different linguistic information is correlated or de-correlated over the learning trajectory. 

[**Neural Language model as psycholinguistic subjects: Representation of syntactic state**](https://arxiv.org/abs/1903.03260)

**Abstract**: We deploy the methods of controlled psycholinguistic experimentation to shed light on the extent to which the behavior of neural network language models reflects incremental representations of syntactic state. To do so, we examine model behavior on artificial sentences containing a variety of syntactically complex structures. We test four models: two publicly available LSTM sequence models of English (Jozefowicz et al., 2016; Gulordava et al., 2018) trained on large datasets; an RNNG (Dyer et al., 2016) trained on a small, parsed dataset; and an LSTM trained on the same small corpus as the RNNG. We find evidence that the LSTMs trained on large datasets represent syntactic state over large spans of text in a way that is comparable to the RNNG, while the LSTM trained on the small dataset does not or does so only weakly.

**NOTES**:This one compared the human's syntactic preference with the neural nets' preference and found that RNNs are closer to human behavior than n-grams and transformer is even better (not in the paper but said in the presentation). My takeaway is that neural nets could imitate lots of human biases (not only the social bias such as gender bias but also linguistic preference)

[**Studying the Inductive biases of RNNs with Synthetic Variations of Neural Languages**](https://arxiv.org/abs/1903.06400)

**Abstract**:How do typological properties such as word order and morphological case marking affect the ability of neural sequence models to acquire the syntax of a language? Crosslinguistic comparisons of RNNs’ syntactic performance (e.g., on subject-verb agreement prediction) are complicated by the fact that any two languages differ in multiple typological properties, as well as by differences in training corpus. We propose a paradigm that addresses these issues: we create synthetic versions of English, which differ from English in one or more typological parameters, and generate corpora for those languages based on a parsed English corpus. We report a series of experiments in which RNNs were trained to predict agreement features for verbs in each of those synthetic languages. Among other findings, (1) performance was higher in subject-verb-object order (as in English) than in subject-object-verb order (as in Japanese), suggesting that RNNs have a recency bias; (2) predicting agreement with both subject and object (polypersonal agreement) improves over predicting each separately, suggesting that underlying syntactic knowledge transfers across the two tasks; and (3) overt morphological case makes agreement prediction significantly easier, regardless of word order.

**NOTES**:The method is interesting: they scrambled the English word order into different types and made up some case markers to see whether different word order makes the acquisition of subject and verb word agreement difficult. They tried to control the training data so they scrambled the English rather than using a different language.
They indeed show some word orders enable easy learning of subject-verb agreement. But they didn't some other stats such as the number of nouns as subjects or objects. I wonder to what extent this finding corresponds to any typological preference. One person askes whether the results reflected RNN's ability or the language complexity. 

###2.Rethink about Measures

[**Correlation Coefficients and Semantic Textual Similarity**](https://arxiv.org/pdf/1905.07790.pdf)

**Abstract**:A large body of research into semantic textual similarity has focused on constructing state-of-the-art embeddings using sophisticated modelling, careful choice of learning signals and many clever tricks. By contrast, little attention has been devoted to similarity measures between these embeddings, with cosine similarity being used unquestionably in the majority of cases. In this work, we illustrate that for all common word vectors, cosine similarity is essentially equivalent to the Pearson correlation coefficient, which provides some justification for its use. We thoroughly characterise cases where Pearson correlation (and thus cosine similarity) is unfit as similarity measure. Importantly, we show that Pearson correlation is appropriate for some word vectors but not others. When it is not appropriate, we illustrate how common nonparametric rank correlation coefficients can be used instead to significantly improve performance. We support our analysis with a series of evaluations on word-level and sentence-level semantic textual similarity benchmarks. On the latter, we show that even the simplest averaged word vectors compared by rank correlation easily rival the strongest deep representations compared by cosine similarity

**NOTES**:It provides an interesting view to look at a word vector as a random variable, and each dimension as an observation of this variable. I wonder other than the similarity measure what could be followup if we adopt this view.

[**What just happened? Evaluating Retrofitted Distributional Word Vectors**](https://www.aclweb.org/anthology/N19-1111)

**Abstract**:Recent work has attempted to enhance vector space representations using information from structured semantic resources. This process, dubbed retrofitting Faruqui et al. (2015), has yielded improvements in word similarity performance. Research has largely focused on the retrofitting algorithm, or on the kind of structured semantic resources used, but little research has explored why some resources perform better than others. We conducted a finegrained analysis of the original retrofitting process, and found that the utility of different lexical resources for retrofitting depends on two factors: the coverage of the resource and the evaluation metric. Our assessment suggests that the common practice of using correlation measures to evaluate increases in performance against full word similarity benchmarks 1) obscures the benefits offered by smaller resources, and 2) overlooks incremental gains in word similarity performance. We propose root-mean-square error (RMSE) as an alternative evaluation metric, and demonstrate that correlation measures and RMSE sometimes yield opposite conclusions concerning the efficacy of retrofitting. This point is illustrated by word vectors retrofitted with novel treatments of the FrameNet data (Fillmore and Baker, 2010).

**NOTES**:This one also looks up how we should evaluate the result and it showed that depending which measure we adopt the results could vary a lot. But I don't think from their presentation they demonstrated why RMSE is better.



