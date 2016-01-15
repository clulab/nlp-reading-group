# Proposed by Mihai:

## Learning Natural Language Inference with LSTM 
http://arxiv.org/pdf/1512.08849v1.pdf

(8 pages) Natural language inference (NLI) is a fun- damentally important task in natural lan- guage processing that has many applications. The recently released Stanford Nat- ural Language Inference (SNLI) corpus has made it possible to develop and eval- uate learning-centered methods such as deep neural networks for the NLI task. In this paper, we propose a special long short-term memory (LSTM) architecture for NLI. Our model builds on top of a recently proposed neutral attention model for NLI but is based on a significantly dif- ferent idea. Instead of deriving sentence embeddings for the premise and the hy- pothesis to be used for classification, our solution uses a matching-LSTM that per- forms word-by-word matching of the hy- pothesis with the premise. This LSTM is able to place more emphasis on important word-level matching results. In particu- lar, we observe that this LSTM remembers important mismatches that are critical for predicting the contradiction or the neutral relationship label. Our experiments on the SNLI corpus show that our model outper- forms the state of the art, achieving an ac- curacy of 86.1% on the test data.


## The Goldilocks Principle: Reading Children's Books with Explicit Memory Representations - JOHN, 8
http://arxiv.org/pdf/1511.02301v3.pdf

(9 pages) We introduce a new test of how well language models capture meaning in chil- dren’s books. Unlike standard language modelling benchmarks, it distinguishes the task of predicting syntactic function words from that of predicting lower- frequency words, which carry greater semantic content. We compare a range of state-of-the-art models, each with a different way of encoding what has been previ- ously read. We show that models which store explicit representations of long-term contexts outperform state-of-the-art neural language models at predicting seman- tic content words, although this advantage is not observed for syntactic function words. Interestingly, we find that the amount of text encoded in a single memory representation is highly influential to the performance: there is a sweet-spot, not too big and not too small, between single words and full sentences that allows the most meaningful information in a text to be effectively retained and recalled. Fur- ther, the attention over such window-based memories can be trained effectively through self-supervision. We then assess the generality of this principle by ap- plying it to the CNN QA benchmark, which involves identifying named entities in paraphrased summaries of news articles, and achieve state-of-the-art performance.


## Deep Compositional Question Answering with Neural Module Networks
http://arxiv.org/pdf/1511.02799v2.pdf

(8 pages) Visual question answering is fundamentally composi- tional in nature—a question like where is the dog? shares substructure with questions like what color is the dog? and where is the cat? This paper seeks to simultaneously exploit the representational capacity of deep networks and the com- positional linguistic structure of questions. We describe a procedure for constructing and learning neural module net- works, which compose collections of jointly-trained neural “modules” into deep networks for question answering. Our approach decomposes questions into their linguistic sub- structures, and uses these structures to dynamically instan- tiate modular networks (with reusable components for rec- ognizing dogs, classifying colors, etc.). The resulting com- pound networks are jointly trained. We evaluate our ap- proach on two challenging datasets for visual question an- swering, achieving state-of-the-art results on both the VQA natural image dataset and a new dataset of complex ques- tions about abstract shapes.


## Learning to Compose Neural Networks for Question Answering
http://arxiv.org/pdf/1601.01705v1.pdf

(8 pages) We describe a question answering model that applies to both images and structured knowl- edge bases.The model uses natural language strings to automatically assemble neural net- works from a collection of composable mod- ules. Parameters for these modules are learned jointly with network-assembly parameters via reinforcement learning, with only (world, question, answer) triples as supervision. Our approach, which we term a dynamic neural module network, achieves state-of-the-art re- sults on benchmark datasets in both visual and structured domains.


## Language to Logical Form with Neural Attention - ROBERT, 10
http://arxiv.org/pdf/1601.01280v1.pdf

(8 pages) Semantic parsing aims at mapping natural language to machine interpretable meaning representations. Traditional approaches rely on high-quality lexicons, manually-built tem- plates, and linguistic features which are either domain- or representation-specific. In this pa- per, we present a general method based on an attention-enhanced sequence-to-sequence model. We encode input sentences into vec- tor representations using recurrent neural net- works, and generate their logical forms by conditioning the output on the encoding vec- tors. The model is trained in an end-to-end fashion to maximize the likelihood of target logical forms given the natural language in- puts. Experimental results on four datasets show that our approach performs competi- tively without using hand-engineered features and is easy to adapt across domains and mean- ing representations.


## Basic Reasoning with Tensor Product Representations - CLAY, 11
http://arxiv.org/abs/1601.02745 

(12 pages (very theoretical)) In this paper we present the initial development of a general theory for mapping inference in predicate logic to computation over Tensor Product Representations (TPRs; Smolensky (1990), Smolensky & Legendre (2006)). After an initial brief synopsis of TPRs (Section 0), we begin with particular examples of inference with TPRs in the ‘bAbI’ question-answering task of Weston et al. (2015) (Section 1). We then present a simplification of the general analysis that suffices for the bAbI task (Section 2). Finally, we lay out the general treatment of inference over TPRs (Section 3). We also show the simplification in Section 2 derives the inference methods described in Lee et al. (2016); this shows how the simple methods of Lee et al. (2016) can be formally extended to more general reasoning tasks.


## Deep Neural Decision Forests - DANE, 3
http://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Kontschieder_Deep_Neural_Decision_ICCV_2015_paper.pdf

(8 pages (Training decision trees using backpropagation)) We present Deep Neural Decision Forests – a novel ap- proach that unifies classification trees with the representa- tion learning functionality known from deep convolutional networks, by training them in an end-to-end manner. To combine these two worlds, we introduce a stochastic and differentiable decision tree model, which steers the rep- resentation learning usually conducted in the initial lay- ers of a (deep) convolutional network. Our model differs from conventional deep networks because a decision for- est provides the final predictions and it differs from con- ventional decision forests since we propose a principled, joint and global optimization of split and leaf node param- eters. We show experimental results on benchmark machine learning datasets like MNIST and ImageNet and find on- par or superior results when compared to state-of-the-art deep models. Most remarkably, we obtain Top5-Errors of only 7.84%/6.38% on ImageNet validation data when in- tegrating our forests in a single-crop, single/seven model GoogLeNet architecture, respectively. Thus, even without any form of training data set augmentation we are improv- ing on the 6.67% error obtained by the best GoogLeNet ar- chitecture (7 models, 144 crops).


## Return of Frustratingly Easy Domain Adaptation
http://arxiv.org/pdf/1511.05547v2.pdf

(7 pages (not NLP)) Unlike human learning, machine learning often fails to handle changes between training (source) and test (target) input distributions. Such domain shifts, common in practical scenarios, severely damage the performance of conventional machine learning methods. Supervised domain adaptation methods have been proposed for the case when the target data have labels, including some that perform very well despite being "frustratingly easy" to implement. However, in practice, the target domain is often unlabeled, requiring unsupervised adaptation. We propose a simple, effective, and efficient method for unsupervised domain adaptation called CORrelation ALignment (CORAL). CORAL minimizes domain shift by aligning the second-order statistics of source and target distributions, without requiring any target labels. Even though it is extraordinarily simple--it can be implemented in four lines of Matlab code--CORAL performs remarkably well in extensive evaluations on standard benchmark datasets.


## Solving Geometry Problems: Combining Text and Diagram Interpretation - ZECKY, 13
http://geometry.allenai.org/assets/emnlp2015.pdf

(9 pages) This paper introduces GEOS, the first au- tomated system to solve unaltered SAT ge- ometry questions by combining text un- derstanding and diagram interpretation. We model the problem of understanding geometry questions as submodular opti- mization, and identify a formal problem description likely to be compatible with both the question text and diagram. GEOS then feeds the description to a geometric solver that attempts to determine the cor- rect answer. In our experiments, GEOS achieves a 49% score on official SAT ques- tions, and a score of 61% on practice ques- tions.1 Finally, we show that by integrat- ing textual and visual information, GEOS boosts the accuracy of dependency and se- mantic parsing of the question text.


## Addressing a Question Answering Challenge by Combining Statistical Methods with Inductive Rule Learning and Reasoning - PETER, 12
http://www.public.asu.edu/~cbaral/papers/aaai2016-sub.pdf

(6 pages) A group of researchers from Facebook has recently pro- posed a set of 20 question-answering tasks (Facebook’s bAbl dataset) as a challenge for the natural language un- derstanding ability of an intelligent agent. These tasks are designed to measure various skills of an agent, such as: fact based question-answering, simple induction, the ability to find paths, co-reference resolution and many more. Their goal is to aid in the development of sys- tems that can learn to solve such tasks and to allow a proper evaluation of such systems. They show existing systems cannot fully solve many of those toy tasks. In this work, we present a system that excels at all the tasks except one. The proposed model of the agent uses the Answer Set Programming (ASP) language as the pri- mary knowledge representation and reasoning language along with the standard statistical Natural Language Processing (NLP) models. Given a training dataset con- taining a set of narrations, questions and their answers, the agent jointly uses a translation system, an Induc- tive Logic Programming algorithm and Statistical NLP methods to learn the knowledge needed to answer simi- lar questions. Our results demonstrate that the introduc- tion of a reasoning module significantly improves the performance of an intelligent agent.


## Extracting Thee-Structured Representations of Trained Networks - MIHAI, 2
https://papers.nips.cc/paper/1152-extracting-tree-structured-representations-of-trained-networks.pdf

(7 pages) A significant limitation of neural networks is that the represen- tations they learn are usually incomprehensible to humans. We present a novel algorithm, TREPAN, for extracting comprehensible, symbolic representations from trained neural networks. Our algo- rithm uses queries to induce a decision tree that approximates the concept represented by a given network. Our experiments demon- strate that TREPAN is able to produce decision trees that maintain a high level of fidelity to their respective networks while being com- prehensible and accurate. Unlike previous work in this area, our algorithm is general in its applicability and scales well to large net- works and problems with high-dimensional input spaces.


## sense2vec - A Fast and Accurate Method for Word Sense Disambiguation In Neural Word Embeddings - GUS, 14
http://arxiv.org/pdf/1511.06388v1.pdf

(7 pages) Neural word representations have proven useful in Natural Language Processing (NLP) tasks due to their ability to efficiently model complex semantic and syntactic word relationships. However, most techniques model only one representation per word, despite the fact that a single word can have multiple meanings or "senses". Some techniques model words by using multiple vectors that are clustered based on context. However, recent neural approaches rarely focus on the application to a consuming NLP algorithm. Furthermore, the training process of recent word-sense models is expensive relative to single-sense embedding processes. This paper presents a novel approach which addresses these concerns by modeling multiple embeddings for each word based on supervised disambiguation, which provides a fast and accurate way for a consuming NLP model to select a sense-disambiguated embedding. We demonstrate that these embeddings can disambiguate both contrastive senses such as nominal and verbal senses as well as nuanced senses such as sarcasm. We further evaluate Part-of-Speech disambiguated embeddings on neural dependency parsing, yielding a greater than 8% average error reduction in unlabeled attachment scores across 6 languages.


## Reasoning in Vector Space: An Exploratory Study of Question Answering
http://arxiv.org/pdf/1511.06426v2.pdf

(11 pages) Question answering tasks have shown remarkable progress with distributed
vector representations. In this paper, we look into the recently proposed
Facebook 20 tasks (FB20). Finding the answers for questions in FB20 requires
complex reasoning. Because the previous work on FB20 consists of end-to-end
models, it is unclear whether errors come from imperfect understanding of
semantics or in certain steps of the reasoning. To address this issue, we
propose two vector space models inspired by tensor product representation (TPR)
to perform analysis, knowledge representation, and reasoning based on
common-sense inference. We achieve near-perfect accuracy on all categories,
including positional reasoning and pathfinding that have proved difficult for
all previous approaches due to the special two-dimensional relationships
identified from this study. The exploration reported in this paper and our
subsequent work on generalizing the current model to the TPR formalism suggest
the feasibility of developing further reasoning models in tensor space with
learning capabilities.

# Proposed by Michael:

## Collaborative Filering for Implicit Feedback Datasets
http://yifanhu.net/PUB/cf.pdf

(10 pages) A common task of recommender systems is to improve customer experience through personalized recommendations based on prior implicit feedback. These systems passively track different sorts of user behavior, such as purchase history, watching habits and browsing activity, in order to model user preferences. Unlike the much more extensively researched explicit feedback, we do not have any direct input from the users regarding their preferences. In particular, we lack substantial evidence on which products consumer dislike. In this work we identify unique properties of implicit feedback datasets. We propose treating the data as indication of positive and negative preference associated with vastly varying confidence levels. This leads to a factor model which is especially tailored for implicit feedback recommenders. We also suggest a scalable optimization procedure, which scales linearly with the data size. The algorithm is used successfully within a recommender system for television shows. It compares favorably with well tuned implementations of other known methods. In addition, we offer a novel way to give explanations to recommendations given by this factor model.


##LTSM: A Search Space Odyssey - MICHAEL, 6 (PLUS BLOG)
http://arxiv.org/pdf/1503.04069v1.pdf

(10 pages + supplementary material) Several variants of the Long Short-Term Memory (LSTM) architecture for recurrent neural networks have been proposed since its inception in 1995. In recent years, these networks have become the state-of-the-art models for a variety of machine learning problems. This has led to a renewed interest in understanding the role and utility of various computational components of typical LSTM variants. In this paper, we present the first large-scale analysis of eight LSTM variants on three representative tasks: speech recognition, handwriting recognition, and polyphonic music modeling. The hyperparameters of all LSTM variants for each task were optimized separately using random search and their importance was assessed using the powerful fANOVA framework. In total, we summarize the results of 5400 experimental runs (≈ 15 years of CPU time), which makes our study the largest of its kind on LSTM networks. Our results show that none of the variants can improve upon the standard LSTM architecture significantly, and demonstrate the forget gate and the output activation function to be its most critical components. We further observe that the studied hyperparameters are virtually independent and derive guidelines for their efficient adjustment.

##Sequence to Sequence Learning with Neural Networks - JOSH, 7
http://arxiv.org/pdf/1409.3215v3.pdf

(9 pages) Deep Neural Networks (DNNs) are powerful models that have achieved excellent performance on difficult learning tasks. Although DNNs work well whenever large labeled training sets are available, they cannot be used to map sequences to sequences. In this paper, we present a general end-to-end approach to sequence learning that makes minimal assumptions on the sequence structure. Our method uses a multilayered Long Short-Term Memory (LSTM) to map the input sequence to a vector of a fixed dimensionality, and then another deep LSTM to decode the target sequence from the vector. Our main result is that on an English to French translation task from the WMT’14 dataset, the translations produced by the LSTM achieve a BLEU score of 34.8 on the entire test set, where the LSTM’s BLEU score was penalized on out-of-vocabulary words. Additionally, the LSTM did not have difficulty on long sentences. For comparison, a phrase-based SMT system achieves a BLEU score of 33.3 on the same dataset. When we used the LSTM to rerank the 1000 hypotheses produced by the aforementioned SMT system, its BLEU score increases to 36.5, which is close to the previous best result on this task. The LSTM also learned sensible phrase and sentence representations that are sensitive to word order and are relatively invariant to the active and the passive voice. Finally, we found that reversing the order of the words in all source sentences (but not target sentences) improved the LSTM’s performance markedly, because doing so introduced many short term dependencies between the source and the target sentence which made the optimization problem easier. 


##Feedforward Sequential Memory Networks: A New Structure to Learn Long-term Dependency
http://arxiv.org/pdf/1512.08301v2.pdf

(12 pages) In this paper, we propose a novel neural network structure, namely feedforward sequential memory networks (FSMN), to model long-term dependency in time series without using recurrent feedback. The proposed FSMN is a standard fully-connected feedforward neural network equipped with some learnable memory blocks in its hidden layers. The memory blocks use a tapped-delay line structure to encode the long context information into a fixed-size representation as short-term memory mechanism. We have evaluated the proposed FSMNs in several standard benchmark tasks, including speech recognition and language modelling. Experimental results have shown FSMNs significantly outperform the conventional recurrent neural networks (RNN), including LSTMs, in modeling sequential signals like speech or language. Moreover, FSMNs can be learned much more reliably and faster than RNNs or LSTMs due to the inherent non-recurrent model structure.


##Understanding LSTM Networks
http://colah.github.io/posts/2015-08-Understanding-LSTMs/

**This is a blog post that goes through the concept of the LSTM architecture.**


##On the saddle point problem for non-convex optimization
http://arxiv.org/pdf/1405.4604v2.pdf

(12 pages) A central challenge to many fields of science and engineering involves minimizing non-convex error functions over continuous, high dimensional spaces. Gradient descent or quasi-Newton methods are almost ubiquitously used to perform such minimizations, and it is often thought that a main source of difficulty for the ability of these local methods to find the global minimum is the proliferation of local minima with much higher error than the global minimum. Here we argue, based on results from statistical physics, random matrix theory, and neural network theory, that a deeper and more profound difficulty originates from the proliferation of saddle points, not local minima, especially in high dimensional problems of practical interest. Such saddle points are surrounded by high error plateaus that can dramatically slow down learning, and give the illusory impression of the existence of a local minimum. Motivated by these arguments, we propose a new algorithm, the saddle-free Newton method, that can rapidly escape high dimensional saddle points, unlike gradient descent and quasi-Newton methods. We apply this algorithm to deep neural network training, and provide preliminary numerical evidence for its superior performance.


# Proposed by Marco:

## Show and Tell: A Neural Image Caption Generator - MARCO, 9
http://arxiv.org/pdf/1411.4555v2.pdf

(9 pages) Automatically describing the content of an image is a fundamental problem in artificial intelligence that connects computer vision and natural language processing. In this paper, we present a generative model based on a deep recurrent architecture that combines recent advances in computer vision and machine translation and that can be used to generate natural sentences describing an image. The model is trained to maximize the likelihood of the target description sentence given the training image. Experiments on several datasets show the accuracy of the model and the fluency of the language it learns solely from image descriptions. Our model is often quite accurate, which we verify both qualitatively and quantitatively. For instance, while the current state-of-the-art BLEU-1 score (the higher the better) on the Pascal dataset is 25, our approach yields 59, to be compared to human performance around 69. We also show BLEU-1 score improvements on Flickr30k, from 56 to 66, and on SBU, from 19 to 28. Lastly, on the newly released COCO dataset, we achieve a BLEU-4 of 27.7, which is the current state-of-the-art.

## A Transition-based Algorithm for AMR Parsing - BECKY, 5
http://www.cs.brandeis.edu/~xuen/publications/wang-2015-naacl.pdf

(10 pages) We present a two-stage framework to parse a sentence into its Abstract Meaning Representation (AMR). We first use a dependency parser to generate a dependency tree for the sentence. In the second stage, we design a novel transition-based algorithm that transforms the dependency tree to an AMR graph. There are several advantages with this approach. First, the dependency parser can be trained on a training set much larger than the training set for the tree-to-graph algorithm, resulting in a more accurate AMR parser overall. Our parser yields an improvement of 5% absolute in F-measure over the best previous result. Second, the actions that we design are linguistically intuitive and capture the regularities in the mapping between the dependency structure and the AMR of a sentence. Third, our parser runs in nearly linear time in practice in spite of a worst-case complexity of O(n 2 ).

## Unsupervised Morphology Induction Using Word Embeddings - NICK, 1
http://www.aclweb.org/anthology/N15-1186.pdf

(11 pages) We present a language agnostic, unsupervised method for inducing morphological transformations between words. The method relies on certain regularities manifest in high-dimensional vector spaces. We show that this method is capable of discovering a wide range of morphological rules, which in turn are used to build morphological analyzers. We evaluate this method across six different languages and nine datasets, and show significant improvements across all languages.

# Proposed by Enrique:

## Improving Topic Models with Latent Feature Word Representations - ENRIQUE, 4
https://tacl2013.cs.columbia.edu/ojs/index.php/tacl/article/view/582/158

(15 pages) Probabilistic topic models are widely used to discover latent topics in document collections, while latent feature vector representations of words have been used to obtain high performance in many NLP tasks. In this paper, we extend two different Dirichlet multinomial topic models by incorporating latent feature vector representations of words trained on very large corpora to improve the word-topic mapping learnt on a smaller corpus. Experimental results show that by using information from the external corpora, our new models produce significant improvements on topic coherence, document clustering and document classification tasks, especially on datasets with few or short documents.

## Improving Distributional Similarity with Lessons Learned from Word Embeddings
https://tacl2013.cs.columbia.edu/ojs/index.php/tacl/article/view/570/124

(15 pages) Recent trends suggest that neural-network-inspired word embedding models outperform traditional count-based distributional models on word similarity and analogy detection tasks. We reveal that much of the performance gains of word embeddings are due to certain system design choices and hyperparameter optimizations, rather than the embedding algorithms themselves. Furthermore, we show that these modifications can be transferred to traditional distributional models, yielding similar gains. In contrast to prior reports, we observe mostly local or insignificant performance differences between the methods, with no global advantage to any single approach over the others.

## A Neural Network Approach to Context-Sensitive Generation of Conversational Responses
http://arxiv.org/pdf/1506.06714v1

(11 pages) We present a novel response generation system that can be trained end to end on large quantities of unstructured Twitter conversations. A neural network architecture is used to address sparsity issues that arise when integrating contextual information into classic statistical models, allowing the system to take into account previous dialog utterances. Our dynamic-context generative models show consistent gains over both context-sensitive and non-context-sensitive Machine Translation and Information Retrieval baselines.

