# Lecture Series III (Deep Models for Text and Sequences)

## Introduction
* Rare words characterize documents better but since the words are rare, 
training is hard and more examples are needed for a better model.
* There are completely different words which are really similar like (cat and kitty).
* These can be solved using unsupervised learning. 
**Prime Intuition: Similar words occur in similar context.** 
* Can be achieved using embeddings. Similar words have similar embeddings thus 
signifying a similar context.

## Word2Vec
* Each word in the vector is mapped to a embedding.
* Intialy the vector is random.
* THe embedding is used to predict context of the word. A simple choice of context
can be to choose words in a window around it and then train a simple model like logistic
regression to check the word's context.
* The embeddings cluster together by themselves. Can be verified using
  * Nearest neighbor lookup
  * Tool called tSNE that projects the points on a 2D space at the same time maintains
   the relationship b/w embeddings.
* Closeness of embedding vectors are measured using cosine distance.
* Embedding vectors are often normalized to have unit norm.
* In estimating other words in context of input words, 
  * The word is first converted to its embedding.
  * Sent into a linear model.
  * Softmax is performed and compared against actual value.
  * If the vocabulary is very large, computing softmax becomes very hard.
  * To solve this, we can use **sampled softmax** where except the target word,
  all other words are randomly sampled( since they are 0 anyways). And softmax
  is computed using these samples.
* Using embedding vectors we can perform semantic and syntactic analogies.
This makes it possible to perform arithmetic(sort off) using words. 

## RNN
* The intuition is similar to CNN. In CNN weights were shared across space. 
Here weights are shared across time.
* Since the input varies across time, a natural idea would be to build a model for each
instance of time for the input. The model can also share the state of the previous instance
of the model. This would make the network.
* Rather this can be achieved in a loop where the weight stays constant across all instances
 of time which is connected to the input and another part which contains the previous instance
  of the model.
* This results in numerous correlated updates for the same weight matrix which will result in 
extremely poor sgd.
* This causes two problems
  * Exploding Gradients: Solved using gradient clipping i.e, Compute the norm and shrink the gradient
   when the norm grows too big.
  * Vanishing Gradients: Makes model remember only recent events and forget the distant past. 
  This can be solved using LSTMs
* L2 Regularization works. Dropout regularization works only on the input and 
output not on the recurrent connections/

   

### LSTM
* Stands for Long Short Term Memory.
* Replaces the central weight matrix(and additional non-linearities) with an LSTM cell.
* The LSTM cell has three 4 components.
  * A central memory(Same dimensions as the Weight matrix/NN replacing).
  * A gate controlling the write to the memory.
  * A gate controlling the read from the memory.
  * A gate controlling the rate at which data is retained in memory.
* All the three gates represent a sigmoid function which is continuous and derivable.
* Additionally the gate controlling the read operation is prefixed with a hyperbolic tan function
to keep the input between 0 and 1.
  
  
## Beam Search
* Application of RNN used to construct words/sentences.
* A set of letters is predicted(called hypotheses) and the ones with low probabilities
 are pruned to avoid exponential complication while prediction.