# RAG_implementation_evaluation
rag development 
we taken a dataset that contains questions and answers for building RAG
1. load dataset
2. apply prepprocessing like converting all to lowercase,stopwrd removal,lematization
3. used bert tokenizer on "question" filed for creating embedings we can also follow other embeding tecniques like gpt, openai's davincei etc., i used bert because length of each question is less than 514 tokens which is thelimit of bert
4. then stored the vectors in vector database i used FAISS (Facebook AI similarity search) with cosine similarity distance that returns te cosine distance score for the retrived articles.
5. when user asks a question on this question applying same preprocessing steps that we followed before then converting to vector embedings with bert
6. then indexing over the faiss vector data base and get the top 5 results, (we can also apply a thershold on the top 5 then pass to llm model.)
7. then pass the top 5 results with a prompt to the llm model to decide which is the best answer for the user this improves search results. ALso we can intilize conversational_buffer_window_memory in langchain if it is a chatbot so that model can easily understand the query based on history conversation
8. now model will return the most relavent answer improving the retrival.

RAG Evaluation
I preared a validation dataset that contains question and actual answer and LLM generated answer.
Now we are evaluating the RAG model performance by comparing LLMgenerated answer vs actual answer.
here we used ROUGE,BLEU and perplexity to evaluate the model performance since the datset is q&A and the answers are LLm generative model generated responses based on the given q&A we use this 3 techniques, If the dataset is labelled data then we can use Precision, Recall with F1 scores.


ROUGE:
ROUGE (Recall-Oriented Understudy for Gisting Evaluation) measures the overlap between the generated text and a reference text, particularly in terms of n-grams (sequences of words). Common variants are ROUGE-1 (unigram overlap), ROUGE-2 (bigram overlap), and ROUGE-L (longest common subsequence).
A ROUGE score of 0 means there is no overlap between the generated text and the reference.
In your example, the ROUGE score of 0 indicates no overlap, meaning the generated text and reference text are not similar based on n-grams.


BLEU Score
BLEU (Bilingual Evaluation Understudy) is another metric that compares the overlap between the generated text and reference text, but it focuses on precision—how many words in the generated text also appear in the reference. It's commonly used in machine translation but can be used for text generation as well.
A BLEU score of 1.0 means perfect overlap, indicating that the generated text and reference are nearly identical.

Perplexity:
Perplexity measures how well a language model predicts the next word in a sequence. A lower perplexity score indicates that the model finds the text easier to predict, meaning the text is more "fluent" or natural. In contrast, higher perplexity suggests the text is harder for the model to predict and is potentially less fluent or coherent.


Precision: Measures how many of the retrieved documents are relevant.
Recall: Measures how many of the relevant documents are retrieved.
F1 Score: The harmonic mean of precision and recall. It balances precision and recall into a single score.
