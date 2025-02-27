

# README_MD



## How to execute

The fine tuned model  ` "./lora_bert_model_v2"` is provided along with the script

Execute the code with 

```
docker build . -t ai-assignment
docker run -it --mount type=bind,source="$(pwd)",target=/usr/src/app/ ai-assignment bash -c "python3 lora_finetune.py"
```

After placing the model in the same directory.





## Methods

###  Data

The dataset consists of Japanese text, where the task is to classify whether a given text is a question or not. Unlike English, where questions are often marked by a question mark (`?`), Japanese questions frequently omit explicit punctuation, making it challenging to rely on simple rule-based heuristics. Additionally, the dataset exhibits a diverse vocabulary, including various levels of formality, sentence-ending particles (e.g., `か`, `の`), and implicit interrogative structures. This complexity suggests that traditional machine learning approaches such as Support Vector Machines (SVM) with handcrafted features would be insufficient, as designing effective features manually would be both time-consuming and error-prone.

Given these observations, I determined that a deep learning-based approach leveraging contextual embeddings would be more suitable. Transformer-based models, particularly BERT, are well-suited for such tasks because they can capture contextual dependencies and nuanced linguistic structures that define interrogative sentences in Japanese.



## Model

Since my computing resources are limited—I do not have access to a GPU but instead have an Apple M2 chip—I opted to utilize Apple’s `mps` backend for efficient training. Rather than fine-tuning a large transformer model from scratch, which would be computationally expensive, I used **LoRA (Low-Rank Adaptation)** to reduce memory and compute requirements. LoRA modifies only a small subset of model parameters, significantly lowering the cost of training while retaining strong performance.

For the base model, I selected **`cl-tohoku/bert-base-japanese`**, a pre-trained Japanese BERT model developed by the Tohoku University NLP group. This model has been widely used in Japanese NLP tasks and has demonstrated strong performance on various benchmarks, including text classification and named entity recognition. The findings at the developers shows that this model performs well on text classification benchmarks(MARC-ja) as well as other text tasks defined in [JGLUE](https://github.com/yahoojapan/JGLUE). 

By leveraging LoRA with `cl-tohoku/bert-base-japanese`, I was able to efficiently fine-tune the model on my dataset while maintaining high classification accuracy despite hardware constraints.

I split `training.csv` to use 80% for training and 20% for validation.  

## Results
Based on the validation set, the model achieves 
 99.9% accuracy, 99.8% f1 score, 99.8% precision and 99.9% recall.

![image-20250227150314144](/Users/ts-shinka.mori/Library/Application Support/typora-user-images/image-20250227150314144.png)

For reference here are the sentences it misclassified. Upon looking at the data, we observed that some of the misclassified questions may me wrongly labeled in the original data; for example [岡部はアメリカ遠征の際に誰と比較して自身の社会常識等が身についていないことを自覚した] is a statement, not a question, and [なぜ彼は消息を知らせてこないのか。] could be interpreted as a question. This shows that the model may underperform slightly on ambiguous sentences.

|predicted|true|sentence|
|----|---|----|
|True | False | 貸出制限は飛騨市に居住・通勤・通学している者。|
|False | True | 岡部はアメリカ遠征の際に誰と比較して自身の社会常識等が身についていないことを自覚した。|
|True | False | なぜ彼は消息を知らせてこないのか。|
|False | True | ミレーはいつポーリーヌ=ヴィルジニー・オノと結婚した。|
|True | False | この悲しい時をどう耐えたらいいのだろう。|
|False | True | 『AHistoryofInventions,Discoveries,andOrigins』と『ExtraordinaryPopularDelusionsandtheMadnessofCrowds』と、どっちの方が先に書かれた。|
|True | False | 死相を出すというのかな。|




## Future work
While I focused on fine-tuning BERT with LoRA, other methods could be explored for this task.

Generative models like GPT or T5 could be fine-tuned with LoRA to classify text by generating labels ("question" or "not a question"). They may better capture implicit interrogative structures but risk hallucinating labels, especially with limited training data. Techniques like confidence calibration or prompt engineering could help mitigate this issue.

Given the small label set, RNN-based models (LSTMs or GRUs) could serve as a lightweight alternative. They efficiently capture sequential dependencies but struggle with long-range dependencies, scalability, and contextual understanding compared to transformers. However, when paired with pretrained word embeddings, RNNs may be effective in low-resource environments.


## Q & A



- \-  Let’s say you don’t have a premade dataset, and so you need to process whole documents of text to extract sentences containing questions, how would you approach the problem?
  - I would explore two methods, rule-based or using an LLM.
  - For rule-based method, we can take advantage of the common question markers such as か, の, だろうか, でしょうか, or explicit question marks (？) and use pattern matching with regex to extract sentences.  Another option is to use a dependency parser such as SpaCy to check sentence-ending structures commonly associated with questions.
  - Alternatively, you can pass the documents through an LLM, for example `Sarashina2-8x70B` from SB Institutions and use prompt engineering to extract sentences.  
- \-  The customer wants to understand which questions come up most frequently, and have an overview of semantically unique questions asked, how would you go about this? Two questions may have dissimilar words but very similar meaning, and thus could be treated as a single, unique question.
 - Since two questions may have different words but similar meanings, I would use sentence embeddings (e.g., Sentence-BERT, cl-tohoku/bert-base-japanese, or mJPN-BERT) to encode each question into a high-dimensional vector. Cosine similarity can then be used to compare sentences and measure their similarity. A cosine similarity score(~1) indicates high semantic similarity and conversely a low cosine similarity (~0) indicates low semantic similarity.
  To find distinct question groups, I would:
  - Vectorize the questions using sentence embeddings.
  - Reduce dimensionality using UMAP to make clustering more effective.
  - Cluster the embeddings using HDBSCAN (density-based clustering) or K-Means.
  - HDBSCAN is preferred if the customer wants natural, well-separated clusters that vary in size.
  - K-Means is useful if fixed-size clusters are acceptable.
  - Extract cluster centroids as representative questions. These centroids would be the most typical questions within each cluster.
Once questions are grouped into clusters, I would:
	- Count the number of occurrences per cluster.
	- Rank clusters by frequency to highlight the most commonly asked questions.
 
- \-  Imagine your data is call transcripts from an e-commerce business. Their customers have various topics they may contact the business with, but among them would be statements about concerns on certain products, or issues with the website, etc. How would you go about extracting concerns raised?
  - To find sentences specficially raising a concern, I can use an LLM to classify texts that mentions a concern by using prompts such as "You are a call center agent; label whether the customer utterance is experessing a concern. "
  - Once the texts are extracted, I can vectorize the sentences using sentence embeddings sucn as `Sentence-BERT` or `cl-tohoku/bert-base-japanese` and use UMAP for dimensionality reduction, followed by HDBSCAN to cluster concerns into key themes.  I can then manually categorize the clusters or use an LLM to summarize the clusters and assign a label name.  Once they are labeled, I can infer the representative concerns from each cluster by using centroid sentences.







