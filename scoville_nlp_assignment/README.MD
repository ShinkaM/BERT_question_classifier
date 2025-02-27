

# README_MD



## How to execute

The fine tuned model  ` "./lora_bert_model_v2"` is provied along with the script

Execute the code with 

```
docker build . -t ai-assignment
docker run -it --mount type=bind,source="$(pwd)",target=/usr/src/app/ ai-assignment bash -c "python3 lora_finetune.py"
```

After placing the model in the same directory.





## Methods

###  Reasoning

I chose my method based on my observations about the data, my computing resources, and the complexity of the task.

Looking at the data, I noticed that the data is in Japanese and that the vocabulary is very diverse; the questions are not marked by a question mark, meaning that feature engineering would be difficult if using traditional methods such as SVM.  For tasks like this, methods that can create more representative features is more ideal, such as BERT models.

Since I do not have access to a GPU but have an Apple M2, I chose to utilize the mps and use LORA since it is less computationally expensive than fine tuning from scratch.  I chose to use the pre-trained model `cl-tohoku/bert-base-japanese` since it has been demonstrated to do well on Japanese tasks.  

### Data
I split `training.csv` to use 80% for training and 20% for validation.  

## Results
Based on the validation set, the model achieves 
 99.9% accuracy, 99.8% f1 score, 99.8% precision and 99.9% recall.

![image-20250227150314144](/Users/ts-shinka.mori/Library/Application Support/typora-user-images/image-20250227150314144.png)

For reference here are the sentences it misclassified.

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

While I stopped at fine-tuning BERT with LORA there are other methods to explore.

For example fine tuning a generative model such as  with LORA may be effective, though there is a risk of hallucinationg the labels.

Another approach would be to use an RNN since the label set is small. 



## Q & A



- \-  Let’s say you don’t have a premade dataset, and so you need to process whole documents of text to extract sentences containing questions, how would you approach the problem?
  - A simple rule-based method can be used to extract sentences that may be questions. For Japanese, sentences ending in ですか、でしょうか、か etc are likely to be questions so this pattern can be used to extract quesitons.
- \-  The customer wants to understand which questions come up most frequently, and have an overview of semantically unique questions asked, how would you go about this? Two questions may have dissimilar words but very similar meaning, and thus could be treated as a single, unique question.
  - To match sentences that have dissimilar words but similar in meaning, vectorizing the sentence and finding sentences that has a low cosine similarity with the target sentence can retrieve the most semantically similar sentence.
  - To find which semantically distinct questions come up most frequently, I would vectorize the data using sentence embeddings, for example by using `SentencePiece` and use UMAP, a dimension reduction algorithm, to reduce the dimensions. Then I can use a clustering method such as HDBSCAN or K-means to cluster the sentences.  If they want to have more control over the topic clusters I recommend using HDBSCAN to create clusters that are farther away from each other. Then the centroids of the clusters would be the most representative sentence for the cluster that is also distinct from the others.
  - Using sentence embeddings and clustering would handle the issue of sentences having similar meanings but different words since sentence embeddings can capture the overall meaning of the sentence in a vectorized format.
- \-  Imagine your data is call transcripts from an e-commerce business. Their customers have various topics they may contact the business with, but among them would be statements about concerns on certain products, or issues with the website, etc. How would you go about extracting concerns raised?
  - This question is similar to the task of intent classification. I would extract the concerns by vectorizing the call transcripts and using UMAP to perform dimension reduction; then I would use HDBSCAN to cluster the sentences.  The size and quality of the clusters can be adjusted by fine tuning the hyperparameters.  I would then either manually lable the clusters based on its contents or use GENAI to summarize them to facilitate the task.