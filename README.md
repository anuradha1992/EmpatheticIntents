# EmpatheticIntents

### BERT transformer based classifier for EmpatheticDialogues.

Given a dialogue utterance, the classifier labels it with one of 32 emotions or 9 intents. The classifier is trained and tested on the EmpatheticDialogues dataset introduced by Rashkin et al (2019).  

In empathetic conversations, the speaker often carries a certain emotion, however, the listener being empathetic does not necessarily need to carry a specific emotion. Instead, it can be a question or an expression of acknowledgement or agreement. By manually analyzing a subset of listener utterances in the EmpatheicDialogues datasets, it was discovered specific means or intents that a listener uses to express his empathy towards the speaker. The following are the most frequent intents that were discovered:

1.  Questioning (to know further details orclarify) e.g. Whatare you looking forward to?

2.    Acknowledging  (Admitting  as  beingfact) e.g. That  sounds likedouble  good  news.   It was  probably fun having  your  hard  workrewarded

3. Consoling e.g. I hopehe gets the help he needs.

4. Agreeing (Thinking/Saying the same) e.g. That’s a great feeling, I agree!

5. Encouraging e.g. Hopefullyyou  will  catch  those  greatdeals!

6.  Sympathizing (Express feeling pity orsorrow for the person in trouble) e.g. So sorry to hearthat.

7. Suggesting e.g. Maybeyou two shouldgo to the pet storeto try and find a new dog for him!

8. Wishing e.g. Hey... congratulations to you on bothfronts!

- Preparing training data: Preparing_Training_Data_EmoBERT_with_Intents.ipynb
- Load weights: Load_weights_Emobert_with_intents.ipynb
- Train: Training_Emobert_with_intents.ipynb
- Apply: Applying_on_EmpatheticDialogues_EmoBERT_with_intents.ipynb 

Pretrained weights can be downloaded from: [drive.google.com/drive/folders/1KvTt1aK2a2JFKR_YcGaQnc_eNKW-qAos?usp=sharing](https://drive.google.com/drive/folders/1KvTt1aK2a2JFKR_YcGaQnc_eNKW-qAos?usp=sharing)

All the model checkpoints can be downloaded from: [drive.google.com/drive/folders/12S9BDbFZYy8TZUV-0l4XCZTtC-KP_a2_?usp=sharing](https://drive.google.com/drive/folders/12S9BDbFZYy8TZUV-0l4XCZTtC-KP_a2_?usp=sharing)

The checkpoint 5 was giving the highest accuracy on the test set. 
