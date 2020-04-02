# EmpatheticIntents

### Introduction

In empathetic human social conversations, the speaker often carries a certain emotion, however, the listener being empathetic does not necessarily carry a specific emotion. Instead, by means of a question or an expression of acknowledgement or agreement, a listener can show his empathy towards the other person. By manually analyzing a subset of listener utterances in the EmpatheicDialogues datasets (Rashkin et al., 2019) containing 25K empathetic human coversations, we discovered specific means or intents that a listener uses to express his empathy towards the speaker. The following are the most frequent intents that were discovered:

1. **Questioning** (to know further details orclarify) e.g. *What are you looking forward to?*

2. **Acknowledging**  (Admitting  as  beingfact) e.g. *That  sounds like double  good  news.   It was  probably fun having  your  hard  work rewarded*

3. **Consoling** e.g. *I hope he gets the help he needs.*

4. **Agreeing** (Thinking/Saying the same) e.g. *That’s a great feeling, I agree!*

5. **Encouraging** e.g. *Hopefully you  will  catch  those  great deals!*

6. **Sympathizing** (Express feeling pity orsorrow for the person in trouble) e.g. *So sorry to hear that.*

7. **Suggesting** e.g. *Maybe you two should go to the pet store to try and find a new dog for him!*

8. **Wishing** e.g. *Hey... congratulations to you on both fronts!*

We have extended the number of examples per each intent by searching through the rest of the dataset using words and phrases that are most indicative of the intent. For example, words and phrases such as *100%*, *exactly*, *absolutely*, *definitely*, *i agree*, *me neither*, *me too* and *i completely understand* are indicative of the category *Agreeing*.

Using those categories as well as situation descriptions in EmpatheticDialogues tagged with 32 emotion categories, we trained and tested a BERT transformer based classifier to automatically annotate all the 25K conversations in the EmpatheticDialogues dataset. 

This repository contains the code used to train and evaluate the classifier, the datasets used and the annotated results.  

### BERT transformer based classifier for EmpatheticDialogues.

Given a dialogue utterance, the classifier annotates it with one out of a list of labels containing 33 emotions including neutral and 8 response intents. The classifier is trained and tested on the EmpatheticDialogues dataset introduced by Rashkin et al (2019).  

The pretrained weights necessary to initiate the model prior to training can be downloaded from: [drive.google.com/drive/folders/1KvTt1aK2a2JFKR_YcGaQnc_eNKW-qAos?usp=sharing](https://drive.google.com/drive/folders/1KvTt1aK2a2JFKR_YcGaQnc_eNKW-qAos?usp=sharing)

All the model checkpoints can be downloaded from: 
[drive.google.com/drive/folders/12S9BDbFZYy8TZUV-0l4XCZTtC-KP_a2_?usp=sharing](https://drive.google.com/drive/folders/12S9BDbFZYy8TZUV-0l4XCZTtC-KP_a2_?usp=sharing)

The checkpoint 5 was giving the highest accuracy on the test set.

### Datasets

The original EmpatheticDialogues dataset can be downloaded from: [github.com/facebookresearch/EmpatheticDialogues](https://github.com/facebookresearch/EmpatheticDialogues).

The repository contains the following datasets:

1. ***manual_annotations***: Listener utterances manually annotated with intent labels
2. ***lexically_extended_intent_data***: Listener utterances corresponding to the most frequent intents found by searching through the dataset using words and phrases that are most indicative of each intent. 
3. ***train_data***: The train, validation and test sets used to train and evaluate the BERT transformer based classifier. They consists of situation descriptions tagged with emotion labels and randomly selected subset of listener utterances tagged with intent labels (this subset is randomly chosen from ***lexically_extended_intent_data*** so that the train, validation and test sets are balanced more or less equally across all the labels).
4. ***empatheticdialogues_unannotated***: EmpatheticDialogues dataset processed into different csv files based on the emotion the dialogues are conditioned on
5. ***empatheticdialogues_annotated***: EmpatheticDialogues dataset automatically annotated with output from the BERT transformer based classifier

### Dependencies

Following are the versions of python packages the code has been tested on.

- numpy (1.18.2)
- tensorflow (2.2.0rc2)
- tensorflow-gpu (2.1.0)
- pytorch-transformers (1.2.0)
- h5py (2.10.0)
- pyyaml (3.13)

### Bibliography

Hannah Rashkin, Eric Michael Smith, Margaret Li and Y-Lan Boureau. 2019.  Towards Empathetic Open-domain Conversation  Models:  A  New  Benchmark  and  Dataset.   In *Proceedings  of  the  57th  Annual  Meeting  of  the Association for Computational Linguistics*, pages 5370–5381, Florence, Italy.

### Acknowledgements

This is project carried out by the [Human Computer Interaction Group](https://hci.epfl.ch/) at the School of Computer and Communication Sciences, EPFL, Switzerland. Special thanks goes to (Dr. Pearl Pu)[https://hci.epfl.ch/members/pearl/] for her supervision on this project and PhD student, (Yubo Xie)[https://hci.epfl.ch/members/yubo/] for initial compilation of the code. 
