# NER_Knowledge
## MRC for NER
use machine reading comprehension (MRC) model to solve NER task.
each data is a tuple (question,passage,start_pisition,end_position)
In NER, question is the lable definition for each entity type, passage is each sentence, start_position is the start position of each entity
and end_position is the end position of each entity.
## Single One-pass Model (SOne) for NER
use single one-pass model to solve NER task.
Each data ia a tuple (passage, start_position1, end_position1, start_position2, end_position2, ...)

## Notice
Because we use the last checkpoint of BERT to predict, so the development set is just to verify the performance of model.

### data_mrc
we just set an example for mrc data
### data_sone
we just set an example for SOne data