### data
train.out: traing data
dev.out: development data
test.out: test data
data format: question \t context \t entity type1 start index \t entity type1 end index \t  POS taging information of question \t POS tagging information of context
NOTE: each data file has a corresponding gold entities file.
train_entities: gold entities of training data. each line is the entities of one type
dev_entities: gold entities of dev data. each line is the entities of one type
test_entities: gold entities of test data. each line is the entities of one type