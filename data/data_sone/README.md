### data
train.out: traing data
dev.out: development data
test.out: test data
data format: sentence \t entity type1 start index \t entity type1 end index \t entity type2 start index \t entity type2 end index \t
entity type3 start index \t entity type3 end index \t entity type4 start index \t entity type4 end index \t POS taging information
NOTE: each data file has a corresponding gold entities file.
train_entities: gold entities of training data. each line is the entities of one type, in this data, each sentence correspond to four lines: norm_entity_type, non_norm_entity_type, protein_entity_type and unclear_entity_type.
dev_entities: gold entities of dev data. each line is the entities of one type, in this data, each sentence correspond to four lines: norm_entity_type, non_norm_entity_type, protein_entity_type and unclear_entity_type.
test_entities: gold entities of test data. each line is the entities of one type, in this data, each sentence correspond to four lines: norm_entity_type, non_norm_entity_type, protein_entity_type and unclear_entity_type.