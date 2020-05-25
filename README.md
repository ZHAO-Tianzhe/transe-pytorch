## TransE-Pytorch

### Overview

An implementation of TransE* in Pytorch.

This implementation has been evaluated on FB15K and WN18 following the MeanRank (raw) metric*. And the evaluation results are reported in the following table.        

|  | FB15K | WN18 |
| --- | --- | --- |
| This Implementation | 250  | **258**  |
| Results Reported in the Paper*  | **243**   | 263 |


* * *


### Data

##### Input Data

`train2id.txt`, `valid2id.txt`, and `test2id.txt`: the first line is the number of training/validation/test triples, and the following lines are in the format of "head_entity_id tail_entity_id relation_id".

`entity2id.txt`: the first line is the number of entities, and the following lines are in the format of "entity_label \t entity_id".

`relation2id.txt`: the first line is the number of relations, and the following line are in the format of "relation_label \t relation_id".

##### Output Data

`entity_embeddings.pickle`: the embedding vectors of entities.

`relation_embeddings.pickle`: the embedding vectors of relations.


* * *


### Parameters

Please configure the training parameters in ``train.py``.


* * *


*\* Bordes A, Usunier N, Garcia-Duran A, et al. Translating Embeddings for Modeling 
Multi-relational Data. Advances in Neural Information Processing Systems. 2013: 
2787-2795.*

