## Implementation of XC-Net

XC-Net handles a heterogeneous graph of jobs and skills and runns GNN operations to generate node embeddings. 

These node embeddings are then used for an XMLC task by using a one vs all classifier

`train_main.py`: main training file
`utils.py`: helper functions