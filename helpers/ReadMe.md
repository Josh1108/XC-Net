
# Guide to helper files
### similarity.py

returns visualizations and cosine similarities

### analysis.py

Pre-processing and other steps, this is imported in `make_initial_embeddings/make_embeddings_preprocessed.py`

To form the embeddings, do the following:
- Activate venv from GalaXC
- run `python3 make_initital_embeddings/make_embeddings_preprocessed.py`

This will create a column with preprocessed type and store it in `/home/jsk/skill-prediction/XC-Net/dumps/df.csv`


Note: It's imporant to run the files from helper module because of imports.

## Training Embeddings


