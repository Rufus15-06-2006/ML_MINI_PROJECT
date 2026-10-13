# Clinical Trial Keyword Recommender

Reimplementation of collaborative filtering for recommending
MeSH keywords in clinical-trial search.

## Setup
pip install -r requirements.txt

## Run the application
streamlit run app.py

## Main files
- app.py: Interactive application
- extract.py: Keyword extraction
- pipeline.py: Model and data pipeline
- recommend.py: Keyword recommendation
- model.npz: Trained model artifact
- terms_mesh.pkl: MeSH vocabulary artifact
- synonyms.json: Synonym mapping
