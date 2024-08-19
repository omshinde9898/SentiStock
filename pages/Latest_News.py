import streamlit as st
from src.data.scraper import MC_Scraper
import pandas as pd
from src.pipelines import InferencePipeline


st.set_page_config(
    page_title='News Sentiment'
)

st.subheader("Get Latest News Analysis here")

data = MC_Scraper().get_headlines()
pipeline = InferencePipeline()
results = pipeline.run_pipeline(data=data)

frame = pd.DataFrame([data,results])

st.dataframe(frame.transpose())