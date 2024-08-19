import streamlit as st
from src.pipelines import InferencePipeline
# Load the pre-trained model


st.set_page_config(
    page_title='News Sentiment'
)

st.subheader("Enter News Headline to Analyze Sentiment")

title = st.text_input('Enter Headline')
bt = st.button(label="Analyze",)

