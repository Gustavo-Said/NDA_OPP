#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import tempfile
import pandas as pd
from backend.processor import process_nda  # You’ll wrap your logic into this

st.set_page_config(page_title="NDA Clause Rewriter")

st.title("📄 NDA Clause Rewriter")

uploaded_file = st.file_uploader("Upload your NDA (.docx)", type=["docx"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    st.info("Processing... This might take a minute.")
    
    df_result = classify_and_rewrite_clauses(tmp_path)  # Your function returns a DataFrame

    st.success("Done! Download your results below:")
    st.download_button(
        label="📥 Download Excel Output",
        data=df_result.to_excel(index=False, engine='openpyxl'),
        file_name="nda_clauses_rewritten.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

