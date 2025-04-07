{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5acb5b0b-d699-4ce7-bf54-c3147ed83e42",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "import tempfile\n",
    "import pandas as pd\n",
    "from backend.processor import process_nda  # You’ll wrap your logic into this\n",
    "\n",
    "st.set_page_config(page_title=\"NDA Clause Rewriter\")\n",
    "\n",
    "st.title(\"📄 NDA Clause Rewriter\")\n",
    "\n",
    "uploaded_file = st.file_uploader(\"Upload your NDA (.docx)\", type=[\"docx\"])\n",
    "\n",
    "if uploaded_file is not None:\n",
    "    with tempfile.NamedTemporaryFile(delete=False, suffix=\".docx\") as tmp_file:\n",
    "        tmp_file.write(uploaded_file.read())\n",
    "        tmp_path = tmp_file.name\n",
    "\n",
    "    st.info(\"Processing... This might take a minute.\")\n",
    "    \n",
    "    df_result = classify_and_rewrite_clauses(tmp_path)  # Your function returns a DataFrame\n",
    "\n",
    "    st.success(\"Done! Download your results below:\")\n",
    "    st.download_button(\n",
    "        label=\"📥 Download Excel Output\",\n",
    "        data=df_result.to_excel(index=False, engine='openpyxl'),\n",
    "        file_name=\"nda_clauses_rewritten.xlsx\",\n",
    "        mime=\"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet\"\n",
    "    )\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python nda_env",
   "language": "python",
   "name": "nda_env"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
