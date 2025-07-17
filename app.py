# app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import gzip
import shutil
import requests

st.set_page_config(page_title="Food Recommendation System", layout="centered")

@st.cache_data
def download_and_load_data():
    url = "https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv.gz"
    compressed_file = "products.csv.gz"
    extracted_file = "products.csv"

    if not os.path.exists(extracted_file):
        with requests.get(url, stream=True) as r:
            with open(compressed_file, 'wb') as f:
                shutil.copyfileobj(r.raw, f)
        with gzip.open(compressed_file, 'rb') as f_in:
            with open(extracted_file, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

    usecols = ['product_name', 'nutriscore_grade', 'energy_100g', 'proteins_100g',
               'fat_100g', 'carbohydrates_100g', 'sugars_100g', 'fiber_100g', 'salt_100g']

    chunks = pd.read_csv(extracted_file, sep='\t', usecols=usecols,
                         low_memory=False, chunksize=100000)
    df_list = []
    for chunk in chunks:
        chunk.dropna(inplace=True)
        if len(chunk) > 0:
            df_list.append(chunk.sample(min(len(chunk), 1000)))
        if len(df_list) * 1000 >= 5000:
            break
    df = pd.concat(df_list)
    return df

# Load data
st.info("Loading data...")
df = download_and_load_data()

GOALS = {
    'Weight Loss': df[df['fat_100g'] < 5],
    'Muscle Gain': df[df['proteins_100g'] > 10],
    'Better Skin': df[df['fiber_100g'] > 2],
    'Mood Boost': df[df['sugars_100g'] > 10],
    'Weight Gain': df[df['energy_100g'] > 300]
}

st.title("🥗 AI-Powered Food Recommendation System")
selected_goal = st.selectbox("Select Your Health Goal", list(GOALS.keys()))

if selected_goal:
    st.subheader(f"Top Food Recommendations for: {selected_goal}")
    filtered_df = GOALS[selected_goal]
    recommendations = filtered_df['product_name'].dropna().unique().tolist()[:5]
    for i, item in enumerate(recommendations, 1):
        st.write(f"{i}. {item}")

    st.subheader("Nutrient Breakdown")
    chart_df = filtered_df[['proteins_100g', 'fat_100g', 'carbohydrates_100g']].dropna().head(10)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=chart_df, ax=ax)
    ax.set_title(f'Nutrient Comparison - {selected_goal}')
    st.pyplot(fig)

st.caption("Powered by OpenFoodFacts • Built with Streamlit")
