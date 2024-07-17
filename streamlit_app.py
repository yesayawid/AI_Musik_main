import streamlit as st
import pandas as pd
from pymongo import MongoClient
from collections import Counter

# MongoDB configuration
client = MongoClient('mongodb://localhost:27017/')
db = client['audio_classification']
collection = db['predictions']

def get_most_common_predictions(limit=10):
    # Mengambil semua prediksi dari MongoDB
    predictions = list(collection.find({}, {'predicted_class': 1, '_id': 0}))
    
    # Menghitung frekuensi setiap kelas
    prediction_counts = Counter([pred['predicted_class'] for pred in predictions])
    
    # Mengurutkan berdasarkan frekuensi (dari yang terbanyak)
    most_common = prediction_counts.most_common(limit)
    
    return most_common

# Streamlit app
st.title('Most Common Audio Predictions')

# Slider untuk memilih jumlah hasil yang ditampilkan
limit = st.slider('Select number of results to display', 1, 20, 10)

# Mengambil data
most_common = get_most_common_predictions(limit)

# Membuat DataFrame
df = pd.DataFrame(most_common, columns=['Predicted Class', 'Count'])

# Menampilkan data dalam bentuk tabel
st.table(df)

# Menampilkan data dalam bentuk bar chart
st.bar_chart(df.set_index('Predicted Class'))

# Menampilkan total prediksi
total_predictions = sum(df['Count'])
st.write(f"Total predictions: {total_predictions}")

# Menampilkan persentase untuk setiap kelas
df['Percentage'] = df['Count'] / total_predictions * 100
st.write("Percentage for each class:")
st.table(df[['Predicted Class', 'Percentage']].set_index('Predicted Class'))

# Pie chart untuk visualisasi persentase
st.write("Prediction Distribution:")
st.pie_chart(df.set_index('Predicted Class')['Count'])