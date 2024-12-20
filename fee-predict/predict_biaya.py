import pickle
import os
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

# Directory where models are saved
save_dir = "models_directory"

# Initialize an empty dictionary to store loaded models
models = {}
targets = []
# Load each model from the directory and add it back to the dictionary
for filename in os.listdir(save_dir):
    if filename.endswith("_model.pkl"):
        target = filename.replace("_model.pkl", "")  # Get the target name from the filename
        targets.append(target)
        file_path = os.path.join(save_dir, filename)
        
        with open(file_path, "rb") as file:
            models[target] = pickle.load(file)

df = pd.read_csv('data_ekspor.csv')
df['hs_code'] = df['hs_code'].astype(str)

ordinal_encoder = OrdinalEncoder()
categorical_features = ['hs_code', 'kategori_barang', 'deskripsi_produk', 
                        'tujuan_negara', 'kota_asal', 'provinsi_asal']
fitur = ['hs_code', 'kategori_barang', 'deskripsi_produk', 'tujuan_negara', 
         'kota_asal', 'provinsi_asal', 'kuantitas', 'volume_m3', 'berat_kg', 'hpp']
X = df[fitur]
X[categorical_features] = ordinal_encoder.fit(X[categorical_features])


# Fungsi untuk meminta input dari pengguna
def get_user_input():
    user_input = {
        'hs_code': '9603.21',
        'kategori_barang': 'Sikat gigi',
        'deskripsi_produk': 'Sikat gigi manual',
        'tujuan_negara': 'Filipina',
        'kota_asal': 'Jakarta',
        'provinsi_asal': 'DKI Jakarta',
        'kuantitas': 777,
        'volume_m3': 2.5,
        'berat_kg': 40,
        'hpp': 1000000
    }
    return user_input

# Mengambil input pengguna
user_data = get_user_input()

# Konversi input pengguna ke DataFrame
user_df = pd.DataFrame([user_data])

user_df[categorical_features] = ordinal_encoder.transform(user_df[categorical_features])

# Melakukan prediksi untuk setiap target
prediksi = {}
for target, model in models.items():
    prediksi[target] = model.predict(user_df)[0]

# Menampilkan hasil prediksi
print("\nHasil Prediksi Biaya:")
for target, nilai in prediksi.items():
    number = round(nilai)
    formatted_number = f"{number:,}".replace(",", ".")
    print(f"{target}: Rp {formatted_number}")

total_biaya = round(sum(prediksi.values()))
formatted_total_biaya = f"{total_biaya:,}".replace(",", ".")
print(f"Total biaya: Rp {formatted_total_biaya}")