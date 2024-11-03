import pickle
import os
import pandas as pd

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


# Fungsi untuk meminta input dari pengguna
def get_user_input():
    user_input = {
        'hs_code': '8414.51',
        'kategori_barang': 'Kipas angin',
        'deskripsi_produk': 'Kipas angin dengan daya listrik',
        'tujuan_negara': 'Jepang',
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
