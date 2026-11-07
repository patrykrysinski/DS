import pandas as pd
import os


## Powtórz tworzenie agenta poprzez klasy 
## Stworzenie agenta przez lang chain
import kagglehub

# Download latest version
path = kagglehub.dataset_download("shubhambathwal/flight-price-prediction")


print("Path to dataset files:", path)


files = os.listdir(path)


dfs= {i.split(".")[0]: pd.read_csv(os.path.join(path,i)) for i in files}

dfs.keys()

dfs['economy']

dfs['business']

dfs['Clean_Dataset']

dfs['Clean_Dataset']['stops'].value_counts()


from azure.storage.blob import BlobServiceClient

def download_blob_to_file(blob_service_client: BlobServiceClient, container_name, blob_name, download_file_path):
    # Pobierz klienta blobu
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    
    # Pobierz blob do lokalnego pliku
    with open(file=download_file_path, mode="wb") as download_file:
        download_stream = blob_client.download_blob()
        download_file.write(download_stream.readall())

# Ustawienia połączenia
connection_string = os.getenv("AZURE_CONNECTION_FP")
container_name = "danetotal"
blob_name = "Clean_Dataset.csv"
download_file_path = os.path.join("/Users/patrykrysinski/DS/AI training", blob_name)

# Utwórz klienta usługi blob
blob_service_client = BlobServiceClient.from_connection_string(connection_string)

# Pobierz plik
download_blob_to_file(blob_service_client, container_name, blob_name, download_file_path)

df_2 = pd.read_csv("Clean_Dataset.csv")

df_2.head()
