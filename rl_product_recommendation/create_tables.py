from google.cloud import bigquery
from google.auth import exceptions
from google.oauth2 import service_account
from google.api_core.exceptions import Conflict
import config

PROJECT_ID = config.PROJECT_ID
DATASET_ID = config.DATASET_ID
TABLE_ID_INTERACTION = config.TABLE_ID_INTERACTION
TABLE_ID_REWARDS = config.TABLE_ID_REWARDS
TABLE_ID_EMBEDDINGS = config.TABLE_ID_EMBEDDINGS
TABLE_ID_METADATA = config.TABLE_ID_METADATA
CREDENTIALS_PATH = config.CREDENTIALS_PATH

credentials = service_account.Credentials.from_service_account_file(CREDENTIALS_PATH)
client = bigquery.Client(credentials=credentials, project=PROJECT_ID)
dataset_id = f"{PROJECT_ID}.{DATASET_ID}"

def create_dataset():
    try:
        client.get_dataset(dataset_id) 
        print(f"El dataset {dataset_id} ya existe.")
    except:
        dataset = bigquery.Dataset(dataset_id)
        dataset.location = "US" 
        client.create_dataset(dataset)
        print(f"Dataset {dataset_id} creado.")

def create_tables():
    table_id_1 = f"{dataset_id}.{TABLE_ID_INTERACTION}"
    schema_1 = [
        bigquery.SchemaField("id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("date", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("item", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("category", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("price", "INTEGER", mode="REQUIRED"),
        bigquery.SchemaField("brand", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("event", "STRING", mode="REQUIRED")
    ]
    
    table_1 = bigquery.Table(table_id_1, schema=schema_1)
    try:
        client.create_table(table_1) 
        print(f"Tabla {table_id_1} creada.")
    except Conflict:
        print(f"La tabla {table_id_1} ya existe.")

    table_id_2 = f"{dataset_id}.{TABLE_ID_REWARDS}"
    schema_2 = [
        bigquery.SchemaField("item", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("reward", "INTEGER", mode="NULLABLE")
    ]
    
    table_2 = bigquery.Table(table_id_2, schema=schema_2)
    try:
        client.create_table(table_2)
        print(f"Tabla {table_id_2} creada.")
    except Conflict:
        print(f"La tabla {table_id_2} ya existe.")

    table_id_3 = f"{dataset_id}.{TABLE_ID_EMBEDDINGS}"
    schema_3 = [
        bigquery.SchemaField("item", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("embedding", "FLOAT64", mode="REPEATED") 
    ]
    
    table_3 = bigquery.Table(table_id_3, schema=schema_3)
    try:
        client.create_table(table_3)
        print(f"Tabla {table_id_3} creada.")
    except Conflict:
        print(f"La tabla {table_id_3} ya existe.")

    table_id_4 = f"{dataset_id}.{TABLE_ID_METADATA}"
    schema_4 = [
        bigquery.SchemaField("item", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("category", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("price", "INTEGER", mode="REQUIRED"),
        bigquery.SchemaField("brand", "STRING", mode="REQUIRED")
    ]
    
    table_4 = bigquery.Table(table_id_4, schema=schema_4)
    try:
        client.create_table(table_4)
        print(f"Tabla {table_id_4} creada.")
    except Conflict:
        print(f"La tabla {table_id_4} ya existe.")

create_dataset()
create_tables()
