from kafka import KafkaConsumer
import json
from google.cloud import bigquery
from google.oauth2 import service_account
import config  
from config import PROJECT_ID, DATASET_ID, TABLE_ID_INTERACTION, TABLE_ID_REWARDS, TABLE_ID_EMBEDDINGS, TABLE_ID_METADATA, CREDENTIALS_PATH

credentials = service_account.Credentials.from_service_account_file(CREDENTIALS_PATH)
client = bigquery.Client(credentials=credentials, project=PROJECT_ID)

dataset_id = f"{PROJECT_ID}.{DATASET_ID}"

consumer = KafkaConsumer(
    'streaming_data',
    bootstrap_servers='localhost:9092',
    value_deserializer=lambda v: json.loads(v.decode('utf-8'))
)

print("Waiting messages...")

def insert_data(table_id, rows):
    errors = client.insert_rows_json(table_id, rows)
    if errors == []:
        print(f"Data inserted succesfully {table_id}")
    else:
        print(f"Error inserting {table_id}: {errors}")

for message in consumer:
    data = message.value
    print(f"Received: {data}")
    
    rows_1 = [{
        "id": data["id"],
        "date": data["date"],
        "item": data["item"],
        "category": data["category"],
        "price": data["price"],
        "brand": data["brand"],
        "event": data["event"]
    }]
    insert_data(f"{dataset_id}.{TABLE_ID_INTERACTION}", rows_1)
    
    rows_2 = [{
        "item": data["item"],
        "category": data["category"],
        "price": data["price"],
        "brand": data["brand"]
    }]
    insert_data(f"{dataset_id}.{TABLE_ID_METADATA}", rows_2)
    

    query = f"""
    MERGE `{f"{dataset_id}.{TABLE_ID_REWARDS}"}` AS target
    USING (SELECT '{data["item"]}' AS item) AS source
    ON target.item = source.item
    WHEN MATCHED THEN
    UPDATE SET target.reward = target.reward + 1
    WHEN NOT MATCHED THEN
    INSERT (item, reward) VALUES (source.item, 1)
    """

    query_job = client.query(query)
    query_job.result()

    print("REWARD: Row inserted correctly.")