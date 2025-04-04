from kafka import KafkaProducer
import json
import time
import pandas as pd


url = "./client_clicks.csv"
df = pd.read_csv(url)
producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

for _, row in df.iterrows():
    data = row.to_dict()
    producer.send('streaming_data', value=data)
    print(f"Enviado: {data}")
    time.sleep(3) 
