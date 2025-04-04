import re
import os
import random
import string
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
import streamlit as st
from scipy.stats import beta
from datetime import datetime
import matplotlib.pyplot as plt
from langchain.llms import Ollama
from google.cloud import bigquery
from collections import defaultdict
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from config import PROJECT_ID, DATASET_ID, TABLE_ID_INTERACTION, CREDENTIALS_PATH, TABLE_ID_REWARDS

st.set_page_config(layout="wide")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = CREDENTIALS_PATH
client = bigquery.Client()

query = f"""
    SELECT date, item, event, category, price, brand 
    FROM `{PROJECT_ID}.{DATASET_ID}.{TABLE_ID_INTERACTION}`
"""
data = client.query(query).to_dataframe()

epsilon = 0.2

def decay_epsilon():
    global epsilon
    epsilon = max(0.05, epsilon * 0.99)

click_data = data[data['event'] == 'click'].groupby('item').size().reset_index(name='clicks')
purchase_data = data[data['event'] == 'purchase'].groupby('item').size().reset_index(name='purchases')
metadata = data[['item', 'category', 'price', 'brand']].drop_duplicates()

items = data["item"].unique()
click_history = defaultdict(int, {row["item"]: row["clicks"] for _, row in click_data.iterrows()})
purchase_history = defaultdict(int, {row["item"]: row["purchases"] for _, row in purchase_data.iterrows()})

counts = defaultdict(lambda: 1, {row["item"]: row["clicks"] + 1 for _, row in click_data.iterrows()})
rewards = defaultdict(lambda: 1.0, {row["item"]: row["purchases"] + 1.0 for _, row in purchase_data.iterrows()})

def log_interaction(item_name, action):

    item_data = metadata[metadata["item"] == item_name].iloc[0]
    category = item_data["category"]
    price = int(item_data["price"])
    brand = item_data["brand"]
    random_id = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
    print("-"*10)
    print("LO QUE SE GUARDARA:")
    print(item_name)
    print("."*10)
    print(item_data, category, price, brand, random_id)
    print("-"*10)

    date_item = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("DATE ITEM", date_item)
    query = f"""
        INSERT INTO `{PROJECT_ID}.{DATASET_ID}.{TABLE_ID_INTERACTION}` (id, date, item, event, category, price, brand)
        VALUES (@id, @date, @item, @event, @category, @price, @brand)
    """
    
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("id", "STRING", random_id),
            bigquery.ScalarQueryParameter("date", "STRING", date_item),
            bigquery.ScalarQueryParameter("item", "STRING", item_name),
            bigquery.ScalarQueryParameter("event", "STRING", action),
            bigquery.ScalarQueryParameter("category", "STRING", category),
            bigquery.ScalarQueryParameter("price", "INT64", price),
            bigquery.ScalarQueryParameter("brand", "STRING", brand),
        ]
    )
    
    client.query(query, job_config=job_config).result()
    print("GUARDADO!")


def select_item_ts():
    sampled_means = {}
    for item in items:
        alpha = rewards[item] + 1  
        beta_param = max(1, counts[item] - rewards[item] + 1)
        sampled_means[item] = beta.rvs(alpha, beta_param) 
    selected = max(sampled_means, key=sampled_means.get)
    selected_brand = metadata.loc[metadata["item"] == selected, "brand"].values[0]
    
    return f"{selected}, {selected_brand}"

def select_item(eps=epsilon):
    if random.random() < eps:
        selected = random.choice(items)
    else:
        selected = max(items, key=lambda x: (
            (rewards[x] / counts[x]) +
            (click_history[x] * 0.1) +
            (purchase_history[x] * 0.2) 
        ))
    selected_brand = metadata.loc[metadata["item"] == selected, "brand"].values[0]
    return f"{selected}, {selected_brand}"

def plot_probabilities(probabilities, title):
    fig, ax = plt.subplots(figsize=(10, 6)) 
    
    labels, values = zip(*probabilities.items())  
    y_pos = np.arange(len(labels))  

    bars = ax.barh(y_pos, values, align='center', color='skyblue')  
    ax.set_yticks(y_pos)  
    ax.set_yticklabels(labels)  
    ax.invert_yaxis()  
    
    ax.set_xlabel('Probabilidad')  
    ax.set_title(title)  

    max_value = max(values)
    
    for bar, value in zip(bars, values):
        text_x = min(value - 0.02, max_value * 0.9) 
        color = 'black' if text_x == value - 0.02 else 'black'  
        
        ax.text(text_x, bar.get_y() + bar.get_height()/2, f"{value:.2%}", 
                ha='right', va='center', fontsize=12, color=color)

    plt.xlim(0, max_value + 0.05) 
    plt.tight_layout()  
    return fig


def update_item_reward(item_name):
    print("ANTES ANTES ITEM: ", item_name)
    query = f"""
        UPDATE `{PROJECT_ID}.{DATASET_ID}.{TABLE_ID_REWARDS}`
        SET reward = reward + 1
        WHERE item = @item_name
    """
    
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("item_name", "STRING", item_name)
        ]
    )
    print("ANTES DE ACTUALIZAR REWARD")
    query_job = client.query(query, job_config=job_config)
    query_job.result()  # Espera a que la consulta termine
    print("DESPUES DE ACTUALZIAR")

st.title("Product Recommendation System with Reinforcement Learning 🤖")

if "current_item" not in st.session_state:
    st.session_state.current_item = select_item()


st.subheader("Do you like this item?")
st.info(st.session_state.current_item)
curr_item = st.session_state.current_item
image = Image.open(f'/Users/fernando/Documents/mojix/project2_base_4_cloud/assets/{curr_item.split()[0][:-1]}.png')
st.image(image, width=150)

if "show_purchase_options" not in st.session_state:
    st.session_state.show_purchase_options = False
if st.button("Yes 👍"):
    item_name = st.session_state.current_item.split(",")[0]
    rewards[item_name] += 1
    purchase_history[item_name] += 1
    counts[item_name] += 1
    click_history[item_name] += 1
    #NEW
    st.session_state.show_purchase_options = True
    st.session_state.last_item = item_name
    


if st.session_state.show_purchase_options:
    bleft, bright = st.columns(2)
    st.subheader("¿Quieres comprar este producto?")
    if bleft.button("Buy ✅`🛒", use_container_width=True):
        bleft.markdown("Your purchase has been registered.")
        print("SE ESTA POR COMPRAR")
        log_interaction(st.session_state.last_item, "purchase")
        update_item_reward(st.session_state.last_item)
        st.session_state.show_purchase_options = False

        decay_epsilon()
        st.session_state.current_item = select_item()
        curr_item = st.session_state.current_item
        image = Image.open(f'/Users/fernando/Documents/mojix/project2_base_4_cloud/assets/{curr_item.split()[0][:-1]}.png')
        st.image(image, width=150)
        st.rerun()

    elif bright.button("Discard item ❌🛒", use_container_width=True):
        bright.markdown("Your item has been discarded")
        print("NO COMPRARA")
        log_interaction(st.session_state.last_item, "click")
        st.session_state.show_purchase_options = False
        decay_epsilon()
        st.session_state.current_item = select_item()
        st.rerun()

if st.button("No 👎"):
    st.session_state.current_item = select_item(eps=1)

    st.rerun()
    print("CLICK NO")

def extract_text(query):
    lines = query.split("\n")
    for line in lines:
        line = line.strip()  
        if re.match(r"[A-Z]", line): 
            pattern = r"([A-Z][^`]*)"
            match = re.search(pattern, line + "\n" + "\n".join(lines[lines.index(line) + 1:]), re.MULTILINE)
            
            if match:
                extracted = match.group(1)
                extracted = re.sub(r',', '', extracted)  
                if ";" not in extracted[-2]:
                    extracted = extracted[:-1] + ";"  
                return extracted
    
    return "No se pudo extraer el texto."


#LLM

llm = Ollama(model="gemma3")

def extract_text(query):
    lines = query.split("\n")
    len_lines = len(lines)
    c = 0
    while True:
      aux = lines[c].strip()
      if re.match(r"[A-Z]", aux):
        break
      else:
        c += 1
      if c == len_lines:
        return "ERROR"
        break
      
    new_query = "\n".join(lines[c:-1])
    return new_query

bq_dataset = PROJECT_ID + "." + DATASET_ID
print("!"*20)
print(bq_dataset)
print("!"*20)
print(f"{bq_dataset}.{TABLE_ID_INTERACTION}")

def replace_steam_potential(text):
    return re.sub(r'steam_potential_455109_u6', 'steam-potential-455109-u6', text)


def execute_code_from_langchain(input_text):
    prompt_template = f"""I have three tables in BigQuery:

{bq_dataset}.{TABLE_ID_INTERACTION} – Stores client interactions. Each row represents an event where a client either makes a purchase or clicks on an item.

Table Schema:

item: Unique identifier for an item (pants, coat, jeans, shoe, sweater, t-shirt).

date: Event date.

category: Category of the item.

price: Price of the item.

brand: Brand of the item.

event: Type of action (purchase or click).

{bq_dataset}.{TABLE_ID_REWARDS} – Stores the number of purchases per item.

Table Schema:

item: Unique identifier for an item.

reward: Number of times the item was purchased.

Purchases are recorded in {bq_dataset}.{TABLE_ID_REWARDS}, but clicks are not.

Query Generation Guidelines:
Intent Recognition: Identify the purpose of {input_text} (e.g., aggregation, filtering, ranking).

Efficient Data Retrieval: Select only necessary columns.

Optimized Query Logic: Use WHERE, GROUP BY, or JOINs to eliminate redundant operations.

Minimalist Approach: The query must be concise, without extra clauses or explanations.

Output Constraints:
Return only the SQL query solving {input_text}.

The name of the dataset is {bq_dataset}, dont change the name.

No comments, explanations, or redundant code.

Tables must not be enclosed in quotes.

The word "SQL" must not appear in the output.

Optimize efficiency by avoiding unnecessary operations.

Now, process {input_text} and generate the query accordingly."""

    prompt = PromptTemplate(template=prompt_template)
    chain = LLMChain(llm=llm, prompt=prompt)
    try:
        
        result = chain.run(input_text=input_text)
        print("RESULTADO LLM:", result)
        
        q = result.strip()
        print("QUERY GENERADA:", q)

        q = extract_text(q)
        print("QUERY LIMPIA:", q)
        
        q = replace_steam_potential(q)
        query_job = client.query(q)
        print("QUERY JOB:", query_job)

        results = query_job.result()
        print("RESULTADOS DE LA CONSULTA:", results)

        data = [dict(row) for row in results]
        text_result = "\n".join([", ".join([str(value) for value in row.values()]) for row in data])

        return q, text_result
    
    except Exception as e:
        st.error(f"Error: {e}")
        return None, None

st.subheader("Chat (Interaction with Gemma3)")
txt = st.text_area("Write your requirement (in english for better results):", "Show me the last record of a purchased item")

if st.button("Ejecutar"):
    generated_code, output = execute_code_from_langchain(txt)
    
    if generated_code:
        st.text_area("Answer from LLM:", generated_code, height=100)
    
    if output:
        st.text_area("Answer:", output, height=200)


st.subheader("Dashboard")
ileft, iright = st.columns(2)

iright.subheader("Clicks & Purchases per day")

query = f"""
    SELECT DATE(date) AS day,
           SUM(CASE WHEN event = 'click' THEN 1 ELSE 0 END) AS clicks,
           SUM(CASE WHEN event = 'purchase' THEN 1 ELSE 0 END) AS purchases
    FROM `{PROJECT_ID}.{DATASET_ID}.{TABLE_ID_INTERACTION}`
    GROUP BY day
    ORDER BY day
"""
interaction_data = client.query(query).to_dataframe()

# Crear el gráfico de barras
fig, ax = plt.subplots(figsize=(10, 4.6))
x = np.arange(len(interaction_data['day']))  # Posiciones en el eje X
width = 0.4  # Ancho de las barras

ax.bar(x - width/2, interaction_data['clicks'], width, label='Clicks', color='red')
ax.bar(x + width/2, interaction_data['purchases'], width, label='Purchases', color='green')

ax.set_xlabel("Day")
ax.set_ylabel("Quantity")
ax.set_title("Quantity of Clicks and Purchases per Day")
ax.set_xticks(x)
ax.set_xticklabels(interaction_data['day'], rotation=45)
ax.legend()

iright.pyplot(fig)


ileft.subheader("Probability Distribution")

query = f"""
    SELECT item, 
           SUM(CASE WHEN event = 'click' THEN 1 ELSE 0 END) AS clicks,
           SUM(CASE WHEN event = 'purchase' THEN 1 ELSE 0 END) AS purchases
    FROM `{PROJECT_ID}.{DATASET_ID}.{TABLE_ID_INTERACTION}`
    GROUP BY item
"""
interaction_data = client.query(query).to_dataframe()
item_counts = defaultdict(int, {row["item"]: row["clicks"] + row["purchases"] for _, row in interaction_data.iterrows()})
total_interactions = sum(item_counts.values())
item_probabilities = {item: count / total_interactions for item, count in item_counts.items() if total_interactions > 0}

# Ordenar los ítems por probabilidad de forma ascendente
sorted_items = sorted(item_probabilities.items(), key=lambda x: x[1])

# Convertirlo nuevamente en un diccionario ordenado
sorted_item_probabilities = dict(sorted_items)

#st.write(item_probabilities)
ileft.pyplot(plot_probabilities(sorted_item_probabilities, "Probability of Choice"))

i2left, i2right = st.columns(2)


i2right.subheader("Clicks & Purchases per Item")
# Convertir el DataFrame en formato largo para facilidad de graficado
interaction_data_melted = interaction_data.melt(id_vars=["item"], 
                                                value_vars=["clicks", "purchases"], 
                                                var_name="event", 
                                                value_name="count")

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=interaction_data_melted, x="item", y="count", hue="event", ax=ax, palette=["red", "green"])
ax.set_title("Quantity of Clicks and Purchases per Item")
ax.set_xlabel("Item")
ax.set_ylabel("Quantity")
ax.legend(title="Event")


for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='bottom', fontsize=10, color='black')

i2right.pyplot(fig)

i2left.subheader("Clicks & Purchases")

total_clicks = interaction_data["clicks"].sum()
total_purchases = interaction_data["purchases"].sum()

total_data = pd.DataFrame({
    "event": ["clicks", "purchases"],
    "count": [total_clicks, total_purchases]
})

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=total_data, x="event", y="count", ax=ax, hue="event", palette=["red", "green"], legend=False)

ax.set_title("Total of Clicks and Purchases")
ax.set_xlabel("Event")
ax.set_ylabel("Quantity")

for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='bottom', fontsize=10, color='black')

i2left.pyplot(fig)


