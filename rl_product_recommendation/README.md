# Instructions
In order to run this project you must:
* Have installed the required libraries.
* Install gemma3 through Ollama: https://ollama.com/library/gemma3
* Have json credentials of your GCP account
* Execute create_tables.py in order to create dataset in GCP, through BigQuery
* Run receiver.py which will listen the simulated online clicks
* Run sender.py which will send the simulated online clicks
* Run engine.py which have the online recommendation system powered with Reinforcement Learning,
also it allows interaction with gemma3 and Dashboards.