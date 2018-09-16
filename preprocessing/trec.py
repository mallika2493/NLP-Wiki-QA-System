import requests

import config

def preprocess_trec():
    get_trec()

def get_trec():
    response = requests.get(config.TREC_LOCATION)
    with open(config.TREC_PATH, 'w') as f:
        f.write(response.text)
