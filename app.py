from lang import run_lang
import os
import json

#get the languages.json file
with open('languages.json') as json_file:
    data = json.load(json_file)

for lang in data:
    run_lang(data[lang]['name'], data[lang]['train_url'], data[lang]['dev_url'], data[lang]['model_name'])

