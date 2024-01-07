import gradio as gr
from save_model import load
import numpy as np
from sentence_transformers import util

#create a drop down model and get the model name
lang_mapping={
    'Amharic':'amh',
    'Arabic':'arq',
    'Armenian':'ary',
    'English':'eng',
    'Spanish':'esp',
    'Hausa':'hau',
    'Kinyarwanda':'kin',
    'Marathi':'mar',
    'Telugu':'tel'
}
lang_list = [lang for lang in lang_mapping]
model_name_full = gr.Dropdown(lang_list,label="Select Language")
model_name = lang_mapping[model_name_full]


#get 2 text inputs
text1 = gr.Textbox(lines=5,label="Text 1")
text2 = gr.Textbox(lines=5,label="Text 2")

#predict the score
def predict_score(model_name,text1,text2):
    model = load(model_name)

    embedding1 = model.encode([text1])
    embedding2 = model.encode([text2])
    score = util.cos_sim(embedding1, embedding2)[0][0]  # Access specific element
    return np.round(float(score.item()), 2)

#display the score
output = gr.Textbox(label="Score")

#launch the app
gr.Interface(fn=predict_score, inputs=[model_name,text1,text2], outputs=output, title="Semantic Textual Similarity").launch()

