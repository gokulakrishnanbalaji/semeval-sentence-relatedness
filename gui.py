import gradio as gr
from save_model import load

#create a drop down model and get the model name
model_name = gr.Dropdown(["tel","tam","mal","kan","hin","ben","eng"],label="Select Language")

#load the model
model = load(model_name)

#get 2 text inputs
text1 = gr.Textbox(lines=5,label="Text 1")
text2 = gr.Textbox(lines=5,label="Text 2")

#predict the score
def predict_score(text1,text2):
    embedding1 = model.encode([text1])
    embedding2 = model.encode([text2])
    score = util.cos_sim(embedding1, embedding2)[0][0]  # Access specific element
    return np.round(float(score.item()), 2)

#display the score
output = gr.Textbox(label="Score")

#launch the app
gr.Interface(fn=predict_score, inputs=[text1,text2], outputs=output, title="Semantic Textual Similarity", description="Semantic Textual Similarity between two sentences").launch()

