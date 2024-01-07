from model import model
from pipeline import download_data
from predict import predict_and_save
from training import training
from save_model import save, load

train_url = 'https://raw.githubusercontent.com/semantic-textual-relatedness/Semantic_Relatedness_SemEval2024/main/Track%20A/tel/tel_train.csv'
dev_url = 'https://raw.githubusercontent.com/semantic-textual-relatedness/Semantic_Relatedness_SemEval2024/main/Track%20A/tel/tel_dev.csv'
name='tel'

train_df,val_df, dev_df = download_data(train_url,dev_url)
model = training(model,train_df, val_df,name)
save(model,name)
predict_and_save(model, dev_df,name)