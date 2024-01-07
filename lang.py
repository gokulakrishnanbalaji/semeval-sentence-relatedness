from model import model
from pipeline import download_data
from predict import predict_and_save
from training import training
from save_model import save, load

def run_lang(name, train_url,dev_url):
    train_df,val_df, dev_df = download_data(train_url,dev_url)
    finetuned_model = training(model,train_df, val_df,name)
    save(finetuned_model,name)
    predict_and_save(finetuned_model, dev_df,name)