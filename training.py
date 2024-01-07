from sentence_transformers import  InputExample, losses, util
from torch.utils.data import DataLoader
from scipy.stats import spearmanr
import numpy as np
import pandas as pd
import os

def compute_spearman_correlation(predictions, targets):
    return spearmanr(predictions, targets).correlation

def on_epoch_end(epoch, val_df, model,name):
    val_scores = []  # Empty list to store all predicted scores

    for i, row in val_df.iterrows():
        text1 = row["text1"]
        text2 = row["text2"]

        # Generate individual embeddings
        embedding1 = model.encode([text1])
        embedding2 = model.encode([text2])

        # Calculate and append score
        score = util.cos_sim(embedding1, embedding2)[0][0]  # Access specific element
        val_scores.append(np.round(float(score.item()), 2))

    spear_score=spearmanr(val_scores, val_df['Score']).correlation
    print(f"{name}: Spearman Coefficient on {epoch}th epoch: {spear_score}")
    return spear_score

def training(model, train_df, val_df, name):

    train_examples = [InputExample(texts=[row["text1"], row["text2"]], label=row["Score"]) for i, row in train_df.iterrows()]

    train_dataloader = DataLoader(train_examples, batch_size=8, shuffle=True)  

    train_loss = losses.CosineSimilarityLoss(model=model)

    num_epochs = 7
    spearman_values=[]

    spearman_values.append(on_epoch_end(0,val_df,model,name))
    for epoch in range(1,num_epochs+1):
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=1,
            warmup_steps=100,
        )
        spearman_values.append(on_epoch_end(epoch, val_df, model,name))

    if os.path.exists("eval.csv"):
        eval_df=pd.read_csv("eval.csv")
        eval_df[name]=spearman_values
    else:
        eval_df=pd.DataFrame(columns=[name], data=spearman_values)

    eval_df.to_csv("eval.csv", index=False)

    return model
