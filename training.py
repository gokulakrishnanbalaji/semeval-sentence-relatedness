from sentence_transformers import  InputExample, losses, util
from torch.utils.data import DataLoader
from scipy.stats import spearmanr
import numpy as np

def compute_spearman_correlation(predictions, targets):
    return spearmanr(predictions, targets).correlation

def on_epoch_end(epoch, val_df, model):
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

    print(f"Spearman Coefficient on {epoch+1}th epoch: {spearmanr(val_scores, val_df['Score']).correlation}")

def training(model, train_df, val_df):

    train_examples = [InputExample(texts=[row["text1"], row["text2"]], label=row["Score"]) for i, row in train_df.iterrows()]

    train_dataloader = DataLoader(train_examples, batch_size=8, shuffle=True)  

    train_loss = losses.CosineSimilarityLoss(model=model)

    num_epochs = 7
    on_epoch_end(0)
    for epoch in range(1,num_epochs+1):
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=1,
            warmup_steps=100,
        )
        on_epoch_end(epoch, val_df, model)

    return model
