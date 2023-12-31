from sentence_transformers import  util
import numpy as np
import os
import zipfile

def predict_and_save(model, dev_df,name):
    all_scores = []  # Empty list to store all predicted scores

    for i, row in dev_df.iterrows():
        text1 = row["text1"]
        text2 = row["text2"]

        # Generate individual embeddings
        embedding1 = model.encode([text1])
        embedding2 = model.encode([text2])

        # Calculate and append score
        score = util.cos_sim(embedding1, embedding2)[0][0]  # Access specific element
        all_scores.append(np.round(float(score.item()), 2))

    dev_df["Pred_Score"] = np.round(all_scores, 2)

    dev_df = dev_df[['PairID','Pred_Score']]

    dev_df.to_csv(f'predictions/pred_{name}_a.csv', index=False)

    csv_file_path = f'predictions/pred_{name}_a.csv'
    zip_file_path = f'zipped_predictions/pred_{name}_a.zip'

    if os.path.exists(csv_file_path):
        with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zip:
            zip.write(csv_file_path, arcname=os.path.basename(csv_file_path))


