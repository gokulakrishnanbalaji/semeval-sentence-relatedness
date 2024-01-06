from sentence_transformers import SentenceTransformer

def save(model,name):
    model.save(f"/models/finetuned_{name}_model")

def load(name):
    if os.path.exists(f"/models/finetuned_{name}_model"):
        model = SentenceTransformer(f"/models/finetuned_{name}_model")
        return model
    else:
        print("Model not found")
        return None