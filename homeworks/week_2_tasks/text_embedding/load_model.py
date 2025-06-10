from sentence_transformers import SentenceTransformer

def load_transformer_model(model_name):
    model = SentenceTransformer(model_name)
    return model