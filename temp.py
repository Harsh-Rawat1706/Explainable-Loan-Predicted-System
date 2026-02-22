import pickle

with open("model/feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)
    
print("FEATURE NAMES:", feature_names)