import pandas as pd
import yaml

df = pd.read_csv("data/mnist_train_prepared.csv")

input_features = []

for col in df.columns:
    if col != "label":
        input_features.append({
            "name": col,
            "type": "number"
        })

config = {
    "input_features": input_features,
    "output_features": [
        {
            "name": "label",
            "type": "category"
        }
    ],
    "trainer": {
        "epochs": 3,
        "batch_size": 128,
        "learning_rate": 0.001
    }
}

with open("config.yaml", "w") as file:
    yaml.dump(config, file, sort_keys=False)

print("config.yaml created successfully.")
print("Number of input features:", len(input_features))