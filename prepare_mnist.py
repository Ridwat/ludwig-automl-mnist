import pandas as pd
import os
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

os.makedirs("data", exist_ok=True)

print("Downloading MNIST from OpenML...")

mnist = fetch_openml("mnist_784", version=1, as_frame=True)

X = mnist.data
y = mnist.target

X.columns = [f"pixel_{i}" for i in range(784)]

df = X.copy()
df["label"] = y

cols = ["label"] + [col for col in df.columns if col != "label"]
df = df[cols]

train_df, test_df = train_test_split(
    df,
    train_size=10000,
    test_size=2000,
    random_state=42,
    stratify=df["label"]
)

# Remove columns that are constant in the TRAINING set
constant_cols = [
    col for col in train_df.columns
    if col != "label" and train_df[col].std() == 0
]

train_df = train_df.drop(columns=constant_cols)
test_df = test_df.drop(columns=constant_cols)

print("Removed constant columns:", len(constant_cols))

train_df.to_csv("data/mnist_train_prepared.csv", index=False)
test_df.to_csv("data/mnist_test_prepared.csv", index=False)

print("Done.")
print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)
print(train_df.head())