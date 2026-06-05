from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as rt
import joblib
import numpy as np
import os

iris = load_iris()
X, y = iris.data, iris.target

model = LogisticRegression(max_iter=200)
model.fit(X, y)

os.makedirs("models", exist_ok=True)

joblib.dump(
    model,
    "models/iris_model.pkl"
)

initial_type = [
    ("input", FloatTensorType([None, 4]))
]

onnx_model = convert_sklearn(
    model,
    initial_types=initial_type
)

with open(
    "models/iris_model.onnx",
    "wb"
) as f:
    f.write(
        onnx_model.SerializeToString()
    )

sess = rt.InferenceSession(
    "models/iris_model.onnx"
)

input_name = sess.get_inputs()[0].name

np.random.seed(42)

random_samples = np.random.choice(
    len(X),
    10,
    replace=False
)

print("\n--- Extended Validation: 10 Random Samples ---\n")

match_count = 0
mismatch_count = 0

for idx in random_samples:

    sample = X[idx:idx+1].astype(np.float32)

    sklearn_pred = model.predict(sample)[0]

    onnx_pred = sess.run(
        None,
        {input_name: sample}
    )[0][0]

    if sklearn_pred == int(onnx_pred):

        match_count += 1

        print(f"Match at index {idx}: {sklearn_pred}")

    else:

        mismatch_count += 1

        print(
            f"Mismatch at index {idx}: "
            f"Scikit-Learn={sklearn_pred}, "
            f"ONNX={int(onnx_pred)}"
        )

print("\nTotal mismatches:", mismatch_count)

if mismatch_count == 0:
    print("All predictions match — ONNX conversion validated successfully!")
else:
    print("Validation failed.")