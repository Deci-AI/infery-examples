from typing import List

import infery
import numpy as np


def main():
    model = infery.load(
        model_path="../../models/resnet18_batchsize_64.onnx",
        framework_type="onnx",
        inference_hardware="cpu",
    )
    outputs: List[np.ndarray] = model.predict(
        np.random.random((1, 3, 224, 224)).astype("float32")
    )
    print(model.benchmark(batch_size=1))


if __name__ == "__main__":
    main()
