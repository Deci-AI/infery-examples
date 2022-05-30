from typing import List

import infery
import numpy as np


def main():
    model = infery.load(
        model_path="../../models/mobilenet_v2_1.0_224.tflite",
        framework_type="tflite",
        inference_hardware="cpu",
    )
    outputs: List[np.ndarray] = model.predict(
        np.random.random((1, 224, 224, 3)).astype("float32")
    )
    print(model.benchmark(batch_size=1))


if __name__ == "__main__":
    main()
