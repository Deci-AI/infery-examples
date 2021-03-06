from typing import List

import infery
import numpy as np


def main():
    model = infery.load(
        model_path="../../models/mobilenet_v2.zip",
        framework_type="tf2",
        inference_hardware="gpu",
    )
    outputs: List[np.ndarray] = model.predict(
        np.random.random((1, 224, 224, 3)).astype("float32")
    )
    print(model.benchmark(batch_size=1))


if __name__ == "__main__":
    main()
