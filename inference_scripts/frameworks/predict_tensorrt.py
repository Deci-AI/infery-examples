from typing import List

import infery
import numpy as np


def main():
    model = infery.load(
        model_path="../../models/resnet18_batchsize_64_RTX3070.pkl",
        framework_type="trt",
    )
    outputs: List[np.ndarray] = model.predict(
        np.random.random((64, 3, 224, 224)).astype("float32")
    )
    print(model.benchmark(batch_size=1))
    print(model.benchmark(batch_size=64))
    print(model.benchmark(batch_size=64, include_io=False))


if __name__ == "__main__":
    main()
