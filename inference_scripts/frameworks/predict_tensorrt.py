from typing import List

import infery
import numpy as np


def main():
    model = infery.load(
        model_path="../../models/hardware_specific_models/nvidia/rtx3070/resnet18_batchsize_64_RTX3070.pkl",
        # Nvidia RTX 3070 (TensorRT 8.0.1.6)
        # model_path="../../models/hardware_specific_models/nvidia/t4/mobilenet-ssd_t4_optimized.pkl",                       # Nvidia Tesla T4 (TensorRT 8.0.1.6)
        # model_path="../../models/hardware_specific_models/nvidia/v100/mobilenet-ssd_v100_optimized.pkl",                   # Nvidia Volta V100 (TensorRT 8.0.1.6)
        # model_path="../../models/hardware_specific_models/nvidia/jetson/nano/mobilenet-ssd_jetson_nano_optimized.pkl",     # Nvidia Jetson Nano (JetPack 4.6, TensorRT 8.0.1.6)
        # model_path="../../models/hardware_specific_models/nvidia/jetson/xavier/mobilenet-ssd_jetson_xavier_optimized.pkl", # Nvidia Jetson Xavier NX (JetPack 4.6, TensorRT 8.0.1.6)
        # model_path="../../models/hardware_specific_models/nvidia/jetson/orin/mobilenet-ssd_jetson_orin_optimized.pkl",     # Nvidia Jetson Orin AGX (JetPack 5.0, TensorRT 8.4.0.8)
        framework_type="trt",
        logging_verbosity="DEBUG",  # Show TensorRT verbose logs
    )

    # Creating a random input dynamically
    batch_size = 1
    inference_input_dimensions = model.input_dims[0]
    inference_dtype = np.float32
    inference_shape = (batch_size,) + inference_input_dimensions
    inference_inputs = np.random.random(inference_shape).astype(inference_dtype)

    # Running inference
    outputs: List[np.ndarray] = model.predict(inference_inputs)
    print(outputs)

    # Benchmarking. include_io refers to cuda's H_TO_D and D_TO_H (data transfers should be included in the benchmark?).
    print("With IO:", model.benchmark(batch_size=1))
    print("Without IO:", model.benchmark(batch_size=1, include_io=False))

    # Quick benchmark - 100 forward passes, exclude IO and data copies.
    print(model.benchmark(batch_size=1, include_io=False, repetitions=100))


if __name__ == "__main__":
    main()
