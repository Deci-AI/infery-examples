"""
A simple example for how to profile an ONNX model using infery.
For more advanced use cases, please see /inference_notebooks/model_inspection.ipynb
"""

import infery


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
        profiling=True,
    )
    print(model.benchmark(batch_size=1))

    # List all the layers, by percentile of execution time
    print("\nModel Layers:")
    print(model.get_layers_profile_dataframe())

    print("\nTop 20 layers by % of time:")
    print(model.get_bottlenecks(20))


if __name__ == "__main__":
    main()
