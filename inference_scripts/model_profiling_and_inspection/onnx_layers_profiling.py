"""
A simple example for how to profile an ONNX model using infery.
For more advanced use cases, please see /inference_notebooks/model_inspection.ipynb
"""

import infery


def main():
    model = infery.load(
        model_path="../../models/resnet18_batchsize_64.onnx",
        framework_type="onnx",
        inference_hardware="cpu",
        profiling=True,
    )

    # List all the layers, by percentile of execution time
    print("\nModel Layers:")
    print(model.get_layers_profile_dataframe())

    print("\nTop 20 layers by % of time:")
    print(model.get_bottlenecks(20))


if __name__ == "__main__":
    main()
