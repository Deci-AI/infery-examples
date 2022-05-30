import infery

if __name__ == "__main__":
    model = infery.load(
        model_path="../../models/resnet18_batchsize_64_RTX3070.pkl",
        framework_type="trt",
        profiling=True,
    )
    print(model.benchmark(batch_size=1))

    # List all the layers, by percentile of execution time
    print("\nModel Layers:")
    print(model.get_layers_profile_dataframe())

    print("\nTop 20 layers by % of time:")
    print(model.get_bottlenecks(20))
