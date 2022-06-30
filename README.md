# Deci Inference Examples

A collection of demo-apps and inference scripts for various frameworks using infery.

`infery` is Deci's Inference Engine for Python.<br>`infery` aims to provide a robust API for inference and benchmarks,
on any hardware.

### Prerequisites:
- Infery
    - https://docs.deci.ai/docs/installing-infery
- Git LFS
    - We use Git LFS to store the checkpoints.
    - Make sure to install git LFS, e.g `apt-get install git-lfs`.

```shell
# 1. Install dependencies: 
#   - python3 -m pip install infery | infery-gpu | infery-openvino | infery-tensorrt | infery-onnx-gpu | infery-onnx-cpu | infery-tensorflow-gpu | infery-tensorflow-cpu | ...
#   - apt-get install git-lfs

# 2. Clone this repo and download the example models
git clone https://github.com/Deci-AI/infery-examples.git && cd infery-examples/ && git lfs fetch

# 3. Run the ONNX (or any other) example.
cd /inference_scripts/frameworks && python3 predict_onnx.py
```

### Custom Hardware Examples
#### Nvidia
<a href="https://github.com/Deci-AI/infery-examples/blob/master/inference_scripts/frameworks/predict_tensorrt.py"><img src="https://img.shields.io/badge/Nvidia-GPU-green"></a> <br>
<a href="https://github.com/Deci-AI/infery-examples/blob/master/inference_scripts/frameworks/predict_tensorrt.py">
<img src="https://img.shields.io/badge/Jetson-Orin AGX-green">
</a>
<a href="https://github.com/Deci-AI/infery-examples/blob/master/inference_scripts/frameworks/predict_tensorrt.py">
<img src="https://img.shields.io/badge/Jetson-Xavier AGX-green">
</a>
<a href="https://github.com/Deci-AI/infery-examples/blob/master/inference_scripts/frameworks/predict_tensorrt.py">
<img src="https://img.shields.io/badge/Jetson-Nano-green">
</a>
<br>
<a href="https://github.com/Deci-AI/infery-examples/blob/master/inference_scripts/frameworks/predict_tensorrt.py">
<img src="https://img.shields.io/badge/Cloud-T4-green">
</a>
<a href="https://github.com/Deci-AI/infery-examples/blob/master/inference_scripts/frameworks/predict_tensorrt.py">
<img src="https://img.shields.io/badge/Cloud-V100-green">
</a>
    
#### Intel
<a href="https://github.com/Deci-AI/infery-examples/blob/master/inference_scripts/frameworks/predict_openvino.py"><img src="https://img.shields.io/badge/Intel-CPU-green"></a>
#### Apple
<a href="https://github.com/Deci-AI/infery-examples/blob/master/inference_scripts/frameworks/predict_coreml.py">
    <img src="https://img.shields.io/badge/Apple-CoreML-blue">
</a>


### Frameworks Examples (Copy-Paste scripts)

<a href="https://github.com/Deci-AI/infery-examples/blob/master/inference_scripts/frameworks/predict_tensorrt.py">
    <img src="https://img.shields.io/badge/example-TensorRT-blue">
</a>
<a href="https://github.com/Deci-AI/infery-examples/blob/master/inference_scripts/frameworks/predict_onnx.py">
    <img   src="https://img.shields.io/badge/example-ONNX-blue">
</a>
<a href="https://github.com/Deci-AI/infery-examples/blob/master/inference_scripts/frameworks/predict_tensorflow.py">
    <img src="https://img.shields.io/badge/example-TensorFlow-blue">
</a>
<a href="https://github.com/Deci-AI/infery-examples/blob/master/inference_scripts/frameworks/predict_openvino.py">
    <img src="https://img.shields.io/badge/example-OpenVino-blue"> 
</a>
<a href="https://github.com/Deci-AI/infery-examples/blob/master/inference_scripts/frameworks/predict_torch.py">
    <img src="https://img.shields.io/badge/example-PyTorch-blue">
</a>
<a href="https://github.com/Deci-AI/infery-examples/blob/master/inference_scripts/frameworks/predict_tflite.py">
    <img src="https://img.shields.io/badge/example-TFLite-blue">
</a>
<a href="https://github.com/Deci-AI/infery-examples/blob/master/inference_scripts/frameworks/predict_coreml.py">
    <img src="https://img.shields.io/badge/example-CoreML-blue">
</a>

### Profiling

<a href="https://github.com/Deci-AI/infery-examples/blob/master/inference_scripts/model_profiling_and_inspection/tensorrt_layers_profiling.py">
<img src="https://img.shields.io/badge/example-TensorRT Layers Profiling-purple"> 
</a>

<a href="https://github.com/Deci-AI/infery-examples/blob/master/inference_notebooks/model_inspection.ipynb">
<img src="https://img.shields.io/badge/notebook-Model Inspection-purple"> 
</a>


### Applications
<a href="https://github.com/Deci-AI/infery-examples/blob/master/inference_scripts/tasks/nlp/bert_sentiment_analysis.py">
    <img src="https://img.shields.io/badge/Bert-Sentiment Analysis-orange">
</a>

### Contributing
- Feel free to request a feature/example/application.
  - Open an issue on GitHub and describe your desired usage.
- Please format the code before opening a pull-request
    - `./scripts/lint.sh`
