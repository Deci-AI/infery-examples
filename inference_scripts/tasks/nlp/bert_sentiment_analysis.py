import infery
import numpy as np

try:
    from transformers import AutoTokenizer
except (ImportError, ModuleNotFoundError):
    print(
        'The "transformers" python package is required in order to run the NLP example.'
    )


def softmax(x: np.ndarray):
    return np.exp(x) / np.sum(np.exp(x))


def main():
    tokenizer = AutoTokenizer.from_pretrained(
        "distilbert-base-uncased-finetuned-sst-2-english"
    )
    model = infery.load(
        model_path="../../../models/bert_sts_model.onnx",
        framework_type="onnx",
        inference_hardware="cpu",
        # inference_hardware="gpu",
    )

    sentences_for_cassification = [
        "I'm very happy!",
        "I had the worst day...",
        "The Deep-Learning ecosystem is hard to understand.",
        "Deci improved my model's architecture.",
        "Deci increased my model's throughput.",
        "Super-Gradients trains state-of-the-art models.",
    ]
    print("-" * 8)
    for sentence in sentences_for_cassification:
        tokenized_inputs = tokenizer(
            sentence, return_tensors="np", max_length=384, padding="max_length"
        )
        output = model.predict(**tokenized_inputs)[0]
        sentiment = softmax(output)

        # After softmax, index 1 will hold the confidence in a positive answer.
        # 0 = Negative Sentiment; 1 = Positive Sentiment;
        sentiment = sentiment[0][1] * 100
        print(f"- {sentiment:.2f}% positive.", "Sentence:", f'"{sentence}"')


if __name__ == "__main__":
    main()
