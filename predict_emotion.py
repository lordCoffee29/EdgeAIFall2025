from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import torch
from typing import Tuple, Union


class EmotionPredictor:

    def __init__(self, model_name: str = 'abhilash88/face-emotion-detection', device: str | None = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        self.model = ViTForImageClassification.from_pretrained(model_name).to(self.device)
        self.emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        self._last_result: Tuple[str, float] | None = None
        self._last_image_path: str | None = None

    def _prepare_image(self, image: Union[str, Image.Image]) -> Image.Image:
        if isinstance(image, str):
            return Image.open(image)
        if isinstance(image, Image.Image):
            return image
        raise TypeError("image must be a file path or PIL.Image.Image")

    def _predict(self, image: Union[str, Image.Image]) -> Tuple[str, float]:
        if isinstance(image, str) and image == self._last_image_path and self._last_result is not None:
            return self._last_result

        pil_image = self._prepare_image(image)
        inputs = self.processor(pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            idx = int(torch.argmax(probs, dim=-1).item())
            emotion = self.emotions[idx]
            confidence = float(probs[0][idx].item())

        if isinstance(image, str):
            self._last_image_path = image
            self._last_result = (emotion, confidence)
        else:
            self._last_image_path = None
            self._last_result = None

        return emotion, confidence

    def predict_emotion(self, image: Union[str, Image.Image]) -> str:
        # Return predicted emotion
        return self._predict(image)[0]

    def predict_confidence(self, image: Union[str, Image.Image]) -> float:
        # Return confidence score as a float (0.0 - 1.0)
        return self._predict(image)[1]


if __name__ == '__main__':
    import sys

    if len(sys.argv) <= 1:
        print("Usage: python predict_emotion.py path/to/image.png")
        sys.exit(1)

    img_path = sys.argv[1]
    predictor = EmotionPredictor()
    emo = predictor.predict_emotion(img_path)
    conf = predictor.predict_confidence(img_path)
    print(f"Predicted Emotion: {emo} ({conf:.2f})")