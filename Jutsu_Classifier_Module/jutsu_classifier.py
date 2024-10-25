import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the trained model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("devSubho51347/roberta_multi_label_classifier", num_labels=3)
# model.load_state_dict(torch.load("/content/roberta_multi_text_classifier_model/model.safetensors"))  # Load your trained model weights
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

class_mapping = {0: 1, 1: 2, 2: 3}

converted_labels = []


# Prediction function

class JutsuPredictor:

    def __init__(self, text):
        self.text = [text]
        # self.jutsu = self.predict(self.text)
        # self.jutsu_output()

    def predict(self,texts):

        model.eval()  # Set model to evaluation mode
        predictions = []

        with torch.no_grad():
            for text in texts:
                # Tokenize the input text
                encoding = tokenizer(
                    text,
                    add_special_tokens=True,
                    max_length=512,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )

                input_ids = encoding['input_ids'].to(device)  # Move to the same device
                attention_mask = encoding['attention_mask'].to(device)  # Move to the same device

                # Get model predictions
                outputs = model(input_ids, attention_mask=attention_mask).logits
                preds = torch.sigmoid(outputs).cpu().numpy()  # Move to CPU for numpy conversion
                print(preds)
                # Binarize predictions
                binary_preds = (preds > 0.5).astype(int)
                predictions.append(binary_preds[0])  # Append the first item for each prediction

        return predictions

    def jutsu_output(self):
        predicted_labels = self.predict(self.text)
        texts_to_predict = self.text

        for text, pred in zip(texts_to_predict, predicted_labels):
            print(f"Text: {text}\nPredicted labels: {pred}\n")

        class_mapping = {0: 'Ninjutsu', 1: 'Taijutsu', 2: 'Genjutsu'}

        for ele in predicted_labels:
            label_indices = np.where(pred == 1)[0]
            original_labels = [class_mapping[idx] for idx in label_indices]
            return original_labels[0]


#
# # # Example usage
# texts_to_predict = [
#     "The user places a hallucinatory darkness on a target's eyesight, causing them to see nothing but black; Tō no Sho likens the sensation to being at the bottom of a deep hole. Because the target cannot see, they are very vulnerable to attack. Although this handicap is dangerous even to the likes of the Third Hokage,[1] it is not insurmountable, as the Third is able to sense attacks to try to defend himself and smell his attackers in order to stage a counterattack. When the Third finally captures the user and begins removing their soul, the darkness disperses, something that Orochimaru, an onlooker, immediately notices.[2]"
# ]
# #
# predicted_labels = predict(texts_to_predict)
#
# # Display predicted outputs
# for text, pred in zip(texts_to_predict, predicted_labels):
#     print(f"Text: {text}\nPredicted labels: {pred}\n")
#
# class_mapping = {0: 'Ninjutsu', 1: 'Taijutsu', 2: 'Genjutsu'}
# for ele in predicted_labels:
#     label_indices = np.where(pred == 1)[0]
#     original_labels = [class_mapping[idx] for idx in label_indices]
#     print(original_labels)

# obj = JutsuPredictor("A shorter version of the 1000 Metre Punch, the user concentrates their chakra into their fist and lands a devastating blow on the opponent, sending them flying to at least 100 metres away from where the user is.")
# print(obj.jutsu_output())
#
# stri = "My name is sunny and I am hunny"
#
# print([stri])