from transformers import MarianMTModel, MarianTokenizer

model_name = "Helsinki-NLP/opus-mt-en-fr"

tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

english_text = input("Enter English text: ")

tokens = tokenizer(english_text, return_tensors="pt")

translated = model.generate(**tokens)

french_text = tokenizer.decode(translated[0], skip_special_tokens=True)

print("\nEnglish:", english_text)
print("French:", french_text)