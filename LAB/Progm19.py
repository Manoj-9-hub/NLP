import nltk
from nltk.wsd import lesk
from nltk.corpus import wordnet
nltk.download('wordnet')
nltk.download('omw-1.4')

sentence="We sat on the bank of the river"
words = sentence.split()
result=lesk(words,"bank")

print("sentence:",sentence)
print("Ambiguous word:bank")
if result:
    print("Selected sense:",result.name())
    print("Definition:",result.definition())
else:
    print("No sense found")