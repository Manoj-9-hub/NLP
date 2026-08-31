import nltk
from nltk.corpus import wordnet

nltk.download('wordnet')

word = input("Enter a word: ")

synsets = wordnet.synsets(word)

if synsets:
    print("\nSynsets:")
    for syn in synsets:
        print("Name:", syn.name())
        print("Meaning:", syn.definition())
        print()
else:
    print("No synsets found.")
