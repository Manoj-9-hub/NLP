import nltk
from nltk.corpus import wordnet
nltk.download('wordnet')
nltk.download('omw-1.4')
word="car"
synsets=wordnet.synsets(word)

print("word:",word)
print("no.of synsets:",len(synsets))

for synset in synsets:
    print("\nSynset:",synset.name())
    print("Definition:",synset.definition())
    print("Examples:",synset.examples())

    synonyms=synset.lemmas()
    print("Synonyms:",[lemma.name() for lemma in synonyms])