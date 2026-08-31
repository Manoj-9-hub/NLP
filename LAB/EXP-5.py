from nltk.stem import PorterStemmer

ps = PorterStemmer()

word = input("Enter a word: ")

print("Stemmed word:", ps.stem(word))
