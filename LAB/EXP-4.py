word = input("Enter a noun: ")

if word.endswith("y"):
    print("Plural:", word[:-1] + "ies")
elif word.endswith(("s", "x", "z", "ch", "sh")):
    print("Plural:", word + "es")
else:
    print("Plural:", word + "s")
