import re
text = input("Enter a sentence: ")
pattern = input("Enter the word/pattern to search: ")
match = re.match(pattern, text)

if match:
    print("Match found at the beginning:", match.group())
else:
    print("No match at the beginning.")
search = re.search(pattern, text)
if search:
    print("Pattern found:", search.group())
else:
    print("Pattern not found.")
