import re

def tokenize_text(text):

    text = text.lower().strip()

    clean_text = re.sub(r'[^\w\s]', '', text)

    words = clean_text.split()

    tokenized_words = []

    for word in words:
        tokenized_words.append(word)

    return tokenized_words