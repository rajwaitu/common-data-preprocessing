import nlp.preprocessing as preprocessor
import pandas as pd

def test():
    d = {'a': 'This is the best a book', 'b': 'imo this is awesome', 'c': 'this is a PIIOTT'}
    ser = pd.Series(data=d, index=['a', 'b', 'c'])
    ss = preprocessor.remove_slang_words(ser)
    print(ss)

if __name__ == "__main__":
    test()

