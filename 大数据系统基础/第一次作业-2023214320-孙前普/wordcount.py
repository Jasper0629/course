from operator import add
from pyspark import SparkContext

dict = {}
def word_count(txt_path):
    with open(txt_path, 'r') as f:
        word = f.read()
        word = word.split()
        for word in word:
            dict[word] = dict.get(word, 0) + 1
    return dict
    
if __name__ == '__main__':
    txt_path = 'wc_dataset.txt'
    count = word_count(txt_path)
    print(count)