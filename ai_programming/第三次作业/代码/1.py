from PIL import Image
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import numpy as np
import nltk
import os
string = " "
with open("target.txt","r") as f:
    text_raw = f.read()
    texts = nltk.word_tokenize(text_raw)
    words=[text.lower() for text in texts if text.isalpha()]
    
    
wc = WordCloud(background_color="white",# 设置背景颜色
               max_words=2000, # 词云显示的最大词数
               height=400, # 图片高度
               width=800, # 图片宽度
               max_font_size=50, #最大字体     
               )
for word in words:
    string = string + word + " "
    
wc.generate(string)
print("generate")
wc.to_file(os.path.join("target.png"))
显示图像
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")# 关掉图像的坐标
plt.show()