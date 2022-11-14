import googletrans
from googletrans import Translator
# print(googletrans.LANGUAGES)
t = Translator() # tao 1 object, de tao phuong thuc
a = t.translate("Em dep qua ",dest="en",src= "vi") # goi phuong thuc
b = a.text 
print(b)
