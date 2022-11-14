import pyttsx3
import PyPDF2
sach = open("International Trade.pdf","rb")
pdfReader = PyPDF2.PdfReader(sach)
pages = pdfReader.numPages
print(pages)

bot = pyttsx3.init()
voices = bot.getProperty('voices')
bot.setProperty('voice',voices[1].id)
page = pdfReader.getPage(8)
text = page.extractText()
bot.say(text)
bot.runAndWait()


