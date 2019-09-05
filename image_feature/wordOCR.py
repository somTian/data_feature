import pytesseract
from PIL import Image
import os

pytesseract.pytesseract.tesseract_cmd='C:\Program Files\Tesseract-OCR\\tesseract.exe'



data_files = 'data/allimages/'

for root, dirs, files in os.walk(data_files):

    for file in files:
        file_path = os.path.join(root, file)
        pid = file[:-4]
        img = Image.open(file_path)
        code = pytesseract.image_to_string(img, lang='chi_sim')
        print(pid)
        print(code)



