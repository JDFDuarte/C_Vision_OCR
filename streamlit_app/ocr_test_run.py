import cv2
import pytesseract  # installed but has some issues. Check later
import easyocr   # using instead of pytesseract
import numpy as np
import re
import ast
import operator

import warnings
warnings.filterwarnings('ignore')

pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract"
print(pytesseract.get_tesseract_version())


#____ Helper Functions ___#

def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)   # read the image in grayscale mode
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)   # creates binary image. "THRESH_BINARY_INV": Inverts the image so text is white on a black background
    return binary


def recognize_characters(image):
    reader = easyocr.Reader(['en'])
    result = reader.readtext(image)
    return ' '.join([text for _, text, _ in result])

# def recognize_characters(image):
#     text = pytesseract.image_to_string(image, config='--psm 6')
#     return text.strip()

##___ Parsing the recognized text into a mathematical expression ___##
def parse_expression(text):
    expression = re.sub(r'[^0-9+\-*/().\s]', '', text)  # this removes non-mathematical characters
    return expression


##___ 
def safe_eval(expression):
    operators = {ast.Add: operator.add, ast.Sub: operator.sub, 
                 ast.Mult: operator.mul, ast.Div: operator.truediv}
    node = ast.parse(expression, mode='eval').body
    def _eval(node):
        if isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.BinOp):
            return operators[type(node.op)](_eval(node.left), _eval(node.right))
        else:
            raise TypeError(node)
    return _eval(node)



def process_math_image(image_path):
    preprocessed_image = preprocess_image(image_path)  # preprocess the image
    
    text = recognize_characters(preprocessed_image)   # recognize characters
    
    expression = parse_expression(text)   # parse the expression
    
    # evaluating the expression
    try:
        result = safe_eval(expression)
        return f"Expression: {expression}\nResult: {result}"
    except:
        return "Unable to evaluate the expression"
    

    
# result = process_math_image('data/img_one.png')
# print(result)


# result = process_math_image('data/img_two.png')
# print(result)


# result = process_math_image('data/img_three.png')
# print(result)