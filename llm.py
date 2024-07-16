import ollama
import json
import re

llm_models = ['phi3:3.8b-mini-instruct-4k-q4_K_M']

def extract_from_ocr_text(ocr_text):
    ocr_text = re.sub(r"[\n\t\s]*", "", ocr_text)
    llm_output = ollama.generate(model=llm_models[0], stream=False, format="json", prompt=f'{ocr_text}. extract full_name, date_of_birth and document_id (only numbers). Respond in JSON mode and use the snakecase for keys')
    final_output  = json.loads(llm_output['response'].strip())
    return final_output