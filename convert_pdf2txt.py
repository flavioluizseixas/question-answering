import fitz  # PyMuPDF
import json
from datasets import load_dataset

# Caminho para o seu arquivo PDF e para salvar o arquivo JSON
pdf_path = "./data/KUROSE, James - Redes de Computadores e a Internet_ uma abordagem top-down-Pearson (2013).pdf"
output_path = "./data/book_dataset.json"

# Função para extrair texto de um PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

# Extraindo texto do PDF
text = extract_text_from_pdf(pdf_path)

# Exemplo simples de divisão do texto em capítulos
chapters = text.split("Chapter")
dataset = [{"text": "Chapter" + chapter} for chapter in chapters if chapter.strip()]

# Salvando o dataset em um arquivo JSON
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=4)

# Carregando o dataset do arquivo JSON
book_dataset = load_dataset("json", data_files=output_path, split="train")

# Exibindo um exemplo do dataset
print(book_dataset[0])