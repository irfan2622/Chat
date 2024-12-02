import requests
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pdf2image import convert_from_path
import pytesseract
import docx
import csv
from PyPDF2 import PdfReader
from PIL import Image
import io
import os
import pickle


# Fungsi untuk mengambil ID folder dari URL
def extract_folder_id(folder_url):
    if "folders" in folder_url:
        folder_id = folder_url.split("/folders/")[1].split("?")[0]
        return folder_id
    raise ValueError("Invalid folder URL. Please provide a valid Google Drive folder URL.")

# Fungsi untuk mendapatkan daftar file dalam folder publik
def list_files_in_folder_public(folder_id):
    url = f"https://www.googleapis.com/drive/v3/files?q='{folder_id}'+in+parents&fields=files(id,name,mimeType)&key=YOUR_API_KEY"
    response = requests.get(url)
    if response.status_code == 200:
        files = response.json().get("files", [])
        file_urls = [f"https://drive.google.com/uc?id={file['id']}" for file in files]
        file_types = [file['mimeType'] for file in files]
        return file_urls, file_types
    else:
        print(f"Error fetching file list: {response.status_code} - {response.text}")
        return [], []

# Fungsi untuk mengunduh file publik
def download_file(file_url, file_path):
    response = requests.get(file_url, stream=True)
    if response.status_code == 200:
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        return file_path
    else:
        raise ValueError(f"Failed to download file: {response.status_code}")

# Fungsi untuk mengekstrak teks dari file TXT
def extract_text_from_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

# Fungsi untuk mengekstrak teks dari file PDF yang dipindai (OCR)
def extract_text_from_scanned_pdf(file_path):
    images = convert_from_path(file_path)
    text = ""
    for image in images:
        text += pytesseract.image_to_string(image)
    return text

# Fungsi untuk mengekstrak teks dari file DOCX
def extract_text_from_scanned_docx(file_path):
    doc = docx.Document(file_path)
    text = ""
    for para in doc.paragraphs:
        text += para.text
    
    if not text.strip():  # Jika tidak ada teks, coba OCR pada gambar dalam dokumen
        images = []
        for rel in doc.part.rels.values():
            if "image" in rel.target_ref:
                img_data = rel.target_part.blob
                image = Image.open(io.BytesIO(img_data))
                images.append(image)
        
        for image in images:
            text += pytesseract.image_to_string(image)
    
    return text

# Fungsi untuk mengekstrak teks dari file CSV
def extract_text_from_csv(file_path):
    with open(file_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        text = ""
        for row in reader:
            text += " ".join(row) + "\n"
    return text

# Fungsi preprocessing teks
def preprocess_text(text):
    return text.split('\n')

# Fungsi untuk merangkum teks
def create_summary_with_hf(text):
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    outputs = model.generate(inputs, max_length=150, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Fungsi utama untuk memproses file dan membuat chatbot
def create_chatbot_from_public_folder(folder_id):
    file_urls, file_types = list_files_in_folder_public(folder_id)
    if not file_urls:
        print("No files found in the public folder.")
        return None, None, None, None

    all_sentences = []
    summaries = []

    for file_url, file_type in zip(file_urls, file_types):
        print(f"Processing file: {file_url}")

        # Unduh file sementara
        temp_file_path = f"/tmp/temp_file.{file_type.split('/')[-1]}"
        download_file(file_url, temp_file_path)

        # Ekstrak teks berdasarkan jenis file
        if file_type == "application/pdf":
            file_text = extract_text_from_scanned_pdf(temp_file_path)
        elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            file_text = extract_text_from_scanned_docx(temp_file_path)
        elif file_type == "text/csv":
            file_text = extract_text_from_csv(temp_file_path)
        elif file_type == "text/plain":
            file_text = extract_text_from_txt(temp_file_path)
        else:
            print(f"Unsupported file type: {file_type}")
            continue

        os.remove(temp_file_path)  # Hapus file sementara

        if file_text:
            sentences = preprocess_text(file_text)
            all_sentences.extend(sentences)
            text_to_summarize = file_text[:1000] if len(file_text) > 1000 else file_text
            summaries.append(create_summary_with_hf(text_to_summarize))
        else:
            summaries.append("Ringkasan tidak tersedia.")

    # Buat model embedding dan FAISS index
    sentence_model = SentenceTransformer('all-MPNet-base-v2')
    all_embeddings = sentence_model.encode(all_sentences)
    dimension = all_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(all_embeddings)

    return index, sentence_model, all_sentences, summaries

# Fungsi utama
if __name__ == '__main__':
    folder_url = input("Masukkan URL folder Google Drive: ")
    folder_id = extract_folder_id(folder_url)

    index, sentence_model, sentences, summaries = create_chatbot_from_public_folder(folder_id)
    if index is not None:
        print("Chatbot siap digunakan.")
        while True:
            queries = input("Masukkan pertanyaan Anda: ").split(';')
            if 'exit' in queries:
                break
            responses = chatbot(queries, index, sentence_model, sentences, summaries)
            for response in responses:
                print(response)
