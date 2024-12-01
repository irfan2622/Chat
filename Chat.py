import os
import time
import pickle
import pytesseract
from googleapiclient.discovery import build
from google.oauth2 import service_account
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from PIL import Image
import docx
import csv
import faiss
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pdf2image import convert_from_path
import io
import json

# JSON kredensial Google disematkan dalam kode
GOOGLE_CREDENTIALS_JSON = 'google_credentials.json'

# Link folder Google Drive yang akan digunakan (tanam langsung dalam kode)
DRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1PbTbPboHnqs-eCr63gzYrTC1Ub0XwSaw"

# Fungsi untuk membuat layanan Google Drive
def create_drive_service():
    # Membaca kredensial dari file JSON
    with open(GOOGLE_CREDENTIALS_JSON, 'r') as json_file:
        credentials_info = json.load(json_file)
    
    credentials = service_account.Credentials.from_service_account_info(
        credentials_info, scopes=["https://www.googleapis.com/auth/drive"]
    )
    service = build("drive", "v3", credentials=credentials)
    return service


# Fungsi untuk mengambil ID folder dari URL
def extract_folder_id(folder_url):
    if "folders" in folder_url:
        folder_id = folder_url.split("/folders/")[1].split("?")[0]
        return folder_id
    raise ValueError("Invalid folder URL. Please provide a valid Google Drive folder URL.")

# Fungsi untuk mendapatkan daftar file dalam folder
def list_files_in_folder(service, folder_id):
    query = f"'{folder_id}' in parents and trashed = false"
    results = service.files().list(q=query, fields="files(id, name, mimeType)").execute()
    files = results.get("files", [])
    file_urls = [f"https://drive.google.com/file/d/{file['id']}/view" for file in files if "id" in file]
    file_types = [file["mimeType"] for file in files if "mimeType" in file]
    return file_urls, file_types

# Fungsi untuk mengekstrak teks dari file TXT
def extract_text_from_txt(service, file_url):
    file_id = file_url.split('/')[-2]
    request = service.files().get_media(fileId=file_id)
    file_path = '/tmp/temp.txt'
    with open(file_path, 'wb') as f:
        f.write(request.execute())

    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    return text

# Fungsi untuk mengekstrak teks dari file PDF yang dipindai (OCR)
def extract_text_from_scanned_pdf(service, file_url):
    file_id = file_url.split('/')[-2]
    request = service.files().get_media(fileId=file_id)
    file_path = '/tmp/temp.pdf'
    with open(file_path, 'wb') as f:
        f.write(request.execute())
    
    images = convert_from_path(file_path)
    text = ""
    for image in images:
        text += pytesseract.image_to_string(image)
    
    return text

# Fungsi untuk mengekstrak teks dari file DOCX
def extract_text_from_scanned_docx(service, file_url):
    file_id = file_url.split('/')[-2]
    request = service.files().get_media(fileId=file_id)
    file_path = '/tmp/temp.docx'
    with open(file_path, 'wb') as f:
        f.write(request.execute())
    
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
def extract_text_from_csv(service, file_url):
    file_id = file_url.split('/')[-2]
    request = service.files().get_media(fileId=file_id)
    file_path = '/tmp/temp.csv'
    with open(file_path, 'wb') as f:
        f.write(request.execute())

    with open(file_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        text = ""
        for row in reader:
            text += " ".join(row) + "\n"
    return text

# Fungsi untuk preprocessing teks (misal, membagi ke kalimat)
def preprocess_text(text):
    return text.split('\n')

# Fungsi untuk merangkum teks menggunakan IndoBART
def create_summary_with_hf(text):
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")

    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    outputs = model.generate(inputs, max_length=150, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary

# Fungsi untuk menyimpan data
def save_data(index, sentence_model, sentences, summaries, filepath='chatbot_data.pkl'):
    with open(filepath, 'wb') as f:
        pickle.dump((index, sentence_model, sentences, summaries), f)
    print(f"Data saved to {filepath}")

# Fungsi untuk memuat data
def load_data(filepath='chatbot_data.pkl'):
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            index, sentence_model, sentences, summaries = pickle.load(f)
        print(f"Data loaded from {filepath}")
        return index, sentence_model, sentences, summaries
    return None, None, None, None

# Fungsi untuk chatbot
def chatbot(queries, index, sentence_model, sentences, summaries, top_k=3):
    query_embeddings = sentence_model.encode(queries)
    D, I = index.search(query_embeddings, k=top_k)

    responses = []
    for query, indices in zip(queries, I):
        relevant_sentences = []
        relevant_summaries = []

        for idx in indices:
            if 0 <= idx < len(sentences):
                relevant_sentences.append(sentences[idx])
                summary = summaries[idx] if idx < len(summaries) and summaries[idx] != "Ringkasan tidak tersedia." else ""
                if summary:
                    relevant_summaries.append(summary)

        if relevant_sentences:
            combined_sentences = " ".join(relevant_sentences[:2])
            combined_summaries = " ".join(relevant_summaries[:2]) if relevant_summaries else "Tidak ada ringkasan yang relevan."
            response = f"Pertanyaan: {query}\nJawaban: {combined_sentences}\n\nRingkasan: {combined_summaries}"
        else:
            response = f"Pertanyaan: {query}\nJawaban: Tidak ada konten relevan yang ditemukan."

        responses.append(response)

    return responses

# Fungsi utama
if __name__ == '__main__':
    print("Membuat layanan Google Drive...")
    service = create_drive_service()

    print("Mengambil ID folder dari URL yang disematkan...")
    try:
        folder_id = extract_folder_id(DRIVE_FOLDER_URL)
        print(f"Folder ID berhasil diambil: {folder_id}")
    except ValueError as e:
        print(f"Error: {e}")
        exit()

    index, sentence_model, sentences, summaries = load_data()

    if index is None:
        print("Mendapatkan daftar file dari folder Google Drive...")
        file_urls, file_types = list_files_in_folder(service, folder_id)
        if not file_urls:
            print("Tidak ada file ditemukan di folder.")
        else:
            print("Memproses file untuk membuat chatbot...")
            index, sentence_model, sentences, summaries = create_chatbot_from_files(service, file_urls, file_types)
            if index is not None:
                save_data(index, sentence_model, sentences, summaries)
            else:
                print("Gagal membuat chatbot.")
                exit()

    if index is not None:
        print("Chatbot siap digunakan!")
        while True:
            queries = input("Masukkan pertanyaan Anda (pisahkan dengan ';', atau ketik 'exit' untuk keluar): ").split(';')
            if 'exit' in queries:
                print("Keluar dari chatbot.")
                break

            responses = chatbot(queries, index, sentence_model, sentences, summaries)
            for response in responses:
                print(response)
