import os
import pickle
import pytesseract
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from PIL import Image
import docx
import csv
import faiss
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pdf2image import convert_from_path
import io
import requests
from bs4 import BeautifulSoup

# Fungsi untuk mendapatkan ID folder dari URL
def extract_folder_id(folder_url):
    if "folders" in folder_url:
        folder_id = folder_url.split("/folders/")[1].split("?")[0]
        return folder_id
    raise ValueError("Invalid folder URL. Please provide a valid Google Drive folder URL.")

# Fungsi untuk mendapatkan daftar file dalam folder
def list_files_in_folder(folder_url):
    folder_id = extract_folder_id(folder_url)
    folder_api_url = f"https://drive.google.com/drive/folders/{folder_id}"
    response = requests.get(folder_api_url)
    
    if response.status_code != 200:
        raise ValueError(f"Failed to fetch folder: {response.status_code}")

    soup = BeautifulSoup(response.text, 'html.parser')
    file_elements = soup.find_all('div', {'data-id': True})

    file_urls = []
    file_types = []
    for file_elem in file_elements:
        file_id = file_elem['data-id']
        file_name = file_elem.get_text(strip=True)
        file_url = f"https://drive.google.com/file/d/{file_id}/view"
        file_urls.append(file_url)

        # Prediksi MIME Type berdasarkan ekstensi file
        if file_name.endswith('.pdf'):
            file_types.append('application/pdf')
        elif file_name.endswith('.docx'):
            file_types.append('application/vnd.openxmlformats-officedocument.wordprocessingml.document')
        elif file_name.endswith('.csv'):
            file_types.append('text/csv')
        elif file_name.endswith('.txt'):
            file_types.append('text/plain')
        else:
            file_types.append('unknown')

    return file_urls, file_types

# Fungsi untuk mengekstrak teks dari file TXT
def extract_text_from_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text

# Fungsi untuk mengekstrak teks dari file PDF (OCR jika dipindai)
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
    if not text.strip():  # Coba OCR jika tidak ada teks
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

# Fungsi untuk preprocessing teks
def preprocess_text(text):
    return text.split('\n')

# Fungsi untuk merangkum teks menggunakan HuggingFace
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

# Fungsi untuk membuat chatbot dari banyak file
def create_chatbot_from_files(file_urls, file_types):
    all_sentences = []
    summaries = []

    for file_url, file_type in zip(file_urls, file_types):
        file_id = file_url.split('/')[-2]
        file_path = f'/tmp/{file_id}'

        # Unduh file
        response = requests.get(f"https://drive.google.com/uc?id={file_id}&export=download")
        with open(file_path, 'wb') as f:
            f.write(response.content)

        # Ekstraksi teks berdasarkan jenis file
        if file_type == 'application/pdf':
            file_text = extract_text_from_scanned_pdf(file_path)
        elif file_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
            file_text = extract_text_from_scanned_docx(file_path)
        elif file_type == 'text/csv':
            file_text = extract_text_from_csv(file_path)
        elif file_type == 'text/plain':
            file_text = extract_text_from_txt(file_path)
        else:
            print(f"Unsupported file type: {file_type}. Skipping.")
            continue

        if not file_text.strip():  # Skip jika tidak ada teks
            print(f"Warning: No text extracted from {file_url}. Skipping.")
            continue

        sentences = preprocess_text(file_text)
        if sentences:
            all_sentences.extend(sentences)
            text_to_summarize = file_text[:1000] if len(file_text) > 1000 else file_text
            summary = create_summary_with_hf(text_to_summarize)
            summaries.append(summary)
        else:
            print(f"No valid sentences found in {file_url}. Skipping.")

    if len(all_sentences) > len(summaries):
        summaries.extend(["Ringkasan tidak tersedia."] * (len(all_sentences) - len(summaries)))

    if not all_sentences:
        print("No valid sentences found in any files.")
        return None, None, None, None

    # Buat model embedding dan FAISS index
    sentence_model = SentenceTransformer('all-MPNet-base-v2')
    all_embeddings = sentence_model.encode(all_sentences)
    dimension = all_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(all_embeddings)

    print(f"FAISS index size: {index.ntotal}")
    return index, sentence_model, all_sentences, summaries

# Fungsi chatbot
def chatbot(queries, index, sentence_model, sentences, summaries, top_k=3):
    query_embeddings = sentence_model.encode(queries)
    D, I = index.search(query_embeddings, k=top_k)
    responses = []

    for query, indices in zip(queries, I):
        relevant_sentences = [sentences[idx] for idx in indices if idx < len(sentences)]
        relevant_summaries = [summaries[idx] for idx in indices if idx < len(summaries)]
        combined_sentences = " ".join(relevant_sentences[:2])
        combined_summaries = " ".join(relevant_summaries[:2]) if relevant_summaries else "Tidak ada ringkasan yang relevan."
        response = f"Pertanyaan: {query}\nJawaban: {combined_sentences}\nRingkasan: {combined_summaries}"
        responses.append(response)

    return responses

# Fungsi utama
if __name__ == '__main__':
    folder_url = input("Masukkan URL folder Google Drive: ")
    file_urls, file_types = list_files_in_folder(folder_url)

    index, sentence_model, sentences, summaries = load_data()

    if index is None:
        index, sentence_model, sentences, summaries = create_chatbot_from_files(file_urls, file_types)
        if index is not None:
            save_data(index, sentence_model, sentences, summaries)
        else:
            print("Gagal membuat chatbot.")
            exit()

    if index is not None:
        while True:
            queries = input("Masukkan pertanyaan Anda (pisahkan dengan ';', atau ketik 'exit' untuk keluar): ").split(';')
            if 'exit' in queries:
                print("Keluar dari chatbot.")
                break
            responses = chatbot(queries, index, sentence_model, sentences, summaries)
            for response in responses:
                print(response)
