import os
import streamlit as st
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
from googleapiclient.http import MediaFileUpload
import numpy as np

# Fungsi untuk membuat layanan Google Drive
def create_drive_service(credentials_path):
    creds = service_account.Credentials.from_service_account_file(
        credentials_path, scopes=["https://www.googleapis.com/auth/drive"]
    )
    service = build("drive", "v3", credentials=creds)
    return service

# Fungsi untuk mengunggah file ke Google Drive
def upload_file_to_drive(service, folder_id, file_path, file_name):
    file_metadata = {'name': file_name, 'parents': [folder_id]}
    media = MediaFileUpload(file_path, mimetype='application/octet-stream')
    file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    return f"https://drive.google.com/file/d/{file['id']}/view"

# Fungsi untuk menyimpan data
def save_data(index, sentence_model, sentences, summaries, filepath='chatbot_data.pkl'):
    with open(filepath, 'wb') as f:
        pickle.dump((index, sentence_model, sentences, summaries), f)

# Fungsi untuk memuat data
def load_data(filepath='chatbot_data.pkl'):
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    return None, None, None, None

# Fungsi untuk preprocessing teks
def preprocess_text(text):
    # Bersihkan text dan pisahkan menjadi kalimat yang lebih baik
    text = text.replace('\n', ' ').strip()
    sentences = []
    current = []
    
    for word in text.split():
        current.append(word)
        if word.endswith(('.', '?', '!')):
            if len(current) > 3:  # Minimal 3 kata per kalimat
                sentences.append(' '.join(current))
            current = []
    
    if current:  # Tambahkan sisa kalimat jika ada
        if len(current) > 3:
            sentences.append(' '.join(current))
    
    return sentences

# Fungsi untuk merangkum teks
def create_summary_with_hf(text):
    try:
        tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
        model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
        inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
        outputs = model.generate(inputs, max_length=150, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary
    except Exception as e:
        st.error(f"Error creating summary: {str(e)}")
        return text[:200] + "..."  # Return truncated text as fallback

# Fungsi untuk memproses single file dan update data chatbot
def process_single_file(service, file_id, file_name, mime_type):
    try:
        # Handle Google Workspace files
        if mime_type == 'application/vnd.google-apps.document':
            # Export Google Doc as plain text
            content = service.files().export(
                fileId=file_id,
                mimeType='text/plain'
            ).execute()
            text = content.decode('utf-8')
        elif mime_type == 'application/vnd.google-apps.spreadsheet':
            # Export Google Sheet as CSV
            content = service.files().export(
                fileId=file_id,
                mimeType='text/csv'
            ).execute()
            csv_content = csv.reader(io.StringIO(content.decode('utf-8')))
            text = '\n'.join([' '.join(row) for row in csv_content])
        else:
            # Handle regular binary files
            request = service.files().get_media(fileId=file_id)
            file_content = io.BytesIO(request.execute())
            
            # Ekstrak teks berdasarkan tipe file
            if file_name.endswith('.pdf'):
                pdf = PdfReader(file_content)
                text = ''
                for page in pdf.pages:
                    text += page.extract_text() + '\n'
            elif file_name.endswith('.docx'):
                doc = docx.Document(file_content)
                text = '\n'.join([para.text for para in doc.paragraphs])
            elif file_name.endswith('.txt'):
                text = file_content.getvalue().decode('utf-8')
            elif file_name.endswith('.csv'):
                csv_content = csv.reader(io.StringIO(file_content.getvalue().decode('utf-8')))
                text = '\n'.join([' '.join(row) for row in csv_content])
            else:
                st.warning(f"Format file tidak didukung: {file_name}")
                return None, None
        
        # Preprocess teks
        processed_text = preprocess_text(text)
        if not processed_text:
            st.warning(f"Tidak ada teks yang bisa diekstrak dari file: {file_name}")
            return None, None
            
        # Buat ringkasan
        summary = create_summary_with_hf(' '.join(processed_text))
        return processed_text, summary
        
    except Exception as e:
        st.error(f"Error memproses file {file_name}: {str(e)}")
        return None, None

# Fungsi untuk inisialisasi data default
def initialize_default_data():
    # Data default tentang berbagai topik
    default_texts = [
        # Bank Jago
        "Bank Jago adalah bank digital di Indonesia yang menyediakan layanan perbankan melalui aplikasi mobile.",
        "Aplikasi Jago dapat diunduh melalui Google Play Store untuk Android dan App Store untuk iOS.",
        "Bank Jago menawarkan berbagai layanan seperti tabungan, transfer, pembayaran, dan kartu debit.",
        "Untuk membuka rekening di Bank Jago, pengguna cukup mengunduh aplikasi dan melakukan verifikasi data.",
        "Bank Jago memiliki fitur Pocket yang memungkinkan pengguna mengorganisir keuangan dalam berbagai kategori.",
        
        # GitHub
        "GitHub adalah platform hosting kode yang memungkinkan kolaborasi dan kontrol versi menggunakan Git.",
        "Repository di GitHub bisa bersifat publik atau private untuk menyimpan dan mengelola kode.",
        "Pull Request untuk review dan merge kode di GitHub.",
        "Issues untuk tracking bug dan diskusi di GitHub.",
        "GitHub Actions untuk otomatisasi workflow.",
        "GitHub Copilot adalah AI assistant untuk coding.",
        "Fork untuk membuat salinan repository di GitHub.",
        
        # Laptop
        "Laptop adalah komputer portabel yang bisa dibawa dan digunakan di mana saja.",
        "Komponen utama laptop meliputi processor, RAM, storage, dan layar.",
        "Processor laptop umumnya menggunakan Intel Core atau AMD Ryzen.",
        "RAM laptop menentukan kemampuan multitasking, umumnya 8GB hingga 32GB.",
        "Storage: SSD untuk kecepatan, HDD untuk kapasitas.",
        "Battery life 4-12 jam sesuai penggunaan.",
        "Laptop gaming memiliki kartu grafis dedicated untuk performa tinggi.",
        
        # Bumi
        "Bumi adalah planet ketiga dari Matahari dalam tata surya kita.",
        "Bulan adalah satelit alami Bumi.",
        "Atmosfer: 78% nitrogen, 21% oksigen.",
        "Medan magnet Bumi melindungi dari radiasi.",
        "70% permukaan Bumi adalah lautan.",
        "Rotasi 24 jam, revolusi 365.25 hari.",
        "Inti Bumi: padat dan cair.",
        "Lapisan: kerak, mantel, inti.",
        "Lempeng tektonik penyebab gempa dan gunung berapi."
    ]
    
    # Buat ringkasan untuk setiap teks
    default_summaries = [
        # Bank Jago summaries
        "Bank Jago adalah bank digital Indonesia dengan layanan mobile banking.",
        "Aplikasi Jago tersedia di Play Store dan App Store.",
        "Bank Jago menyediakan layanan tabungan, transfer, pembayaran, dan kartu debit.",
        "Pembukaan rekening Bank Jago melalui aplikasi dengan verifikasi data.",
        "Fitur Pocket membantu mengorganisir keuangan dalam kategori.",
        
        # GitHub summaries
        "GitHub adalah platform hosting kode dengan fitur kolaborasi dan version control.",
        "Repository GitHub bisa public atau private untuk manajemen kode.",
        "Pull Request untuk review dan merge kode di GitHub.",
        "Issues untuk tracking bug dan diskusi di GitHub.",
        "GitHub Actions untuk otomatisasi workflow.",
        "GitHub Copilot adalah AI assistant untuk coding.",
        "Fork untuk membuat salinan repository di GitHub.",
        
        # Laptop summaries
        "Laptop adalah komputer portabel untuk mobilitas.",
        "Komponen laptop: processor, RAM, storage, layar.",
        "Processor laptop: Intel Core atau AMD Ryzen.",
        "RAM laptop 8GB-32GB untuk multitasking.",
        "Storage: SSD untuk kecepatan, HDD untuk kapasitas.",
        "Battery life 4-12 jam sesuai penggunaan.",
        "Laptop gaming dengan GPU dedicated.",
        
        # Bumi summaries
        "Bumi adalah planet ketiga dari Matahari.",
        "Bulan adalah satelit alami Bumi.",
        "Atmosfer: 78% nitrogen, 21% oksigen.",
        "Medan magnet Bumi melindungi dari radiasi.",
        "70% permukaan Bumi adalah lautan.",
        "Rotasi 24 jam, revolusi 365.25 hari.",
        "Inti Bumi: padat dan cair.",
        "Lapisan: kerak, mantel, inti.",
        "Lempeng tektonik penyebab gempa dan gunung berapi."
    ]
    
    return default_texts, default_summaries

# Initialize atau update data chatbot
def initialize_or_update_chatbot_data(new_text=None, new_summary=None):
    try:
        # Inisialisasi model
        sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load data yang sudah ada (jika ada)
        if os.path.exists('chatbot_data.pkl'):
            old_index, _, old_texts, old_summaries = load_data()
            if new_text and new_summary:
                texts = old_texts + new_text
                summaries = old_summaries + [new_summary]
            else:
                texts = old_texts
                summaries = old_summaries
        else:
            # Jika tidak ada data dan tidak ada input baru, gunakan data default
            if not new_text or not new_summary:
                texts, summaries = initialize_default_data()
            else:
                texts = new_text
                summaries = [new_summary]
        
        # Buat embeddings dan index baru
        embeddings = sentence_model.encode(texts)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        
        # Simpan data
        save_data(index, sentence_model, texts, summaries)
        
        # Update session state
        st.session_state.index = index
        st.session_state.sentence_model = sentence_model
        st.session_state.sentences = texts
        st.session_state.summaries = summaries
        st.session_state.chatbot_data_loaded = True
        
        return True
        
    except Exception as e:
        st.error(f"Error updating chatbot data: {str(e)}")
        return False

# Fungsi untuk mendapatkan konteks yang relevan
def get_relevant_context(query, index, sentence_model, sentences, summaries, top_k=8):
    # Encode query
    query_embedding = sentence_model.encode([query])[0]
    
    # Cari dokumen yang relevan menggunakan FAISS
    D, I = index.search(np.array([query_embedding]), top_k)
    
    # Filter berdasarkan threshold similarity
    threshold = 0.6  # Hanya ambil yang similarity-nya > 60%
    max_distance = 2.0  # FAISS distance threshold
    
    contexts = []
    seen = set()
    
    for dist, idx in zip(D[0], I[0]):
        if idx >= len(sentences) or dist > max_distance:
            continue
            
        sentence = sentences[idx]
        
        # Skip jika terlalu pendek atau duplikat
        if len(sentence.split()) < 4 or sentence in seen:
            continue
            
        # Hitung similarity score (convert distance to similarity)
        similarity = 1 - (dist / 4)  # Normalize distance to 0-1 range
        
        if similarity > threshold:
            # Ambil kalimat sebelum dan sesudah untuk konteks
            start_idx = max(0, idx - 1)
            end_idx = min(len(sentences), idx + 2)
            
            context_window = sentences[start_idx:end_idx]
            context = ' '.join(context_window)
            
            if context not in seen:
                contexts.append({
                    'text': context,
                    'similarity': similarity,
                    'summary': summaries[idx] if idx < len(summaries) else None
                })
                seen.add(context)
    
    return contexts

# Fungsi untuk memformat jawaban
def format_answer(query, contexts):
    if not contexts:
        return "Maaf, saya tidak menemukan informasi yang cukup relevan untuk menjawab pertanyaan tersebut."
    
    # Urutkan konteks berdasarkan similarity score
    sorted_contexts = sorted(contexts, key=lambda x: x['similarity'], reverse=True)
    
    # Ambil konteks terbaik (max 3)
    best_contexts = sorted_contexts[:3]
    
    # Format jawaban
    answer_parts = []
    
    for ctx in best_contexts:
        if ctx['similarity'] < 0.6:  # Skip jika similarity terlalu rendah
            continue
            
        answer_parts.append(f"{ctx['text']}\n\n")  # Tambah baris kosong antara paragraf
        if ctx['summary'] and ctx['summary'] not in ctx['text']:
            answer_parts.append(f"Ringkasan: {ctx['summary']}\n\n")
    
    if not answer_parts:
        return "Maaf, saya tidak menemukan informasi yang cukup relevan untuk menjawab pertanyaan tersebut."
    
    return "".join(answer_parts)

# Fungsi untuk chatbot
def chatbot(queries, index, sentence_model, sentences, summaries):
    # Pastikan sentences tidak kosong
    if not sentences:
        return ["Maaf, belum ada dokumen yang diproses."]
    
    # Gabungkan sentences dengan summaries jika ada
    all_texts = []
    for i, sent in enumerate(sentences):
        all_texts.append(sent)
        if i < len(summaries) and summaries[i]:
            all_texts.append(summaries[i])
    
    responses = []
    for query in queries:
        if not query.strip():
            continue
        
        try:
            # Cari matches terbaik
            matches = find_best_match(query.strip(), all_texts, sentence_model)
            
            if not matches:
                responses.append("Maaf, saya tidak menemukan informasi yang relevan untuk pertanyaan tersebut.")
                continue
            
            # Format jawaban tanpa bullet points
            answer = ""
            for text, score in matches:
                answer += f"{text}\n\n"  # Tambah baris kosong antara paragraf
            
            responses.append(answer.strip())
            
        except Exception as e:
            responses.append(f"Maaf, terjadi kesalahan: {str(e)}")
    
    return responses

def find_best_match(query, sentences, sentence_model):
    # Encode query dan semua kalimat
    query_embedding = sentence_model.encode([query])[0]
    sentence_embeddings = sentence_model.encode(sentences)
    
    # Hitung cosine similarity
    similarities = np.dot(sentence_embeddings, query_embedding) / (
        np.linalg.norm(sentence_embeddings, axis=1) * np.linalg.norm(query_embedding)
    )
    
    # Ambil indeks dengan similarity tertinggi
    best_indices = np.argsort(similarities)[::-1][:3]  # Top 3 matches
    
    results = []
    for idx in best_indices:
        if similarities[idx] > 0.5:  # Minimal 50% similar
            results.append((sentences[idx], similarities[idx]))
    
    return results

# Inisialisasi Google Drive service di backend
CREDENTIALS = {
    "type": "service_account",
    "project_id": "sunlit-ace-443502",
    "private_key_id": "740836f0a89c3ee96b693a332c52e201e0d9854e",  # Sesuaikan dengan private_key_id Anda
    "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvAIBADANBgkqhkiG9w0BAQEFAASCBKYwggSiAgEAAoIBAQC2na9meL6l3g8D\nef/HpsEdcVoNOtz7VWKHYdlCexM1wVxgYYZmxQ6qwtMCTnmdP2lFP3hmyXDpfiME\nM6Ev2KW4RdpRyYWSFbiHkNiY+uQ7aomvgDO4oVZwVxyRE8zNksZfX8fJOqWpbOeF\nmrP4I4Y3cSmsp12YYWbhQG4njoMi4hbVDAU8Wtkv7nswvI4PwsSLOXhtlHPwI7IE\ng+JEQxw4iSZC6W+MP+WymDDZw8VtolesPKaI2Rsykv7iMlWwennHkwNvR92Vpntw\nhfKHwd2yJEmES51E5Y/6fVIH8OzDL7Dxg5wqJBA8GdyVmbU5QdWGN99rO2bZno0X\nH+wvdJAdAgMBAAECggEAFx1zj8V14FnwzZBaO4PUiu0HqIgMY7XloFxW207n2hSN\nJFgD4qtL1XqOqgqT4z8dDJphm6Ug6TVBqmz8mKlUJLSW02ZuRVUBhAtmF0seq5Sr\nM+9G3LZzUgn8wDJg6J6NBJKDn+mQAZea4LL5181rVkY5L7UJqFmf46A7sT8dQ0Ni\n9ABbwclv62xAawSemiTiL3Ocu+ZO7MvLN0fRd+wuZJqB9M7QwI7QvWdl9cIkkcWZ\n2D7ClHXkTpHNK96o6+NRGdd2YWIGR1mEAzb3vsGdpQ4MiyVRp8uoTIdtsHj0jMjj\nRKmBYeEeTR0xj4H1M0phy39hkcAA77xoiQoQiQzWEwKBgQDwY+EPktypdkR8kgvk\nok1KI4EPxMKw3pQ/akNrHtsfUokp++0y6WW1oWu/PZItpwGIV4tYfdxQMeWzna45\n60cLKs88T+Q6slV1QintBIyXV9mMlOpf+hS4Qs95bbgJMtOCnrZAQhuaFyIhm7DV\nI0dLe721y7YKMFgzZXUcO3apCwKBgQDCeWXZIJ9S5QhasreWEfWJk/bs+MbP40Nw\nWPNmaZo36NuJrxlSFnuncVBic/bjB+hPJIg8Rq/teX2rYEucLYAudvl/l/peEuTJ\nXTicXYBYVsCi0khzD/ydYtpizAqZG0NqAV6N7yVy8XZcA3HjRa+Tz9Zqs/qb8qo4\ngW4H8fl0dwKBgEMxuARRaerYizZC4J4tG+ugVwAgYMdtwASl4Gh9/IQZ3wtkRx5X\nDT4i++o9/LFUIGKLDgFTVRT5jZqSddPsxzQA6GKUdlhN5wNa1jtRbTcUsFPSgF0J\nm2cSDcqEd8/ibCrz0D/P+sUsuDaqvVgmf1RDJ3k8kwrwDod3Ua/tkzKXAoGATU8/\nSsqqK6T65jtnuhPXnWKh5eCcAGajF5V05UCT2ygJpjlignyHma/1Ob5J5kTteBDY\ny8V6CJikea1lQWfhLheD5dI/6IfwRZB2gcq1y+ho2hFoVb2EOfjjQiVFDUqGSSzU\nLu5j38bXu4pvCt9YBhW4cmCr/rTAMIhbenMaLM8CgYArj+N1BYQ6DbCR8l9jZqom\nGwV+XGvT9O3tSPLzqGG8KkkyxfPR83RzJDvQUGZixCAVSAayIPUKlqj0VhXrgHTp\nARKkv0kLtboIQUiX0pAALPm+09GK9w+jG6KrHvfeXwajrm89LQ64BZvimj992Tcd\nLUuRvtqzvF9dYJby2yg/xw==\n-----END PRIVATE KEY-----\n",  # Sesuaikan dengan private key Anda
    "client_email": "irfan-684@sunlit-ace-443502-f2.iam.gserviceaccount.com",
    "client_id": "110509530191527507937",  # Sesuaikan dengan client_id Anda
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
    "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/irfan-684%40sunlit-ace-443502-f2.iam.gserviceaccount.com"
}

# ID Folder Google Drive yang akan digunakan
FOLDER_ID = '1iJRHIT0bXha60YicUD4T7q--681b1PfK'

def init_google_drive():
    try:
        # Gunakan kredensial yang sudah didefinisikan
        creds = service_account.Credentials.from_service_account_info(
            CREDENTIALS, 
            scopes=["https://www.googleapis.com/auth/drive"]
        )
        service = build("drive", "v3", credentials=creds)
        return service
    except Exception as e:
        st.error(f"Error initializing Google Drive: {str(e)}")
        return None

# Inisialisasi service saat startup
service = init_google_drive()

#Bagian Streamlit
st.title("Smart Assistant")
st.write("Selamat datang! Silakan ajukan pertanyaan tentang berbagai topik yang Anda inginkan.")

# Inisialisasi data jika belum ada
if 'chatbot_data_loaded' not in st.session_state or not st.session_state.chatbot_data_loaded:
    initialize_or_update_chatbot_data()

# Tab untuk upload file dan tanya jawab
tab1, tab2 = st.tabs(["Upload File", "Tanya Jawab"])

# Tab Upload File
with tab1:
    uploaded_file = st.file_uploader("Unggah file lokal", type=["pdf", "docx", "csv", "txt"])
    if uploaded_file:
        with st.spinner("Memproses file..."):
            try:
                # Upload ke Google Drive
                local_file_path = f"/tmp/{uploaded_file.name}"
                with open(local_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Upload file dan dapatkan file ID
                file_metadata = {
                    'name': uploaded_file.name,
                    'parents': [FOLDER_ID]
                }
                media = MediaFileUpload(local_file_path, resumable=True)
                file = service.files().create(body=file_metadata, media_body=media, fields='id,name,mimeType').execute()
                
                st.success(f"File berhasil diunggah ke Google Drive")
                
                # Proses file yang baru diupload
                with st.spinner("Memproses file untuk chatbot..."):
                    processed_text, summary = process_single_file(service, file['id'], file['name'], file['mimeType'])
                    if processed_text and summary:
                        if initialize_or_update_chatbot_data(processed_text, summary):
                            st.success("File berhasil diproses dan data chatbot telah diperbarui")
                        else:
                            st.error("Gagal memperbarui data chatbot")
                    else:
                        st.error("Gagal memproses file")
                        
            except Exception as e:
                st.error(f"Terjadi kesalahan: {str(e)}")

# Tab Tanya Jawab
with tab2:
    # Cek apakah data chatbot sudah dimuat
    chatbot_ready = False
    if "chatbot_data_loaded" in st.session_state and st.session_state.chatbot_data_loaded:
        chatbot_ready = True
    
    if not chatbot_ready:
        st.warning("Belum ada data chatbot. Silakan upload dan proses file terlebih dahulu di tab Upload File")
    else:
        # Inisialisasi state untuk jawaban jika belum ada
        if "current_response" not in st.session_state:
            st.session_state.current_response = None
        
        # Callback untuk memproses pertanyaan
        def process_query():
            if st.session_state.query_input:
                with st.spinner("Memproses pertanyaan..."):
                    try:
                        # Ambil data dari session state
                        index = st.session_state.index
                        sentence_model = st.session_state.sentence_model
                        sentences = st.session_state.sentences
                        summaries = st.session_state.summaries
                        
                        # Panggil fungsi chatbot
                        responses = chatbot(st.session_state.query_input.split(';'), 
                                         index, sentence_model, sentences, summaries)
                        
                        # Simpan jawaban ke session state
                        st.session_state.current_response = responses
                        
                    except Exception as e:
                        st.error(f"Terjadi kesalahan saat memproses pertanyaan: {str(e)}")

        # Text area untuk pertanyaan dengan callback
        st.text_area("Masukkan pertanyaan Anda (pisahkan dengan ';')", 
                    key="query_input",
                    on_change=process_query)
        
        # Tombol untuk bertanya
        if st.button("Tanya Chatbot"):
            process_query()
        
        # Tampilkan hasil jika ada
        if st.session_state.current_response:
            st.subheader("Jawaban:")
            for i, response in enumerate(st.session_state.current_response, 1):
                st.write(response)
