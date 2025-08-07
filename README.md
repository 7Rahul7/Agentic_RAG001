
```markdown
# 🤖 Agentic RAG PDF Chatbot

This is a smart assistant that can **read and answer questions** from PDF documents. If the PDFs don't have enough information, it will automatically **search the web** (using DuckDuckGo) and give a better answer. It's powered by **Google Gemini AI** and works right from your terminal!

---

## 📁 Folder Structure

```

AgentRAG/
│
├── data/                  # 📂 Put your PDF files here
│   └── your-pdf.pdf
├── .env                   # 🔑 Store your Gemini API key here
├── app.py                 # 🚀 Main chatbot program
├── main.py                # (Optional helper)
├── requirements.txt       # 📦 List of needed Python libraries
├── pyproject.toml         # 📦 Project setup (optional)
├── README.md              # 📘 You're reading it!
└── .gitignore             # ❌ Ignores .env and /data/ from Git

````

---

## ⚙️ Setup Instructions (Step-by-Step)

### 🐍 1. Create Virtual Environment

We use `uv` to manage a clean Python environment .

```bash
pip install uv
uv venv
````

Then activate it:

Or we can do 

```bash
uv sync
````

* On **Windows**:

  ```bash
  source .venv\Scripts\activate
  ```

* On **Mac/Linux**:

  ```bash
  source .venv/bin/activate
  ```

---

### 📦 2. Install Dependencies

```bash
uv pip install -r requirements.txt
```

---

### 🧾 3. Add Your PDFs

* Create a folder named `data` if it doesn't exist:

```bash
mkdir data
```

* Then copy your PDF files (e.g. notes, papers, books) into the `data/` folder.

---

### 🔑 4. Set up Gemini API Key

Create a `.env` file in the main folder and paste your **Google Gemini API Key**:

```
GOOGLE_API_KEY=your_gemini_api_key_here
```

> 🛡️ `.env` is automatically hidden from Git (see `.gitignore`), so your key stays safe.

You can get a free Gemini key here: [https://makersuite.google.com/app/apikey](https://makersuite.google.com/app/apikey)

---

## 🚀 Running the App

Once setup is done, just run:

```bash
python app.py
```

Then ask questions like:

```
You: What is the main topic of chapter 2?
```

If your PDF doesn't have the answer, it will smartly look it up online and tell you!

To exit, type:

```
You: exit
```

---

## 💡 Features

* ✅ Reads your PDFs and answers based on them
* 🔍 Searches the web if PDFs are not enough
* 🧠 Uses Google Gemini (Free API)
* 📄 Works with any number of PDFs
* 💬 Chat-like interface in terminal

---





