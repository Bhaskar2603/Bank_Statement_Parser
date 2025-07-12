# 🏦 AI Bank Statement Parser — A Novel Hybrid AI Agent

🚀 [Live Demo](https://huggingface.co/spaces/Bhaskar2611/BankStatement_Parser)

**AI Bank Statement Parser** is an intelligent financial data extractor designed to convert unstructured or semi-structured **bank statements** (PDFs & Excels) into clean, structured transaction summaries.

> A hybrid AI agent that fuses **rule-based parsing** for clean formats and a **Mistral‑7B LLM fallback** for messy, scanned, or unstructured statements.

---

## 📌 Features

- 📤 Upload support for both **native PDFs**, **scanned PDFs**, and **Excel files**
- 🧠 Smart hybrid pipeline:
  - ✅ Rule-based logic for standard table-like statements
  - 🤖 LLM fallback (via [Mistral‑7B](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)) for irregular or noisy text
- 🔍 Extracted fields:
  - **Date**, **Description**, **Amount**, **Credit/Debit Label**, **Category**
- 🧾 PDF report download of parsed transactions
- ⚡ Fast, responsive **Gradio UI**
- 🖼️ OCR support for scanned image-based PDFs

---

## 🛠️ Tech Stack

| Tool/Library       | Purpose                             |
|--------------------|-------------------------------------|
| **Gradio**         | User interface                      |
| **Hugging Face Spaces** | Hosting & LLM inference         |
| **pdfplumber**     | Text extraction from PDFs           |
| **pytesseract**    | OCR for scanned PDF images          |
| **pdf2image**      | Convert PDF pages to images         |
| **pandas**         | Data manipulation                   |
| **FPDF**           | Generating final output PDFs        |
| **Mistral‑7B LLM** | Parsing irregular/unstructured text |

---

## 📁 Example Files

- 🧾 [Sample PDF Bank Statement](https://drive.google.com/file/d/1Yc7JOZK6z2BZEJhVLID4hU7m4etggLr0/view?usp=sharing)
- 📊 [Sample Excel Statement](https://docs.google.com/spreadsheets/d/1xTyWx1qrAPIvdldeq7qyzycSR1armtdq/edit?usp=sharing)

---

## 🧠 How It Works

```
          ┌────────────┐
     ┌───▶│ Upload File│
     │    └────────────┘
     │          ↓
     │   ┌───────────────┐
     │   │ Preprocessing │
     │   └───────────────┘
     │          ↓
     │   ┌─────────────────────┐
     │   │ Rule-Based Parser   │──┐
     │   └─────────────────────┘  │
     │                            ▼
     │    ┌────────────────────────────┐
     └───▶│ LLM Fallback (Mistral‑7B)  │
          └────────────────────────────┘
                      ↓
              ┌──────────────┐
              │ Parsed Output│
              └──────────────┘
                      ↓
             ┌────────────────┐
             │ Download as PDF│
             └────────────────┘
```

---

## 🚧 Future Work

- 🧩 **Layout-Aware Models**: Integration with LayoutLM for handling multi-column or stylized bank formats
- 🔎 **Context-Sensitive Normalization**: For better parsing of edge cases like missing dates or mixed formats
- 📊 **Charts and Summaries**: Add visual analytics like monthly summaries and category-wise spending

---

## 🙏 Acknowledgements

- 💡 Thanks to **Sankari Nair** for the project idea and domain context
- ❤️ Gradio for the incredible UI framework
- 🔥 Hugging Face for free open-source LLM hosting and inference
- 📚 Mistral community for fast and powerful LLMs

---

## 👨‍💻 Author

**Jyothula Bhaskar**  
B.Tech in Computer Science & Engineering (AI & ML)  
[LinkedIn](https://www.linkedin.com/in/bhaskar-jyothula-974bbb271/) | [Hugging Face](https://huggingface.co/Bhaskar2611) | [GitHub](https://github.com/Bhaskar2603) | [Kaggle](https://www.kaggle.com/bhaskarjyothula)

---

title: BankStatement Parser
emoji: ⚡
colorFrom: yellow
colorTo: green
sdk: gradio
sdk_version: 5.33.0
app_file: app.py
pinned: false
license: apache-2.0
---

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for full details.

---




