import os
import re
import json
import gradio as gr
import pandas as pd
import pdfplumber
import pytesseract
from pdf2image import convert_from_path
from huggingface_hub import InferenceClient
from fpdf import FPDF  # Added for PDF generation
import tempfile  # Added for temporary file handling

# Initialize with reliable free model
hf_token = os.getenv("HF_TOKEN")
client = InferenceClient(model="mistralai/Mistral-7B-Instruct-v0.2", token=hf_token)

def extract_excel_data(file_path):
    """Extract text from Excel file"""
    df = pd.read_excel(file_path, engine='openpyxl')
    return df.to_string(index=False)

def extract_text_from_pdf(pdf_path, is_scanned=False):
    """Extract text from PDF with fallback OCR"""
    try:
        # Try native PDF extraction first
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for page in pdf.pages:
                # Extract tables first for structured data
                tables = page.extract_tables()
                for table in tables:
                    for row in table:
                        text += " | ".join(str(cell) for cell in row) + "\n"
                    text += "\n"
                
                # Extract text for unstructured data
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
            return text
    except Exception as e:
        print(f"Native PDF extraction failed: {str(e)}")
        # Fallback to OCR for scanned PDFs
        images = convert_from_path(pdf_path, dpi=200)
        text = ""
        for image in images:
            text += pytesseract.image_to_string(image) + "\n"
        return text

def parse_bank_statement(text, file_type):
    """Parse bank statement using LLM with fallback to rule-based parser"""
    # Clean text differently based on file type
    cleaned_text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    
    if file_type == 'pdf':
        # PDF-specific cleaning
        cleaned_text = re.sub(r'Page \d+ of \d+', '', cleaned_text, flags=re.IGNORECASE)
        cleaned_text = re.sub(r'CropBox.*?MediaBox', '', cleaned_text, flags=re.IGNORECASE)
        
        # Keep only lines that look like transactions
        transaction_lines = []
        for line in cleaned_text.split('\n'):
            if re.match(r'^\d{4}-\d{2}-\d{2}', line):  # Date pattern
                transaction_lines.append(line)
            elif '|' in line and any(x in line for x in ['Date', 'Amount', 'Balance']):
                transaction_lines.append(line)
        
        cleaned_text = "\n".join(transaction_lines)
    
    print(f"Cleaned text sample: {cleaned_text[:200]}...")
    
    # Try rule-based parsing first for structured data
    rule_based_data = rule_based_parser(cleaned_text)
    if rule_based_data["transactions"]:
        print("Using rule-based parser results")
        return rule_based_data
    
    # Fallback to LLM for unstructured data
    print("Falling back to LLM parsing")
    return llm_parser(cleaned_text)

def llm_parser(text):
    """LLM parser for unstructured text"""
    # Craft precise prompt with strict JSON formatting instructions
    prompt = f"""
<|system|>
You are a financial data parser. Extract transactions from bank statements and return ONLY valid JSON.
</s>
<|user|>
Extract all transactions from this bank statement with these exact fields:
- date (format: YYYY-MM-DD)
- description
- amount (format: 0.00)
- debit (format: 0.00)
- credit (format: 0.00)
- closing_balance (format: 0.00 or -0.00 for negative)
- category
Statement text:
{text[:3000]}  [truncated if too long]
Return JSON with this exact structure:
{{
  "transactions": [
    {{
      "date": "2025-05-08",
      "description": "Company XYZ Payroll",
      "amount": "8315.40",
      "debit": "0.00",
      "credit": "8315.40",
      "closing_balance": "38315.40",
      "category": "Salary"
    }}
  ]
}}
RULES:
1. Output ONLY the JSON object with no additional text
2. Keep amounts as strings with 2 decimal places
3. For missing values, use empty strings
4. Convert negative amounts to format "-123.45"
5. Map categories to: Salary, Groceries, Medical, Utilities, Entertainment, Dining, Misc
</s>
<|assistant|>
"""
    
    try:
        # Call LLM via Hugging Face Inference API
        response = client.text_generation(
            prompt,
            max_new_tokens=2000,
            temperature=0.01,
            stop=["</s>"]  # Updated to 'stop' parameter
        )
        print(f"LLM Response: {response}")
        
        # Validate and clean JSON response
        response = response.strip()
        if not response.startswith('{'):
            # Find the first { and last } to extract JSON
            start_idx = response.find('{')
            end_idx = response.rfind('}')
            if start_idx != -1 and end_idx != -1:
                response = response[start_idx:end_idx+1]
        
        # Parse JSON and validate structure
        data = json.loads(response)
        if "transactions" not in data:
            raise ValueError("Missing 'transactions' key in JSON")
            
        return data
    except Exception as e:
        print(f"LLM Error: {str(e)}")
        return {"transactions": []}

def rule_based_parser(text):
    """Enhanced fallback parser for structured tables"""
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    # Find header line - more flexible detection
    header_index = None
    header_patterns = [
        r'Date\b', r'Description\b', r'Amount\b', 
        r'Debit\b', r'Credit\b', r'Closing\s*Balance\b', r'Category\b'
    ]
    
    # First try: Look for a full header line
    for i, line in enumerate(lines):
        if all(re.search(pattern, line, re.IGNORECASE) for pattern in header_patterns[:3]):
            header_index = i
            break
    
    # Second try: Look for any header indicators
    if header_index is None:
        for i, line in enumerate(lines):
            if any(re.search(pattern, line, re.IGNORECASE) for pattern in header_patterns):
                header_index = i
                break
    
    # Third try: Look for pipe-delimited headers
    if header_index is None:
        for i, line in enumerate(lines):
            if '|' in line and any(p in line for p in ['Date', 'Amount', 'Balance']):
                header_index = i
                break
    
    if header_index is None:
        return {"transactions": []}
    
    data_lines = lines[header_index + 1:]
    transactions = []
    
    for line in data_lines:
        # Handle both pipe-delimited and space-delimited formats
        if '|' in line:
            parts = [p.strip() for p in line.split('|') if p.strip()]
        else:
            # Space-delimited format - split by 2+ spaces
            parts = re.split(r'\s{2,}', line)
        
        # Skip lines that don't have enough parts
        if len(parts) < 7:
            continue
            
        try:
            # Handle transaction date validation
            if not re.match(r'\d{4}-\d{2}-\d{2}', parts[0]):
                continue
                
            transactions.append({
                "date": parts[0],
                "description": parts[1],
                "amount": format_number(parts[2]),
                "debit": format_number(parts[3]),
                "credit": format_number(parts[4]),
                "closing_balance": format_number(parts[5]),
                "category": parts[6]
            })
        except Exception as e:
            print(f"Error parsing line: {str(e)}")
    
    return {"transactions": transactions}

def format_number(value):
    """Format numeric values consistently"""
    if not value or str(value).lower() in ['nan', 'nat']:
        return "0.00"
        
    # If it's already a number, format directly
    if isinstance(value, (int, float)):
        return f"{value:.2f}"
        
    # Clean string values
    value = str(value).replace(',', '').replace('$', '').strip()
    
    # Handle negative numbers in parentheses
    if '(' in value and ')' in value:
        value = '-' + value.replace('(', '').replace(')', '')
    
    # Handle empty values
    if not value:
        return "0.00"
    
    # Standardize decimal format
    if '.' not in value:
        value += '.00'
    
    # Ensure two decimal places
    try:
        num_value = float(value)
        return f"{num_value:.2f}"
    except ValueError:
        # If we can't convert to float, return original but clean it
        return value.split('.')[0] + '.' + value.split('.')[1][:2].ljust(2, '0')

def process_file(file, is_scanned=False):
    """Main processing function"""
    if not file:
        return empty_df()
    
    file_path = file.name
    file_ext = os.path.splitext(file_path)[1].lower()
    
    try:
        if file_ext == '.xlsx':
            # Directly process Excel files without text conversion
            df = pd.read_excel(file_path, engine='openpyxl')
            
            # Normalize column names
            df.columns = df.columns.str.strip().str.lower()
            
            # Create mapping to expected columns
            col_mapping = {
                'date': 'date',
                'description': 'description',
                'amount': 'amount',
                'debit': 'debit',
                'credit': 'credit',
                'closing balance': 'closing_balance',
                'closing': 'closing_balance',
                'balance': 'closing_balance',
                'category': 'category'
            }
            
            # Create output DataFrame with required columns
            output_df = pd.DataFrame()
            for col in ['date', 'description', 'amount', 'debit', 'credit', 'closing_balance', 'category']:
                if col in df.columns:
                    output_df[col] = df[col]
                elif any(alias in col_mapping and col_mapping[alias] == col for alias in df.columns):
                    # Find alias
                    for alias in df.columns:
                        if alias in col_mapping and col_mapping[alias] == col:
                            output_df[col] = df[alias]
                            break
                else:
                    output_df[col] = ""
            
            # Format numeric columns
            for col in ['amount', 'debit', 'credit', 'closing_balance']:
                output_df[col] = output_df[col].apply(format_number)
            
            # Rename columns for display
            output_df.columns = ["Date", "Description", "Amount", "Debit", 
                               "Credit", "Closing Balance", "Category"]
            return output_df

        elif file_ext == '.pdf':
            text = extract_text_from_pdf(file_path, is_scanned=is_scanned)
            parsed_data = parse_bank_statement(text, 'pdf')
            df = pd.DataFrame(parsed_data["transactions"])
            
            # Ensure all required columns exist
            required_cols = ["date", "description", "amount", "debit", 
                            "credit", "closing_balance", "category"]
            for col in required_cols:
                if col not in df.columns:
                    df[col] = ""
            
            # Format columns properly
            df.columns = ["Date", "Description", "Amount", "Debit", 
                         "Credit", "Closing Balance", "Category"]
            return df
        
        else:
            return empty_df()
    
    except Exception as e:
        print(f"Processing error: {str(e)}")
        return empty_df()

def empty_df():
    """Return empty DataFrame with correct columns"""
    return pd.DataFrame(columns=["Date", "Description", "Amount", "Debit", 
                               "Credit", "Closing Balance", "Category"])

# New function to generate PDF from DataFrame
def generate_pdf(df):
    """Generate PDF from DataFrame and return file path"""
    if df.empty:
        return None
        
    # Create a PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=8)  # Smaller font to fit more data
    
    # Set column widths
    col_widths = [22, 65, 20, 15, 15, 25, 20]  # Adjusted to fit all columns
    
    # Headers
    headers = df.columns.tolist()
    for i, header in enumerate(headers):
        pdf.cell(col_widths[i], 10, header, border=1)
    pdf.ln()
    
    # Data
    for _, row in df.iterrows():
        for i, col in enumerate(headers):
            # Truncate long descriptions
            value = str(row[col])
            if headers[i] == "Description" and len(value) > 30:
                value = value[:27] + "..."
            pdf.cell(col_widths[i], 10, value, border=1)
        pdf.ln()
    
    # Save to temporary file
    temp_file = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    temp_file.close()
    pdf.output(temp_file.name)
    return temp_file.name

# Modified Gradio Interface
with gr.Blocks() as interface:  # Changed to Blocks for more control
    gr.Markdown("## AI Bank Statement Parser")
    gr.Markdown("Extract structured transaction data from PDF/Excel bank statements")
    
    # File input
    file_input = gr.File(label="Upload Bank Statement (PDF/Excel)")
    
    # Output dataframe
    output_df = gr.Dataframe(
        label="Parsed Transactions",
        headers=["Date", "Description", "Amount", "Debit", "Credit", "Closing Balance", "Category"],
        datatype=["date", "str", "number", "number", "number", "number", "str"]
    )
    
    # State to store the processed DataFrame
    state_df = gr.State(value=pd.DataFrame())
    
    # Download button (initially hidden)
    download_btn = gr.DownloadButton(
        "Download as PDF",
        visible=False,
        elem_classes="download-btn"
    )
    
    # Process file and update state
    def process_and_store(file):
        df = process_file(file)
        return df, df, gr.DownloadButton(visible=not df.empty)
    
    # Connect components
    file_input.change(
        process_and_store,
        inputs=[file_input],
        outputs=[output_df, state_df, download_btn]
    )
    
    # Generate PDF when download button is clicked
    def on_download_click(df):
        return generate_pdf(df)
    
    download_btn.click(
        on_download_click,
        inputs=[state_df],
        outputs=[download_btn]
    )

# Add custom CSS for the download button position
interface.css = """
.download-btn {
    margin-top: 20px !important;
    margin-bottom: 30px !important;
}
"""

if __name__ == "__main__":
    interface.launch()