# 🔍 Streamlit Tesseract OCR Test

A simple Streamlit application to test Tesseract OCR deployment on Streamlit Cloud.

## Features

- 📤 Upload images (PNG, JPG, JPEG, GIF, BMP)
- 🔍 Extract text using Tesseract OCR
- 📊 Display character and word counts
- 🔧 Debug information to check Tesseract installation

## Files Structure

```
streamlit/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── packages.txt        # System packages for Streamlit Cloud
└── README.md          # This file
```

## Local Development

1. Clone the repository:
```bash
git clone https://github.com/yeganehmohammadii98/streamlit.git
cd streamlit
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install Tesseract (if not already installed):
   - **Ubuntu/Debian:** `sudo apt-get install tesseract-ocr tesseract-ocr-eng`
   - **macOS:** `brew install tesseract`
   - **Windows:** Download from [GitHub releases](https://github.com/UB-Mannheim/tesseract/wiki)

4. Run the app:
```bash
streamlit run app.py
```

## Deployment on Streamlit Cloud

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Deploy your app using the GitHub repository URL
4. The `packages.txt` file will automatically install Tesseract on the server

## Testing the App

1. Upload an image containing text
2. Click "Extract Text with Tesseract"
3. View the extracted text results
4. Check the debug section to verify Tesseract installation

## Troubleshooting

- If you see "Tesseract not found" errors, ensure `packages.txt` is present
- For local development, make sure Tesseract is installed on your system
- Check the debug section in the app for system information

## Dependencies

- **streamlit**: Web app framework
- **pytesseract**: Python wrapper for Tesseract OCR
- **Pillow**: Image processing library

## License

MIT License