import streamlit as st
import pandas as pd
import os
from datetime import datetime
from PIL import Image
import sys
import traceback
import time
import logging
from typing import Dict, List, Any, Optional

# Database imports with error handling
try:
    from database.models import init_database, get_db_session, Invoice, OCRResult, FieldExtraction, UserFeedback

    DATABASE_AVAILABLE = True
except ImportError as e:
    st.error(f"Database import error: {e}")
    DATABASE_AVAILABLE = False


    # Create dummy functions for testing
    def init_database():
        return None, None


    def get_db_session():
        return None

# Utils imports with error handling
try:
    from utils.ocr_processor import OCRProcessor, get_tesseract_path
    from utils.file_handler import FileHandler, get_file_preview_info
    from utils.field_extractor import FieldExtractor, calculate_field_confidence_score
    from utils.learning_system import LearningSystem

    UTILS_AVAILABLE = True
except ImportError as e:
    st.warning(f"Utils import error: {e}")
    UTILS_AVAILABLE = False


    # Create dummy classes for testing
    class OCRProcessor:
        def __init__(self, path=None): pass

        def process_file(self, *args): return {'success': False, 'error': 'Not available'}


    class FileHandler:
        def validate_file(self, *args): return {'valid': True, 'file_type': 'unknown'}

        def get_file_content(self, *args): return b''


    class FieldExtractor:
        def extract_all_fields(self, *args): return {}


    class LearningSystem:
        def get_field_statistics(self): return {'total_extractions': 0, 'total_corrections': 0, 'accuracy_rate': 0.0}

        def apply_learned_patterns(self, *args): return None


    def get_tesseract_path():
        return 'tesseract'


    def get_file_preview_info(file):
        return {'name': file.name, 'type': file.type, 'size': file.size, 'size_mb': file.size / 1024 / 1024,
                'extension': os.path.splitext(file.name)[1]}


    def calculate_field_confidence_score(fields):
        return 0.0

# Plotly imports with error handling
try:
    import plotly.express as px
    import plotly.graph_objects as go

    PLOTLY_AVAILABLE = True
except ImportError:
    st.warning("Plotly not available - charts will be disabled")
    PLOTLY_AVAILABLE = False


    # Create dummy plotly objects
    class go:
        class Figure:
            def __init__(self, *args, **kwargs): pass

            def update_layout(self, *args, **kwargs): pass

            def update_xaxes(self, *args, **kwargs): pass

        class Bar:
            def __init__(self, *args, **kwargs): pass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Smart Invoice AI System",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Initialize database with error handling
@st.cache_resource
def initialize_database():
    """Initialize database connection (cached to avoid repeated calls)"""
    if DATABASE_AVAILABLE:
        try:
            return init_database()
        except Exception as e:
            st.error(f"Database initialization error: {e}")
            return None, None
    else:
        return None, None


def tesseract_test_page():
    """Simple Tesseract test page to verify OCR functionality"""
    st.header("🔍 Tesseract OCR Test")
    st.write("Test Tesseract installation and basic OCR functionality")

    # Try to import pytesseract with error handling
    try:
        import pytesseract
        TESSERACT_AVAILABLE = True
        st.success("✅ pytesseract imported successfully!")
    except ImportError as e:
        TESSERACT_AVAILABLE = False
        st.error(f"❌ Failed to import pytesseract: {e}")
        return
    except Exception as e:
        TESSERACT_AVAILABLE = False
        st.error(f"❌ Unexpected error importing pytesseract: {e}")
        return

    # Test Tesseract binary availability
    try:
        version = pytesseract.get_tesseract_version()
        st.success(f"✅ Tesseract binary found! Version: {version}")
    except Exception as e:
        st.error(f"❌ Tesseract binary not accessible: {e}")
        st.info("💡 Make sure packages.txt contains tesseract-ocr and tesseract-ocr-eng")
        return

    # File uploader for testing
    st.subheader("📤 Test Image Upload")
    uploaded_file = st.file_uploader(
        "Choose an image file for OCR testing",
        type=['png', 'jpg', 'jpeg', 'gif', 'bmp']
    )

    if uploaded_file is not None:
        try:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)

            # Show basic image info
            st.info(f"📏 Image size: {image.size} | 🎨 Mode: {image.mode}")

            # Add a button to trigger OCR
            if st.button("🔍 Extract Text with Tesseract"):
                with st.spinner("Processing image with OCR..."):
                    try:
                        # Extract text using Tesseract
                        extracted_text = pytesseract.image_to_string(image)

                        # Display results
                        st.success("✅ Text extraction completed!")
                        st.subheader("Extracted Text:")

                        if extracted_text.strip():
                            st.text_area("OCR Result:", extracted_text, height=200)

                            # Show statistics
                            char_count = len(extracted_text)
                            word_count = len(extracted_text.split())
                            st.info(f"📊 Characters: {char_count} | Words: {word_count}")
                        else:
                            st.warning("⚠️ No text found in the image")

                    except Exception as e:
                        st.error(f"❌ OCR Error: {str(e)}")
                        st.info("💡 This indicates Tesseract system package is not properly installed")

        except Exception as e:
            st.error(f"❌ Error loading image: {str(e)}")

    # System debug info
    if st.checkbox("🔧 Show Debug Info"):
        try:
            import platform
            import shutil

            st.write("**System Information:**")
            st.write(f"Platform: {platform.system()} {platform.release()}")
            st.write(f"Python: {platform.python_version()}")

            # Check tesseract paths
            st.write("**Tesseract Search:**")
            tesseract_which = shutil.which('tesseract')
            st.write(f"which tesseract: {tesseract_which}")

            common_paths = [
                '/usr/bin/tesseract',
                '/usr/local/bin/tesseract',
                '/app/.apt/usr/bin/tesseract'
            ]

            st.write("**Common paths check:**")
            for path in common_paths:
                exists = os.path.exists(path)
                st.write(f"{path}: {'✅' if exists else '❌'}")

        except Exception as e:
            st.write(f"Debug info error: {e}")


def main():
    # Initialize database
    engine, SessionLocal = initialize_database()

    # App header
    st.title("🤖 Smart Invoice AI System")
    st.markdown("### Upload invoices and extract key information automatically")

    # Sidebar for navigation
    with st.sidebar:
        st.header("📋 Navigation")
        page = st.selectbox(
            "Choose a page:",
            ["Upload & Process", "View History", "Feedback & Corrections", "AI Learning Dashboard", "Model Statistics",
             "Settings", "🔍 Tesseract Test"]
        )

    # Main content based on selected page
    if page == "Upload & Process":
        upload_page()
    elif page == "View History":
        history_page()
    elif page == "Feedback & Corrections":
        show_feedback_history()
    elif page == "AI Learning Dashboard":
        show_learning_dashboard()
    elif page == "Model Statistics":
        stats_page()
    elif page == "Settings":
        settings_page()
    elif page == "🔍 Tesseract Test":
        tesseract_test_page()


def upload_page():
    """Main upload and processing page"""
    st.header("📤 Upload Invoices")

    # Create two columns for better layout
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("File Upload")

        # File uploader with multiple file support
        uploaded_files = st.file_uploader(
            "Choose invoice files",
            type=['pdf', 'png', 'jpg', 'jpeg', 'csv'],
            accept_multiple_files=True,
            help="Supported formats: PDF, PNG, JPG, CSV"
        )

        # Display upload instructions
        if not uploaded_files:
            st.info("👆 Upload your invoice files using the file picker above")
            st.markdown("""
            **Supported formats:**
            - 📄 PDF files (scanned or digital)
            - 🖼️ Image files (PNG, JPG, JPEG)
            - 📊 CSV files (structured data)

            **Tips:**
            - You can upload multiple files at once
            - Make sure images are clear and readable
            - For best results, use high-resolution scans
            """)

    with col2:
        st.subheader("Upload Statistics")

        if DATABASE_AVAILABLE and get_db_session:
            db_session = get_db_session()
            try:
                total_invoices = db_session.query(Invoice).count() if db_session else 0
                processed_today = 0
                if db_session:
                    processed_today = db_session.query(Invoice).filter(
                        Invoice.upload_date >= datetime.now().date()
                    ).count()

                st.metric("Total Invoices", total_invoices)
                st.metric("Processed Today", processed_today)
                st.metric("AI Accuracy", "0.0%")

            except Exception as e:
                st.error(f"Database error: {e}")
            finally:
                if db_session:
                    db_session.close()
        else:
            st.info("Database not available")
            st.metric("Total Invoices", "N/A")
            st.metric("Processed Today", "N/A")
            st.metric("AI Accuracy", "N/A")

    # Process uploaded files
    if uploaded_files:
        st.subheader("📋 Uploaded Files")

        for i, uploaded_file in enumerate(uploaded_files):
            with st.expander(f"📎 {uploaded_file.name} ({uploaded_file.size} bytes)"):
                col1, col2, col3 = st.columns([2, 1, 1])

                with col1:
                    st.write(f"**File type:** {uploaded_file.type}")
                    st.write(f"**Size:** {uploaded_file.size:,} bytes")

                with col2:
                    if st.button(f"Preview", key=f"preview_{i}"):
                        preview_file(uploaded_file)

                with col3:
                    if st.button(f"Process", key=f"process_{i}"):
                        process_file(uploaded_file)

        # Bulk actions
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 2])

        with col1:
            if st.button("🔄 Process All", type="primary"):
                process_all_files(uploaded_files)

        with col2:
            if st.button("💾 Save All"):
                save_all_files(uploaded_files)


def preview_file(uploaded_file):
    """Enhanced preview with file information"""
    try:
        # Get file info
        file_info = get_file_preview_info(uploaded_file)

        # Display file information
        col1, col2 = st.columns([1, 1])
        with col1:
            st.write(f"**📄 Name:** {file_info['name']}")
            st.write(f"**📊 Type:** {file_info['type']}")
        with col2:
            st.write(f"**💾 Size:** {file_info['size_mb']:.2f} MB")
            st.write(f"**📎 Extension:** {file_info['extension']}")

        # Show content preview
        if uploaded_file.type.startswith('image/'):
            image = Image.open(uploaded_file)
            st.image(image, caption=f"Preview of {uploaded_file.name}", width=500)

            # Show image details
            st.info(f"📐 Dimensions: {image.size[0]} × {image.size[1]} pixels")

        elif uploaded_file.type == 'application/pdf':
            st.info("📄 PDF files will be converted to images for OCR processing")
            st.write("Click 'Process' to extract text using OCR")

        elif uploaded_file.type == 'text/csv':
            df = pd.read_csv(uploaded_file)
            st.dataframe(df.head(10), use_container_width=True)
            st.info(f"📊 CSV contains {len(df)} rows and {len(df.columns)} columns")

    except Exception as e:
        st.error(f"Error previewing file: {e}")


def process_file(uploaded_file):
    """Process a single file with OCR and field extraction"""
    st.info("⚠️ Full processing functionality requires database and utils modules")
    st.info("📝 For now, use the 'Tesseract Test' page to test OCR functionality")


def process_all_files(uploaded_files):
    """Process all uploaded files"""
    st.info("⚠️ Bulk processing functionality will be available once individual processing is working")


def save_all_files(uploaded_files):
    """Save all files to uploads directory"""
    os.makedirs("uploads", exist_ok=True)
    saved_count = 0

    for uploaded_file in uploaded_files:
        try:
            # Save file to uploads directory
            file_path = os.path.join("uploads", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            saved_count += 1
        except Exception as e:
            st.error(f"Error saving {uploaded_file.name}: {e}")

    st.success(f"💾 Saved {saved_count} files to uploads directory")


def history_page():
    """Display processing history"""
    st.header("📊 Processing History")

    if DATABASE_AVAILABLE and get_db_session:
        db_session = get_db_session()
        try:
            # Get all invoices from database
            invoices = db_session.query(Invoice).order_by(Invoice.upload_date.desc()).all() if db_session else []

            if invoices:
                # Convert to DataFrame for display
                data = []
                for invoice in invoices:
                    data.append({
                        'ID': invoice.id,
                        'Filename': invoice.filename,
                        'Upload Date': invoice.upload_date.strftime('%Y-%m-%d %H:%M'),
                        'File Type': invoice.file_type,
                        'Status': invoice.processing_status,
                        'Invoice Number': invoice.invoice_number or 'Not extracted',
                        'Supplier': invoice.supplier_name or 'Not extracted',
                        'Total Amount': f"${invoice.total_amount:.2f}" if invoice.total_amount else 'Not extracted'
                    })

                df = pd.DataFrame(data)
                st.dataframe(df, use_container_width=True)

                # Summary statistics
                st.subheader("📈 Summary Statistics")
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Total Files", len(invoices))
                with col2:
                    processed = len([i for i in invoices if i.processing_status == 'processed'])
                    st.metric("Processed", processed)
                with col3:
                    pending = len([i for i in invoices if i.processing_status == 'pending'])
                    st.metric("Pending", pending)
                with col4:
                    errors = len([i for i in invoices if i.processing_status == 'error'])
                    st.metric("Errors", errors)
            else:
                st.info("No invoices processed yet. Go to 'Upload & Process' to get started!")

        except Exception as e:
            st.error(f"Database error: {e}")
        finally:
            if db_session:
                db_session.close()
    else:
        st.info("Database not available. History functionality requires database setup.")


def show_feedback_history():
    """Show historical corrections and feedback"""
    st.header("📊 Feedback & Corrections History")
    st.info("⚠️ Feedback functionality requires database and utils modules")


def show_learning_dashboard():
    """Show comprehensive learning and improvement dashboard"""
    st.header("🧠 AI Learning Dashboard")
    st.info("⚠️ Learning dashboard requires database and utils modules")


def stats_page():
    """Display model statistics"""
    st.header("🤖 Model Statistics")
    st.info("Model performance metrics will be available after implementing auto fine-tuning in Phase 6")

    # Placeholder content
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Current Model Performance")
        st.metric("Overall Accuracy", "Coming soon")
        st.metric("Invoice Number Accuracy", "Coming soon")
        st.metric("Date Extraction Accuracy", "Coming soon")
        st.metric("Amount Extraction Accuracy", "Coming soon")

    with col2:
        st.subheader("Training Progress")
        st.metric("Training Samples", "Coming soon")
        st.metric("Model Version", "Coming soon")
        st.metric("Last Retrain Date", "Coming soon")


def settings_page():
    """Application settings"""
    st.header("⚙️ Settings")

    st.subheader("Database Settings")
    if st.button("🔄 Reset Database"):
        if st.checkbox("I understand this will delete all data"):
            # Reset database (placeholder)
            st.warning("Database reset functionality will be implemented in later phases")

    st.subheader("Model Settings")
    confidence_threshold = st.slider(
        "Minimum Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        help="Fields with confidence below this threshold will be flagged for review"
    )

    st.subheader("Upload Settings")
    max_file_size = st.number_input(
        "Maximum File Size (MB)",
        min_value=1,
        max_value=100,
        value=10
    )

    if st.button("💾 Save Settings"):
        st.success("Settings saved! (Note: Settings persistence will be implemented in later phases)")


if __name__ == "__main__":
    main()