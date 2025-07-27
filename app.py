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
    """Enhanced upload page with full-page results display"""
    st.header("📤 Upload Invoices")

    # Upload section (top of page)
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("File Upload")

        uploaded_files = st.file_uploader(
            "Choose invoice files",
            type=['pdf', 'png', 'jpg', 'jpeg', 'csv'],
            accept_multiple_files=True,
            help="Supported formats: PDF, PNG, JPG, CSV"
        )

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

                # Learning statistics
                try:
                    learning_system = LearningSystem()
                    learning_stats = learning_system.get_field_statistics()
                    accuracy = learning_stats.get('accuracy_rate', 0.0) * 100
                except Exception as e:
                    accuracy = 0.0
                    logger.warning(f"Could not load learning statistics: {e}")

                st.metric("Total Invoices", total_invoices)
                st.metric("Processed Today", processed_today)
                st.metric("AI Accuracy", f"{accuracy:.1f}%")

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

    # Process uploaded files (compact section)
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
                        process_file_with_full_display(uploaded_file)

        # Bulk actions
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 2])

        with col1:
            if st.button("🔄 Process All", type="primary"):
                process_all_files(uploaded_files)

        with col2:
            if st.button("💾 Save All"):
                save_all_files(uploaded_files)

    # FULL-PAGE RESULTS SECTION (separate from upload area)
    st.markdown("---")

    # Check if there's a processing result to display
    if 'current_processing_result' in st.session_state:
        result = st.session_state.current_processing_result

        st.markdown("## 🎯 Processing Results")
        display_full_processing_results(
            result['invoice_id'],
            result['filename'],
            result['extracted_fields'],
            result['full_text'],
            result['processing_time'],
            result['confidence'],
            result['pages']
        )
    else:
        # Show placeholder when no results
        st.markdown("## 📋 Results Area")
        st.info("📤 Upload and process an invoice above to see extraction results here.")

        # Show recent processing history
        show_recent_extractions()


def process_file_with_full_display(uploaded_file):
    """Process file with enhanced learning system and store results for full-page display"""

    if not UTILS_AVAILABLE:
        st.error("❌ Utils modules not available - cannot process files")
        return

    if not DATABASE_AVAILABLE:
        st.error("❌ Database not available - cannot save results")
        return

    # Validate file first
    file_handler = FileHandler()
    validation = file_handler.validate_file(uploaded_file)

    if not validation['valid']:
        st.error(f"❌ {validation['error']}")
        return

    with st.spinner(f"Processing {uploaded_file.name}..."):
        start_time = time.time()

        try:
            # Step 1: OCR Processing
            st.info("🔍 Step 1: Extracting text with OCR...")

            tesseract_path = get_tesseract_path()
            ocr_processor = OCRProcessor(tesseract_path)
            file_content = file_handler.get_file_content(uploaded_file)
            file_type = validation['file_type']

            ocr_result = ocr_processor.process_file(
                file_content, file_type, uploaded_file.name
            )

            if not ocr_result['success']:
                st.error(f"❌ OCR failed: {ocr_result.get('error', 'Unknown error')}")
                return

            # Step 2: Enhanced Field Extraction with Learning
            st.info("🧠 Step 2: Applying AI learning and extracting fields...")

            # Initialize learning system and apply learned patterns
            learning_system = LearningSystem()
            base_field_extractor = FieldExtractor()

            # Apply previous learning patterns before extraction
            enhanced_extractor = learning_system.apply_learned_patterns(
                base_field_extractor, uploaded_file.name
            )

            # Extract fields using enhanced extractor
            extracted_fields = enhanced_extractor.extract_all_fields(ocr_result['text'])

            processing_time = time.time() - start_time

            # Step 3: Save to Database
            st.info("💾 Step 3: Saving results to database...")

            db_session = get_db_session()
            if not db_session:
                st.error("❌ Database session could not be created")
                return

            try:
                # Create invoice record
                new_invoice = Invoice(
                    filename=uploaded_file.name,
                    file_type=file_type,
                    processing_status='processed',
                    raw_text=ocr_result['text'][:10000],  # Limit text length
                    invoice_number=extracted_fields.get('invoice_number', {}).get('value', ''),
                    invoice_date=extracted_fields.get('date', {}).get('value', ''),
                    supplier_name=extracted_fields.get('supplier', {}).get('value', ''),
                    total_amount=extracted_fields.get('total', {}).get('value', 0.0),
                    vat_amount=extracted_fields.get('vat', {}).get('value', 0.0),
                    confidence_invoice_number=extracted_fields.get('invoice_number', {}).get('confidence', 0.0),
                    confidence_date=extracted_fields.get('date', {}).get('confidence', 0.0),
                    confidence_supplier=extracted_fields.get('supplier', {}).get('confidence', 0.0),
                    confidence_total=extracted_fields.get('total', {}).get('confidence', 0.0)
                )

                db_session.add(new_invoice)
                db_session.flush()  # Get the ID

                # Create OCR result record
                ocr_record = OCRResult(
                    invoice_id=new_invoice.id,
                    extracted_text=ocr_result['text'],
                    confidence_score=ocr_result['confidence'],
                    processing_time=processing_time,
                    ocr_method='tesseract_enhanced',
                    pages_processed=ocr_result.get('pages', 1)
                )

                db_session.add(ocr_record)
                db_session.commit()

                # Store in session state for full-page display
                st.session_state.current_processing_result = {
                    'invoice_id': new_invoice.id,
                    'filename': uploaded_file.name,
                    'extracted_fields': extracted_fields,
                    'full_text': ocr_result['text'],
                    'processing_time': processing_time,
                    'confidence': ocr_result['confidence'],
                    'pages': ocr_result.get('pages', 1)
                }

                # Show success message with learning info
                st.success(f"✅ {uploaded_file.name} processed successfully with AI learning!")

                # Quick stats with learning enhancement info
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Processing Time", f"{processing_time:.1f}s")
                with col2:
                    st.metric("OCR Confidence", f"{ocr_result['confidence']:.0%}")
                with col3:
                    st.metric("Pages", ocr_result.get('pages', 1))
                with col4:
                    # Show if learning patterns were applied
                    patterns_applied = len(learning_system.learned_patterns)
                    st.metric("AI Patterns", f"{patterns_applied}")

                # Show learning enhancement info
                if patterns_applied > 0:
                    st.info(
                        f"🧠 **AI Enhancement Applied:** Used {patterns_applied} learned patterns from previous corrections to improve accuracy!")
                else:
                    st.info("🌟 **First Invoice:** This will help train the AI for future invoices!")

                # Redirect message
                st.info(
                    "📋 **Scroll down to review and correct the extracted fields. Your corrections will immediately improve the AI!**")

                # Show brief learning stats
                try:
                    stats = learning_system.get_field_statistics()
                    if stats['total_extractions'] > 0:
                        with st.expander("📊 Current AI Learning Progress"):
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Invoices", stats['total_extractions'])
                            with col2:
                                st.metric("AI Accuracy", f"{stats['accuracy_rate'] * 100:.1f}%")
                            with col3:
                                st.metric("User Corrections", stats['total_corrections'])
                except Exception as e:
                    logger.warning(f"Could not display learning stats: {e}")

            except Exception as e:
                st.error(f"❌ Database error: {e}")
                if db_session:
                    db_session.rollback()
            finally:
                if db_session:
                    db_session.close()

        except Exception as e:
            st.error(f"❌ Processing error: {e}")
            logger.error(f"Error processing {uploaded_file.name}: {traceback.format_exc()}")

def display_full_processing_results(invoice_id, filename, extracted_fields, full_text, processing_time, confidence,
                                    pages):
    """Display the processing results with live editing and learning capabilities"""

    # Header with file info
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown(f"### 📄 {filename}")
    with col2:
        st.metric("Processing Time", f"{processing_time:.1f}s")
    with col3:
        st.metric("OCR Confidence", f"{confidence:.0%}")

    # Create tabs for better organization
    tab1, tab2, tab3 = st.tabs(["📝 Review & Correct Fields", "📊 Confidence Analysis", "📄 Raw Text"])

    with tab1:
        st.subheader("Extracted Information - Review and Correct")
        st.info(
            "💡 Edit any incorrect values below. Your corrections will immediately improve the AI for future invoices!")

        # Create the correction form that shows current values
        with st.form(key=f"live_corrections_form_{invoice_id}"):

            # Create two columns for better layout
            col1, col2 = st.columns(2)

            corrected_fields = {}

            with col1:
                st.markdown("#### 📄 Document Details")

                # Invoice Number
                inv_data = extracted_fields.get('invoice_number', {})
                confidence = inv_data.get('confidence', 0)
                color = "🟢" if confidence > 0.8 else "🟡" if confidence > 0.5 else "🔴"

                st.markdown(f"{color} **Invoice Number** (Confidence: {confidence:.0%})")
                corrected_fields['invoice_number'] = st.text_input(
                    "Invoice Number:",
                    value=str(inv_data.get('value', '')),
                    key=f"inv_num_{invoice_id}",
                    help=f"AI extracted: '{inv_data.get('value', 'Not found')}' with {confidence:.0%} confidence"
                )

                # Date
                date_data = extracted_fields.get('date', {})
                confidence = date_data.get('confidence', 0)
                color = "🟢" if confidence > 0.8 else "🟡" if confidence > 0.5 else "🔴"

                st.markdown(f"{color} **Invoice Date** (Confidence: {confidence:.0%})")
                corrected_fields['date'] = st.text_input(
                    "Invoice Date:",
                    value=str(date_data.get('value', '')),
                    key=f"date_{invoice_id}",
                    help=f"AI extracted: '{date_data.get('value', 'Not found')}' with {confidence:.0%} confidence"
                )

                # Supplier
                supplier_data = extracted_fields.get('supplier', {})
                confidence = supplier_data.get('confidence', 0)
                color = "🟢" if confidence > 0.8 else "🟡" if confidence > 0.5 else "🔴"

                st.markdown(f"{color} **Supplier Name** (Confidence: {confidence:.0%})")
                corrected_fields['supplier'] = st.text_input(
                    "Supplier Name:",
                    value=str(supplier_data.get('value', '')),
                    key=f"supplier_{invoice_id}",
                    help=f"AI extracted: '{supplier_data.get('value', 'Not found')}' with {confidence:.0%} confidence"
                )

                # Customer
                customer_data = extracted_fields.get('customer', {})
                confidence = customer_data.get('confidence', 0)
                color = "🟢" if confidence > 0.8 else "🟡" if confidence > 0.5 else "🔴"

                st.markdown(f"{color} **Customer Name** (Confidence: {confidence:.0%})")
                corrected_fields['customer'] = st.text_input(
                    "Customer Name:",
                    value=str(customer_data.get('value', '')),
                    key=f"customer_{invoice_id}",
                    help=f"AI extracted: '{customer_data.get('value', 'Not found')}' with {confidence:.0%} confidence"
                )

            with col2:
                st.markdown("#### 💰 Financial Details")

                # Total Amount
                total_data = extracted_fields.get('total', {})
                confidence = total_data.get('confidence', 0)
                color = "🟢" if confidence > 0.8 else "🟡" if confidence > 0.5 else "🔴"

                st.markdown(f"{color} **Total Amount** (Confidence: {confidence:.0%})")
                corrected_fields['total'] = st.number_input(
                    "Total Amount ($):",
                    value=float(total_data.get('value', 0.0)),
                    key=f"total_{invoice_id}",
                    help=f"AI extracted: ${total_data.get('value', 0):.2f} with {confidence:.0%} confidence",
                    format="%.2f",
                    min_value=0.0
                )

                # Subtotal
                subtotal_data = extracted_fields.get('subtotal', {})
                confidence = subtotal_data.get('confidence', 0)
                color = "🟢" if confidence > 0.8 else "🟡" if confidence > 0.5 else "🔴"

                st.markdown(f"{color} **Subtotal** (Confidence: {confidence:.0%})")
                corrected_fields['subtotal'] = st.number_input(
                    "Subtotal ($):",
                    value=float(subtotal_data.get('value', 0.0)),
                    key=f"subtotal_{invoice_id}",
                    help=f"AI extracted: ${subtotal_data.get('value', 0):.2f} with {confidence:.0%} confidence",
                    format="%.2f",
                    min_value=0.0
                )

                # VAT Amount
                vat_data = extracted_fields.get('vat', {})
                confidence = vat_data.get('confidence', 0)
                color = "🟢" if confidence > 0.8 else "🟡" if confidence > 0.5 else "🔴"

                st.markdown(f"{color} **VAT Amount** (Confidence: {confidence:.0%})")
                corrected_fields['vat'] = st.number_input(
                    "VAT Amount ($):",
                    value=float(vat_data.get('value', 0.0)),
                    key=f"vat_{invoice_id}",
                    help=f"AI extracted: ${vat_data.get('value', 0):.2f} with {confidence:.0%} confidence",
                    format="%.2f",
                    min_value=0.0
                )

                # Currency
                st.markdown("🟡 **Currency** (Manual Entry)")
                corrected_fields['currency'] = st.selectbox(
                    "Currency:",
                    options=["USD", "EUR", "GBP", "CAD", "AUD", "JPY"],
                    index=0,
                    key=f"currency_{invoice_id}",
                    help="Currency detection will be improved in future versions"
                )

            # Line Items Section (Full Width)
            st.markdown("---")
            st.markdown("#### 📋 Line Items Details")

            col3, col4 = st.columns(2)

            with col3:
                # Line Items Count
                line_count_data = extracted_fields.get('line_items_count', {})
                confidence = line_count_data.get('confidence', 0)
                color = "🟢" if confidence > 0.8 else "🟡" if confidence > 0.5 else "🔴"

                st.markdown(f"{color} **Line Items Count** (Confidence: {confidence:.0%})")
                corrected_fields['line_items_count'] = st.number_input(
                    "Number of Line Items:",
                    value=int(line_count_data.get('value', 0)),
                    min_value=0,
                    max_value=100,
                    key=f"line_count_{invoice_id}",
                    help="Count the number of individual items/services listed"
                )

            with col4:
                # Line Items Subtotal
                line_subtotal_data = extracted_fields.get('line_items_subtotal', {})
                confidence = line_subtotal_data.get('confidence', 0)
                color = "🟢" if confidence > 0.8 else "🟡" if confidence > 0.5 else "🔴"

                st.markdown(f"{color} **Line Items Subtotal** (Confidence: {confidence:.0%})")
                corrected_fields['line_items_subtotal'] = st.number_input(
                    "Line Items Subtotal ($):",
                    value=float(line_subtotal_data.get('value', 0.0)),
                    key=f"line_subtotal_{invoice_id}",
                    help="Sum of all line item amounts before tax",
                    format="%.2f",
                    min_value=0.0
                )

            # Overall confidence display
            overall_confidence = calculate_field_confidence_score(extracted_fields)

            st.markdown("---")
            col1, col2, col3 = st.columns([1, 1, 2])

            with col1:
                st.metric("Overall AI Confidence", f"{overall_confidence:.1%}")

            with col2:
                if overall_confidence > 0.8:
                    st.success("🎯 High Confidence")
                elif overall_confidence > 0.6:
                    st.warning("⚠️ Medium Confidence")
                else:
                    st.error("❌ Low Confidence")

            with col3:
                st.info("💡 Green = High confidence (>80%), Yellow = Medium (50-80%), Red = Low (<50%)")

            # Submit buttons section
            st.markdown("---")

            col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])

            with col_btn1:
                submitted = st.form_submit_button(
                    "🧠 Save Corrections & Train AI",
                    type="primary",
                    use_container_width=True,
                    help="Save your corrections and immediately improve the AI"
                )

            with col_btn2:
                approved = st.form_submit_button(
                    "✅ All Correct - Confirm",
                    use_container_width=True,
                    help="Confirm all extractions are correct"
                )

            with col_btn3:
                st.markdown(
                    "**🔄 Real-time Learning:** Your corrections immediately improve accuracy for future invoices!")

            # Handle form submissions
            if submitted:
                handle_user_corrections_with_learning(invoice_id, extracted_fields, corrected_fields, filename)

            if approved:
                handle_user_approval_with_learning(invoice_id, extracted_fields, filename)

    with tab2:
        st.subheader("📊 Confidence Analysis & Learning Progress")

        # Show current extraction confidence
        col1, col2 = st.columns([1, 1])

        with col1:
            if PLOTLY_AVAILABLE:
                create_confidence_chart(extracted_fields)
            else:
                st.warning("Plotly not available - showing text summary")
                show_confidence_text_summary(extracted_fields)

        with col2:
            # Show learning statistics
            st.markdown("#### 🧠 AI Learning Progress")

            try:
                learning_system = LearningSystem()
                learning_stats = learning_system.get_field_statistics()

                st.metric("Total Invoices Processed", learning_stats.get('total_extractions', 0))
                st.metric("User Corrections Made", learning_stats.get('total_corrections', 0))
                st.metric("Current AI Accuracy", f"{learning_stats.get('accuracy_rate', 0) * 100:.1f}%")

                # Show most improved fields
                patterns = learning_system.get_learning_patterns()
                if 'field_accuracy' in patterns and patterns['field_accuracy']:
                    st.markdown("**📈 Field Accuracy:**")
                    for field, accuracy in patterns['field_accuracy'].items():
                        color = "🟢" if accuracy > 0.8 else "🟡" if accuracy > 0.6 else "🔴"
                        st.write(f"{color} {field.title()}: {accuracy:.0%}")

            except Exception as e:
                st.warning(f"Learning statistics temporarily unavailable: {e}")

    with tab3:
        st.subheader("📄 Raw Extracted Text")

        # Text statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Characters", len(full_text))
        with col2:
            word_count = len(full_text.split())
            st.metric("Words", word_count)
        with col3:
            line_count = len(full_text.split('\n'))
            st.metric("Lines", line_count)
        with col4:
            reading_time = max(1, word_count // 200)
            st.metric("Est. Reading Time", f"{reading_time} min")

        # Display text in chunks if too long
        if len(full_text) > 3000:
            st.info("📄 Text is long - showing in expandable sections")
            chunk_size = 1500
            chunks = [full_text[i:i + chunk_size] for i in range(0, len(full_text), chunk_size)]

            for i, chunk in enumerate(chunks):
                with st.expander(f"📝 Text Section {i + 1} of {len(chunks)}"):
                    st.text_area(
                        f"Section {i + 1}:",
                        value=chunk,
                        height=200,
                        disabled=True,
                        key=f"text_chunk_{invoice_id}_{i}"
                    )
        else:
            st.text_area("Full OCR Text:", full_text, height=400, disabled=True)

        # Download option
        st.download_button(
            label="📥 Download Extracted Text",
            data=full_text,
            file_name=f"extracted_text_{filename}.txt",
            mime="text/plain"
        )


def handle_user_corrections_with_learning(invoice_id, original_fields, corrected_fields, filename):
    """Handle user corrections with advanced learning system"""

    try:
        learning_system = LearningSystem()

        # Count and analyze corrections
        corrections_made = 0
        correction_details = []

        for field_name in original_fields.keys():
            original_value = str(original_fields[field_name]['value']).strip()
            corrected_value = str(corrected_fields.get(field_name, original_value)).strip()

            if original_value != corrected_value:
                corrections_made += 1
                correction_details.append({
                    'field': field_name,
                    'original': original_value,
                    'corrected': corrected_value,
                    'confidence': original_fields[field_name]['confidence']
                })

        if corrections_made > 0:
            # Save corrections to database and update learning patterns
            success = learning_system.save_field_corrections(
                invoice_id, original_fields, corrected_fields
            )

            if success:
                # Show detailed feedback about what was learned
                st.success(f"🧠 **AI Training Complete!** Made {corrections_made} corrections")

                # Show learning impact
                with st.expander("📊 See Learning Impact", expanded=True):

                    # Show what was corrected
                    st.markdown("**✏️ Corrections Made:**")
                    for detail in correction_details:
                        st.write(f"• **{detail['field'].title()}:** '{detail['original']}' → '{detail['corrected']}'")

                    # Show updated statistics
                    try:
                        updated_stats = learning_system.get_field_statistics()

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Invoices", updated_stats.get('total_extractions', 0))
                        with col2:
                            st.metric("Total Corrections", updated_stats.get('total_corrections', 0))
                        with col3:
                            accuracy = updated_stats.get('accuracy_rate', 0.0) * 100
                            st.metric("Updated Accuracy", f"{accuracy:.1f}%")

                        # Show most problematic fields
                        problematic = updated_stats.get('most_problematic_fields', [])
                        if problematic:
                            st.markdown("**🎯 Fields Being Improved:**")
                            for field_name, error_count in problematic[:3]:
                                st.write(f"• {field_name.title()}: {error_count} corrections needed")

                        st.success("🚀 The AI will now be more accurate for these fields in future invoices!")

                    except Exception as e:
                        st.warning(f"Could not load updated statistics: {e}")
            else:
                st.error("❌ Failed to save corrections. Please try again.")
        else:
            st.info("ℹ️ No corrections detected. All extractions confirmed as accurate!")
            # Still save as positive feedback
            handle_user_approval_with_learning(invoice_id, original_fields, filename)

    except Exception as e:
        st.error(f"❌ Error processing corrections: {e}")
        logger.error(f"Error in handle_user_corrections_with_learning: {e}")


def handle_user_approval_with_learning(invoice_id, extracted_fields, filename):
    """Handle user approval (all extractions correct) with learning"""

    try:
        learning_system = LearningSystem()

        # Create dummy corrected_fields that match original (no changes)
        corrected_fields = {}
        for field_name, field_data in extracted_fields.items():
            corrected_fields[field_name] = field_data['value']

        # Save as confirmation feedback
        success = learning_system.save_field_corrections(
            invoice_id, extracted_fields, corrected_fields
        )

        if success:
            st.success("✅ **Excellent!** All extractions confirmed correct - AI confidence boosted!")

            with st.expander("📈 Positive Learning Impact"):
                st.write("**🎯 Confirmed Accurate Fields:**")
                for field_name, field_data in extracted_fields.items():
                    confidence = field_data['confidence']
                    value = field_data['value']
                    if confidence > 0 and value:  # Only show fields that had values
                        st.write(f"• **{field_name.title()}:** '{value}' ({confidence:.0%} confidence)")

                st.info("🧠 These confirmations help the AI understand when it's performing well!")
        else:
            st.error("❌ Failed to save confirmation. Please try again.")

    except Exception as e:
        st.error(f"❌ Error processing approval: {e}")
        logger.error(f"Error in handle_user_approval_with_learning: {e}")


def show_confidence_text_summary(extracted_fields):
    """Show confidence summary when Plotly is not available"""
    st.markdown("**📊 Field Confidence Scores:**")

    confidences = []
    for field_name, field_data in extracted_fields.items():
        if isinstance(field_data, dict) and 'confidence' in field_data:
            confidence = field_data['confidence']
            confidences.append((field_name, confidence))

    # Sort by confidence (lowest first to highlight problems)
    confidences.sort(key=lambda x: x[1])

    for field_name, confidence in confidences:
        color = "🟢" if confidence > 0.8 else "🟡" if confidence > 0.5 else "🔴"
        field_display = field_name.replace('_', ' ').title()
        st.write(f"{color} **{field_display}:** {confidence:.0%}")

    # Summary
    if confidences:
        avg_confidence = sum(c[1] for c in confidences) / len(confidences)
        st.metric("Average Confidence", f"{avg_confidence:.0%}")
def show_recent_extractions():
    """Show recent extractions for reference"""
    st.subheader("📚 Recent Extractions")

    if not DATABASE_AVAILABLE:
        st.info("Database not available")
        return

    db_session = get_db_session()
    if not db_session:
        return

    try:
        recent_invoices = db_session.query(Invoice).order_by(
            Invoice.upload_date.desc()
        ).limit(5).all()

        if recent_invoices:
            for invoice in recent_invoices:
                with st.expander(f"📄 {invoice.filename} - {invoice.upload_date.strftime('%Y-%m-%d %H:%M')}"):
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.write(f"**Invoice #:** {invoice.invoice_number or 'Not detected'}")
                        st.write(f"**Supplier:** {invoice.supplier_name or 'Not detected'}")
                    with col2:
                        st.write(f"**Date:** {invoice.invoice_date or 'Not detected'}")
                        st.write(f"**Total:** ${invoice.total_amount or 0:.2f}")
                    with col3:
                        st.write(f"**Status:** {invoice.processing_status}")
                        if st.button(f"View Details", key=f"view_{invoice.id}"):
                            load_invoice_results(invoice.id)
        else:
            st.info("No recent extractions found.")

    except Exception as e:
        st.error(f"Error loading recent extractions: {e}")
    finally:
        db_session.close()


def load_invoice_results(invoice_id):
    """Load and display results for a specific invoice"""
    if not DATABASE_AVAILABLE:
        return

    db_session = get_db_session()
    if not db_session:
        return

    try:
        invoice = db_session.query(Invoice).filter(Invoice.id == invoice_id).first()
        ocr_result = db_session.query(OCRResult).filter(OCRResult.invoice_id == invoice_id).first()

        if invoice and ocr_result:
            # Reconstruct extracted fields from database
            extracted_fields = {
                'invoice_number': {'value': invoice.invoice_number or '',
                                   'confidence': invoice.confidence_invoice_number or 0.0},
                'date': {'value': invoice.invoice_date or '', 'confidence': invoice.confidence_date or 0.0},
                'supplier': {'value': invoice.supplier_name or '', 'confidence': invoice.confidence_supplier or 0.0},
                'customer': {'value': 'Not stored', 'confidence': 0.0},
                'total': {'value': invoice.total_amount or 0.0, 'confidence': invoice.confidence_total or 0.0},
                'subtotal': {'value': invoice.total_amount or 0.0, 'confidence': 0.8},
                'vat': {'value': invoice.vat_amount or 0.0, 'confidence': 0.8},
                'line_items_count': {'value': 0, 'confidence': 0.0},
                'line_items_subtotal': {'value': 0.0, 'confidence': 0.0}
            }

            # Update session state to show results
            st.session_state.current_processing_result = {
                'invoice_id': invoice.id,
                'filename': invoice.filename,
                'extracted_fields': extracted_fields,
                'full_text': ocr_result.extracted_text,
                'processing_time': ocr_result.processing_time,
                'confidence': ocr_result.confidence_score,
                'pages': ocr_result.pages_processed
            }

            st.rerun()  # Refresh page to show results

    except Exception as e:
        st.error(f"Error loading invoice results: {e}")
    finally:
        db_session.close()


def create_confidence_chart(extracted_fields):
    """Create confidence visualization chart"""
    if not PLOTLY_AVAILABLE:
        return

    # Prepare data for chart
    field_names = []
    confidences = []
    colors = []

    field_display_names = {
        'invoice_number': 'Invoice Number',
        'date': 'Date',
        'supplier': 'Supplier',
        'customer': 'Customer',
        'total': 'Total',
        'subtotal': 'Subtotal',
        'vat': 'VAT',
        'line_items_count': 'Line Items Count',
        'line_items_subtotal': 'Line Items Subtotal'
    }

    for field_name, field_data in extracted_fields.items():
        if isinstance(field_data, dict) and 'confidence' in field_data:
            display_name = field_display_names.get(field_name, field_name.title())
            confidence = field_data['confidence']

            field_names.append(display_name)
            confidences.append(confidence)

            # Color based on confidence level
            if confidence > 0.8:
                colors.append('#28a745')  # Green
            elif confidence > 0.5:
                colors.append('#ffc107')  # Yellow
            else:
                colors.append('#dc3545')  # Red

    # Create bar chart using Plotly
    fig = go.Figure(data=[
        go.Bar(
            x=field_names,
            y=confidences,
            marker_color=colors,
            text=[f"{c:.0%}" for c in confidences],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Confidence: %{y:.0%}<extra></extra>'
        )
    ])

    fig.update_layout(
        title="Field Extraction Confidence Scores",
        xaxis_title="Fields",
        yaxis_title="Confidence",
        yaxis=dict(range=[0, 1.0], tickformat='.0%'),
        height=400,
        showlegend=False,
        margin=dict(l=40, r=40, t=60, b=40)
    )

    fig.update_xaxes(tickangle=45)

    st.plotly_chart(fig, use_container_width=True)

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

    if not UTILS_AVAILABLE:
        st.error("❌ Utils modules not available - cannot process files")
        return

    if not DATABASE_AVAILABLE:
        st.error("❌ Database not available - cannot save results")
        return

    # Validate file first
    file_handler = FileHandler()
    validation = file_handler.validate_file(uploaded_file)

    if not validation['valid']:
        st.error(f"❌ {validation['error']}")
        return

    with st.spinner(f"Processing {uploaded_file.name}..."):
        start_time = time.time()

        try:
            # Step 1: OCR Processing
            st.info("🔍 Step 1: Extracting text with OCR...")

            tesseract_path = get_tesseract_path()
            ocr_processor = OCRProcessor(tesseract_path)
            file_content = file_handler.get_file_content(uploaded_file)
            file_type = validation['file_type']

            ocr_result = ocr_processor.process_file(
                file_content, file_type, uploaded_file.name
            )

            if not ocr_result['success']:
                st.error(f"❌ OCR failed: {ocr_result.get('error', 'Unknown error')}")
                return

            # Step 2: Field Extraction
            st.info("🎯 Step 2: Extracting invoice fields...")

            field_extractor = FieldExtractor()
            extracted_fields = field_extractor.extract_all_fields(ocr_result['text'])

            processing_time = time.time() - start_time

            # Step 3: Save to Database
            st.info("💾 Step 3: Saving to database...")

            db_session = get_db_session()
            if not db_session:
                st.error("❌ Database session could not be created")
                return

            try:
                # Create invoice record
                new_invoice = Invoice(
                    filename=uploaded_file.name,
                    file_type=file_type,
                    processing_status='processed',
                    raw_text=ocr_result['text'][:10000],  # Limit text length
                    invoice_number=extracted_fields.get('invoice_number', {}).get('value', ''),
                    invoice_date=extracted_fields.get('date', {}).get('value', ''),
                    supplier_name=extracted_fields.get('supplier', {}).get('value', ''),
                    total_amount=extracted_fields.get('total', {}).get('value', 0.0),
                    vat_amount=extracted_fields.get('vat', {}).get('value', 0.0),
                    confidence_invoice_number=extracted_fields.get('invoice_number', {}).get('confidence', 0.0),
                    confidence_date=extracted_fields.get('date', {}).get('confidence', 0.0),
                    confidence_supplier=extracted_fields.get('supplier', {}).get('confidence', 0.0),
                    confidence_total=extracted_fields.get('total', {}).get('confidence', 0.0)
                )

                db_session.add(new_invoice)
                db_session.flush()  # Get the ID

                # Create OCR result record
                ocr_record = OCRResult(
                    invoice_id=new_invoice.id,
                    extracted_text=ocr_result['text'],
                    confidence_score=ocr_result['confidence'],
                    processing_time=processing_time,
                    ocr_method='tesseract',
                    pages_processed=ocr_result.get('pages', 1)
                )

                db_session.add(ocr_record)
                db_session.commit()

                # Step 4: Display Results
                st.success(f"✅ {uploaded_file.name} processed successfully!")

                # Show processing stats
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Processing Time", f"{processing_time:.1f}s")
                with col2:
                    st.metric("OCR Confidence", f"{ocr_result['confidence']:.0%}")
                with col3:
                    st.metric("Pages", ocr_result.get('pages', 1))

                # Display extracted fields in a nice format
                st.subheader("📋 Extracted Information")

                # Create two columns for extracted fields
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("#### 📄 Document Details")

                    # Invoice Number
                    inv_data = extracted_fields.get('invoice_number', {})
                    confidence = inv_data.get('confidence', 0)
                    color = "🟢" if confidence > 0.8 else "🟡" if confidence > 0.5 else "🔴"
                    st.write(f"{color} **Invoice Number:** {inv_data.get('value', 'Not found')} ({confidence:.0%})")

                    # Date
                    date_data = extracted_fields.get('date', {})
                    confidence = date_data.get('confidence', 0)
                    color = "🟢" if confidence > 0.8 else "🟡" if confidence > 0.5 else "🔴"
                    st.write(f"{color} **Date:** {date_data.get('value', 'Not found')} ({confidence:.0%})")

                    # Supplier
                    supplier_data = extracted_fields.get('supplier', {})
                    confidence = supplier_data.get('confidence', 0)
                    color = "🟢" if confidence > 0.8 else "🟡" if confidence > 0.5 else "🔴"
                    st.write(f"{color} **Supplier:** {supplier_data.get('value', 'Not found')} ({confidence:.0%})")

                    # Customer
                    customer_data = extracted_fields.get('customer', {})
                    confidence = customer_data.get('confidence', 0)
                    color = "🟢" if confidence > 0.8 else "🟡" if confidence > 0.5 else "🔴"
                    st.write(f"{color} **Customer:** {customer_data.get('value', 'Not found')} ({confidence:.0%})")

                with col2:
                    st.markdown("#### 💰 Financial Details")

                    # Total
                    total_data = extracted_fields.get('total', {})
                    confidence = total_data.get('confidence', 0)
                    color = "🟢" if confidence > 0.8 else "🟡" if confidence > 0.5 else "🔴"
                    st.write(f"{color} **Total:** ${total_data.get('value', 0):.2f} ({confidence:.0%})")

                    # Subtotal
                    subtotal_data = extracted_fields.get('subtotal', {})
                    confidence = subtotal_data.get('confidence', 0)
                    color = "🟢" if confidence > 0.8 else "🟡" if confidence > 0.5 else "🔴"
                    st.write(f"{color} **Subtotal:** ${subtotal_data.get('value', 0):.2f} ({confidence:.0%})")

                    # VAT
                    vat_data = extracted_fields.get('vat', {})
                    confidence = vat_data.get('confidence', 0)
                    color = "🟢" if confidence > 0.8 else "🟡" if confidence > 0.5 else "🔴"
                    st.write(f"{color} **VAT:** ${vat_data.get('value', 0):.2f} ({confidence:.0%})")

                # Overall confidence
                overall_confidence = calculate_field_confidence_score(extracted_fields)
                st.markdown("---")
                st.metric("Overall Extraction Confidence", f"{overall_confidence:.1%}")

                if overall_confidence > 0.8:
                    st.success("🎯 High confidence extraction - likely accurate!")
                elif overall_confidence > 0.6:
                    st.warning("⚠️ Medium confidence - please review results")
                else:
                    st.error("❌ Low confidence - manual review recommended")

                # Show extracted text
                with st.expander("📄 View Full Extracted Text"):
                    st.text_area("OCR Text:", ocr_result['text'], height=300, disabled=True)

                # Quick correction form
                st.markdown("---")
                st.subheader("✏️ Quick Corrections (Optional)")

                with st.form(f"corrections_{new_invoice.id}"):
                    st.write("Correct any errors to help improve the AI:")

                    col1, col2 = st.columns(2)

                    with col1:
                        corrected_invoice_number = st.text_input("Invoice Number:",
                                                                 value=str(inv_data.get('value', '')))
                        corrected_supplier = st.text_input("Supplier:", value=str(supplier_data.get('value', '')))

                    with col2:
                        corrected_total = st.number_input("Total Amount:", value=float(total_data.get('value', 0)),
                                                          format="%.2f")
                        corrected_date = st.text_input("Date:", value=str(date_data.get('value', '')))

                    if st.form_submit_button("💾 Save Corrections"):
                        # Save corrections (simplified for now)
                        st.success("✅ Thank you! Your corrections will help improve the AI.")
                        st.info("🧠 Full learning system will be implemented in the next phase.")

            except Exception as e:
                st.error(f"❌ Database error: {e}")
                if db_session:
                    db_session.rollback()
            finally:
                if db_session:
                    db_session.close()

        except Exception as e:
            st.error(f"❌ Processing error: {e}")
            logger.error(f"Error processing {uploaded_file.name}: {traceback.format_exc()}")
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
    """Show comprehensive feedback and corrections history with analytics"""

    st.header("📊 Feedback & Corrections History")
    st.markdown("Track how the AI is learning and improving from user feedback")

    if not DATABASE_AVAILABLE:
        st.error("❌ Database not available - cannot load feedback history")
        return

    db_session = get_db_session()
    if not db_session:
        st.error("❌ Cannot connect to database")
        return

    try:
        # Get all feedback and extractions
        all_feedback = db_session.query(UserFeedback).order_by(UserFeedback.feedback_date.desc()).all()
        all_extractions = db_session.query(FieldExtraction).order_by(FieldExtraction.created_date.desc()).all()

        if not all_feedback and not all_extractions:
            st.info("📋 No feedback history yet. Process some invoices and make corrections to see data here!")
            st.markdown("""
            **How to generate feedback data:**
            1. Go to "Upload & Process" 
            2. Upload and process an invoice
            3. Make corrections in the "Review & Correct Fields" section
            4. Return here to see your feedback history
            """)
            return

        # Summary Statistics
        st.subheader("📈 Learning Summary")

        corrections = [f for f in all_feedback if f.feedback_type == 'correction']
        confirmations = [f for f in all_feedback if f.feedback_type == 'confirmation']

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Feedback Items", len(all_feedback))
        with col2:
            st.metric("Corrections Made", len(corrections))
        with col3:
            st.metric("Confirmations", len(confirmations))
        with col4:
            if len(all_feedback) > 0:
                accuracy_rate = len(confirmations) / len(all_feedback) * 100
                st.metric("Accuracy Rate", f"{accuracy_rate:.1f}%")
            else:
                st.metric("Accuracy Rate", "N/A")

        # Learning Progress Over Time
        if all_extractions:
            st.subheader("📊 Learning Progress Over Time")

            try:
                # Create timeline data
                timeline_data = []
                for extraction in reversed(all_extractions):  # Oldest first for timeline
                    timeline_data.append({
                        'Date': extraction.created_date.strftime('%Y-%m-%d'),
                        'Invoice': extraction.invoice_id,
                        'Corrections': extraction.correction_count,
                        'Accuracy': max(0, 100 - (extraction.correction_count * 12.5))  # Rough accuracy estimate
                    })

                if len(timeline_data) > 1:
                    # Convert to DataFrame for plotting
                    import pandas as pd
                    df_timeline = pd.DataFrame(timeline_data)

                    if PLOTLY_AVAILABLE:
                        # Create accuracy trend chart
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=df_timeline['Date'],
                            y=df_timeline['Accuracy'],
                            mode='lines+markers',
                            name='AI Accuracy',
                            line=dict(color='#1f77b4', width=3),
                            marker=dict(size=8)
                        ))

                        fig.update_layout(
                            title="AI Accuracy Improvement Over Time",
                            xaxis_title="Date",
                            yaxis_title="Estimated Accuracy (%)",
                            yaxis=dict(range=[0, 100]),
                            height=400
                        )

                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.dataframe(df_timeline, use_container_width=True)

            except Exception as e:
                st.warning(f"Could not generate timeline chart: {e}")

        # Field-Specific Accuracy Analysis
        st.subheader("🎯 Field-Specific Learning Analysis")

        if corrections:
            # Analyze corrections by field
            field_analysis = {}
            for correction in corrections:
                field_name = correction.field_name
                if field_name not in field_analysis:
                    field_analysis[field_name] = {
                        'total_corrections': 0,
                        'low_confidence_corrections': 0,
                        'examples': []
                    }

                field_analysis[field_name]['total_corrections'] += 1

                if correction.confidence_before < 0.5:
                    field_analysis[field_name]['low_confidence_corrections'] += 1

                # Store example corrections (limit to 3 per field)
                if len(field_analysis[field_name]['examples']) < 3:
                    field_analysis[field_name]['examples'].append({
                        'original': correction.original_value,
                        'corrected': correction.corrected_value,
                        'confidence': correction.confidence_before,
                        'date': correction.feedback_date
                    })

            # Display field analysis
            for field_name, analysis in field_analysis.items():
                with st.expander(
                        f"📋 {field_name.replace('_', ' ').title()} - {analysis['total_corrections']} corrections"):

                    col1, col2 = st.columns([1, 2])

                    with col1:
                        st.metric("Total Corrections", analysis['total_corrections'])
                        st.metric("Low Confidence Issues", analysis['low_confidence_corrections'])

                        # Calculate improvement trend
                        if analysis['total_corrections'] > 2:
                            st.success("🔄 AI is learning this field!")
                        elif analysis['total_corrections'] > 5:
                            st.warning("⚠️ Field needs attention")

                    with col2:
                        st.markdown("**Recent correction examples:**")
                        for i, example in enumerate(analysis['examples'], 1):
                            confidence_color = "🟢" if example['confidence'] > 0.8 else "🟡" if example[
                                                                                                  'confidence'] > 0.5 else "🔴"
                            st.write(
                                f"{i}. {confidence_color} `'{example['original']}'` → `'{example['corrected']}'` ({example['confidence']:.0%})")

        # Recent Feedback Details
        st.subheader("📝 Recent Feedback Details")

        # Filter and pagination
        col1, col2, col3 = st.columns([1, 1, 2])

        with col1:
            feedback_filter = st.selectbox(
                "Filter by type:",
                ["All", "Corrections Only", "Confirmations Only"]
            )

        with col2:
            show_count = st.selectbox(
                "Show entries:",
                [10, 25, 50, 100],
                index=1
            )

        # Apply filters
        filtered_feedback = all_feedback
        if feedback_filter == "Corrections Only":
            filtered_feedback = [f for f in all_feedback if f.feedback_type == 'correction']
        elif feedback_filter == "Confirmations Only":
            filtered_feedback = [f for f in all_feedback if f.feedback_type == 'confirmation']

        # Show feedback entries
        if filtered_feedback:
            st.markdown(
                f"**Showing {min(show_count, len(filtered_feedback))} of {len(filtered_feedback)} feedback entries:**")

            for i, feedback in enumerate(filtered_feedback[:show_count]):
                # Determine feedback type styling
                if feedback.feedback_type == 'correction':
                    type_emoji = "✏️"
                    type_color = "orange"
                elif feedback.feedback_type == 'confirmation':
                    type_emoji = "✅"
                    type_color = "green"
                else:
                    type_emoji = "📝"
                    type_color = "blue"

                with st.expander(
                        f"{type_emoji} {feedback.feedback_type.title()} #{feedback.id} - {feedback.field_name.replace('_', ' ').title()} - {feedback.feedback_date.strftime('%Y-%m-%d %H:%M')}"):

                    col1, col2 = st.columns([1, 1])

                    with col1:
                        st.markdown("**Feedback Details:**")
                        st.write(f"**Field:** {feedback.field_name.replace('_', ' ').title()}")
                        st.write(f"**Type:** {feedback.feedback_type.title()}")
                        st.write(f"**Date:** {feedback.feedback_date.strftime('%Y-%m-%d %H:%M:%S')}")
                        st.write(f"**Confidence Before:** {feedback.confidence_before:.0%}")
                        if feedback.user_rating:
                            stars = "⭐" * feedback.user_rating
                            st.write(f"**User Rating:** {stars} ({feedback.user_rating}/5)")

                    with col2:
                        st.markdown("**Value Changes:**")

                        if feedback.feedback_type == 'correction':
                            st.markdown("**Original (AI):**")
                            st.code(feedback.original_value or "Empty", language=None)
                            st.markdown("**Corrected (User):**")
                            st.code(feedback.corrected_value or "Empty", language=None)
                        else:
                            st.markdown("**Confirmed Value:**")
                            st.code(feedback.original_value or "Empty", language=None)
                            st.success("✅ User confirmed this extraction was correct")

                    # Show learning impact
                    if feedback.is_used_for_training:
                        st.info("🧠 This feedback was used to improve the AI model")
                    else:
                        st.warning("⏳ This feedback is pending training integration")

        # Export functionality
        st.subheader("📥 Export Feedback Data")

        col1, col2 = st.columns([1, 3])

        with col1:
            if st.button("📊 Export to CSV"):
                # Create CSV data
                csv_data = []
                for feedback in all_feedback:
                    csv_data.append({
                        'ID': feedback.id,
                        'Invoice_ID': feedback.invoice_id,
                        'Field_Name': feedback.field_name,
                        'Feedback_Type': feedback.feedback_type,
                        'Original_Value': feedback.original_value,
                        'Corrected_Value': feedback.corrected_value,
                        'Confidence_Before': feedback.confidence_before,
                        'User_Rating': feedback.user_rating,
                        'Feedback_Date': feedback.feedback_date.isoformat(),
                        'Used_for_Training': feedback.is_used_for_training
                    })

                import pandas as pd
                df_export = pd.DataFrame(csv_data)
                csv_string = df_export.to_csv(index=False)

                st.download_button(
                    label="📥 Download CSV",
                    data=csv_string,
                    file_name=f"invoice_ai_feedback_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

        with col2:
            st.info("💡 Export your feedback data to analyze AI learning patterns or for backup purposes")

        # Learning Insights
        st.subheader("🔍 Learning Insights")

        try:
            learning_system = LearningSystem()
            learning_patterns = learning_system.get_learning_patterns()

            if 'common_mistakes' in learning_patterns and learning_patterns['common_mistakes']:
                st.markdown("**🎯 Most Common Correction Patterns:**")

                for i, mistake in enumerate(learning_patterns['common_mistakes'][:5], 1):
                    field_name = mistake['field'].replace('_', ' ').title()
                    st.write(
                        f"{i}. **{field_name}:** `'{mistake['wrong']}'` → `'{mistake['correct']}'` ({mistake['count']} times)")

                st.info("🧠 The AI has learned these patterns and will apply them to future invoices!")

            # Show field accuracy if available
            if 'field_accuracy' in learning_patterns and learning_patterns['field_accuracy']:
                st.markdown("**📊 Current Field Accuracy:**")

                accuracy_data = learning_patterns['field_accuracy']
                for field_name, accuracy in sorted(accuracy_data.items(), key=lambda x: x[1], reverse=True):
                    field_display = field_name.replace('_', ' ').title()
                    color = "🟢" if accuracy > 0.8 else "🟡" if accuracy > 0.6 else "🔴"
                    st.write(f"{color} **{field_display}:** {accuracy:.0%}")

        except Exception as e:
            st.warning(f"Could not load learning insights: {e}")

    except Exception as e:
        st.error(f"Error loading feedback history: {e}")
    finally:
        db_session.close()


def show_learning_dashboard():
    """Show comprehensive AI learning and improvement dashboard"""

    st.header("🧠 AI Learning Dashboard")
    st.markdown("Monitor how the AI is learning and improving from user feedback")

    if not DATABASE_AVAILABLE:
        st.error("❌ Database not available - cannot load learning data")
        return

    try:
        learning_system = LearningSystem()

        # Get comprehensive statistics
        stats = learning_system.get_field_statistics()
        patterns = learning_system.get_learning_patterns()

        # Overall Performance Overview
        st.subheader("📊 Overall AI Performance")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Invoices Processed", stats.get('total_extractions', 0))
        with col2:
            st.metric("User Corrections Made", stats.get('total_corrections', 0))
        with col3:
            accuracy = stats.get('accuracy_rate', 0.0) * 100
            delta_color = "normal"
            if accuracy > 80:
                delta_color = "inverse"
            st.metric("Current AI Accuracy", f"{accuracy:.1f}%")
        with col4:
            confirmations = stats.get('total_confirmations', 0)
            st.metric("Confirmed Correct", confirmations)

        # Performance Trend Indicator
        trend = stats.get('improvement_trend', 'Unknown')
        if accuracy > 85:
            st.success(f"🎯 **Excellent Performance!** The AI is performing very well with {accuracy:.1f}% accuracy")
        elif accuracy > 70:
            st.info(f"📈 **Good Progress!** The AI is learning well with {accuracy:.1f}% accuracy")
        elif accuracy > 50:
            st.warning(f"⚠️ **Needs Improvement** - AI accuracy is {accuracy:.1f}%. More training data needed")
        else:
            st.error(f"❌ **Requires Attention** - AI accuracy is low at {accuracy:.1f}%")

        # Field-Specific Performance Analysis
        st.subheader("🎯 Field-Specific AI Performance")

        if 'field_accuracy' in patterns and patterns['field_accuracy']:
            accuracy_data = patterns['field_accuracy']

            # Create two columns for field analysis
            col1, col2 = st.columns([1, 1])

            with col1:
                if PLOTLY_AVAILABLE:
                    # Create field accuracy bar chart
                    fields = list(accuracy_data.keys())
                    accuracies = [accuracy_data[field] * 100 for field in fields]

                    # Color bars based on accuracy
                    colors = ['#28a745' if acc > 80 else '#ffc107' if acc > 60 else '#dc3545' for acc in accuracies]

                    fig = go.Figure(data=[
                        go.Bar(
                            x=[field.replace('_', ' ').title() for field in fields],
                            y=accuracies,
                            marker_color=colors,
                            text=[f"{acc:.1f}%" for acc in accuracies],
                            textposition='outside',
                            hovertemplate='<b>%{x}</b><br>Accuracy: %{y:.1f}%<extra></extra>'
                        )
                    ])

                    fig.update_layout(
                        title="Field Extraction Accuracy by Type",
                        xaxis_title="Invoice Fields",
                        yaxis_title="Accuracy (%)",
                        yaxis=dict(range=[0, 100]),
                        height=400,
                        showlegend=False
                    )

                    fig.update_xaxes(tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("📊 Chart visualization requires Plotly - showing text summary")
                    for field, accuracy in accuracy_data.items():
                        field_display = field.replace('_', ' ').title()
                        color = "🟢" if accuracy > 0.8 else "🟡" if accuracy > 0.6 else "🔴"
                        st.write(f"{color} **{field_display}:** {accuracy:.0%}")

            with col2:
                st.markdown("#### 📈 Field Performance Summary")

                # Calculate field statistics
                high_performance = len([acc for acc in accuracy_data.values() if acc > 0.8])
                medium_performance = len([acc for acc in accuracy_data.values() if 0.6 <= acc <= 0.8])
                low_performance = len([acc for acc in accuracy_data.values() if acc < 0.6])

                st.metric("🟢 High Accuracy Fields", f"{high_performance}/{len(accuracy_data)}")
                st.metric("🟡 Medium Accuracy Fields", f"{medium_performance}/{len(accuracy_data)}")
                st.metric("🔴 Needs Improvement", f"{low_performance}/{len(accuracy_data)}")

                # Show best and worst performing fields
                if accuracy_data:
                    best_field = max(accuracy_data.items(), key=lambda x: x[1])
                    worst_field = min(accuracy_data.items(), key=lambda x: x[1])

                    st.markdown("**🏆 Best Performing Field:**")
                    st.success(f"{best_field[0].replace('_', ' ').title()} - {best_field[1]:.0%}")

                    st.markdown("**🎯 Needs Most Attention:**")
                    if worst_field[1] < 0.7:
                        st.error(f"{worst_field[0].replace('_', ' ').title()} - {worst_field[1]:.0%}")
                    else:
                        st.info(f"{worst_field[0].replace('_', ' ').title()} - {worst_field[1]:.0%}")

        # Learning Patterns Analysis
        st.subheader("🔍 AI Learning Patterns")

        if 'common_mistakes' in patterns and patterns['common_mistakes']:
            st.markdown("#### 📚 Most Common Learning Patterns")

            # Show common corrections the AI has learned
            st.info("💡 These are the most frequent corrections users have made. The AI has learned these patterns!")

            for i, mistake in enumerate(patterns['common_mistakes'][:8], 1):
                with st.expander(
                        f"Pattern #{i}: {mistake['field'].replace('_', ' ').title()} - {mistake['count']} occurrences"):
                    col1, col2, col3 = st.columns([1, 1, 1])

                    with col1:
                        st.markdown("**❌ AI Originally Extracted:**")
                        st.code(mistake['wrong'], language=None)

                    with col2:
                        st.markdown("**✅ Users Corrected To:**")
                        st.code(mistake['correct'], language=None)

                    with col3:
                        st.markdown("**📊 Learning Impact:**")
                        st.metric("Times Corrected", mistake['count'])
                        st.success("🧠 Pattern Learned!")
        else:
            st.info(
                "🌟 No learning patterns yet. Process more invoices and make corrections to see AI learning patterns!")

        # Areas for Improvement
        st.subheader("⚠️ Areas for AI Improvement")

        problematic = stats.get('most_problematic_fields', [])

        if problematic:
            st.warning("🎯 These fields need the most attention based on user corrections:")

            for i, (field_name, error_count) in enumerate(problematic, 1):
                with st.expander(f"{i}. {field_name.replace('_', ' ').title()} - {error_count} corrections needed"):

                    col1, col2 = st.columns([1, 2])

                    with col1:
                        st.metric("Total Corrections", error_count)

                        # Calculate improvement suggestions
                        if error_count > 5:
                            st.error("🔴 High Priority")
                            improvement_msg = "This field requires immediate attention. Consider reviewing extraction patterns."
                        elif error_count > 2:
                            st.warning("🟡 Medium Priority")
                            improvement_msg = "This field shows room for improvement with more training data."
                        else:
                            st.info("🟢 Low Priority")
                            improvement_msg = "This field is performing relatively well."

                    with col2:
                        st.markdown("**🔧 Improvement Recommendations:**")
                        st.write(improvement_msg)

                        # Show recent corrections for this field if available
                        try:
                            db_session = get_db_session()
                            if db_session:
                                recent_corrections = db_session.query(UserFeedback).filter(
                                    UserFeedback.field_name == field_name,
                                    UserFeedback.feedback_type == 'correction'
                                ).order_by(UserFeedback.feedback_date.desc()).limit(3).all()

                                if recent_corrections:
                                    st.markdown("**Recent correction examples:**")
                                    for correction in recent_corrections:
                                        st.write(
                                            f"• `'{correction.original_value}'` → `'{correction.corrected_value}'`")
                                db_session.close()
                        except Exception as e:
                            logger.warning(f"Could not load recent corrections: {e}")
        else:
            st.success("🎉 **Excellent!** No major issues detected. The AI is performing well across all fields!")

        # Learning Recommendations
        st.subheader("💡 AI Training Recommendations")

        total_extractions = stats.get('total_extractions', 0)
        total_corrections = stats.get('total_corrections', 0)

        recommendations = []

        if total_extractions < 5:
            recommendations.append("📤 **Process more invoices** - Need at least 10-20 invoices for meaningful learning")

        if total_corrections < 3 and total_extractions > 0:
            recommendations.append(
                "✏️ **Provide more corrections** - Even confirming correct extractions helps the AI learn")

        if accuracy < 70 and total_corrections > 10:
            recommendations.append("🔄 **Review extraction patterns** - Consider improving base extraction rules")

        if accuracy > 85:
            recommendations.append("🎯 **Excellent progress!** - Continue providing feedback to maintain high accuracy")

        if len(problematic) > 3:
            recommendations.append("🎯 **Focus on problematic fields** - Target corrections on fields with most errors")

        if recommendations:
            for rec in recommendations:
                st.info(rec)
        else:
            st.success("🌟 **Great job!** The AI learning system is working optimally!")

        # Future Improvements Preview
        st.subheader("🚀 Planned AI Enhancements")

        st.markdown("""
        **Coming Soon:**
        - 🤖 **Advanced Pattern Recognition** - More sophisticated learning algorithms
        - 📊 **Confidence Score Improvements** - Better accuracy prediction
        - 🧮 **Machine Learning Integration** - Deep learning models for complex documents
        - 📝 **Custom Field Training** - Train AI on company-specific invoice formats
        - 🔄 **Automatic Model Retraining** - Periodic model updates based on corrections
        - 📈 **Advanced Analytics** - Detailed performance metrics and trends
        """)

        # Quick Actions
        st.subheader("⚡ Quick Actions")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("📤 Process New Invoice", use_container_width=True):
                st.switch_page("Upload & Process")

        with col2:
            if st.button("📊 View Feedback History", use_container_width=True):
                # This would switch to feedback page - for now just show info
                st.info("💡 Go to 'Feedback & Corrections History' page to see detailed feedback data")

        with col3:
            if st.button("🔄 Refresh Analytics", use_container_width=True):
                st.rerun()

    except Exception as e:
        st.error(f"Error loading AI learning dashboard: {e}")
        logger.error(f"Error in show_learning_dashboard: {e}")

        # Show basic troubleshooting
        st.subheader("🔧 Troubleshooting")
        st.info("""
        **If you're seeing errors:**
        1. Make sure you've processed at least one invoice
        2. Try making some corrections to generate learning data
        3. Check that the database is working properly
        4. Refresh the page or restart the application
        """)
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