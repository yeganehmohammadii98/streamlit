import pytesseract
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import io
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OCRProcessor:
    """Handles OCR processing for different file types"""

    def __init__(self, tesseract_path=None):
        """Initialize OCR processor

        Args:
            tesseract_path: Path to tesseract executable (for Windows)
        """
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path

        # Test if tesseract is available
        try:
            pytesseract.get_tesseract_version()
            logger.info("Tesseract OCR initialized successfully")
        except Exception as e:
            logger.error(f"Tesseract initialization failed: {e}")
            raise

    def process_file(self, file_content, file_type, filename):
        """Process file and extract text based on file type

        Args:
            file_content: File content as bytes
            file_type: Type of file (pdf, image, csv)
            filename: Original filename

        Returns:
            dict: Processing results with text, confidence, etc.
        """
        try:
            if file_type.lower() == 'pdf':
                return self._process_pdf(file_content, filename)
            elif file_type.lower() in ['png', 'jpg', 'jpeg']:
                return self._process_image(file_content, filename)
            elif file_type.lower() == 'csv':
                return self._process_csv(file_content, filename)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")

        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")
            return {
                'success': False,
                'error': str(e),
                'text': '',
                'confidence': 0.0,
                'pages': 0
            }

    def _process_pdf(self, pdf_content, filename):
        """Extract text from PDF using OCR"""
        logger.info(f"Processing PDF: {filename}")

        try:
            # Try to import pdf2image
            try:
                from pdf2image import convert_from_bytes
            except ImportError:
                return {
                    'success': False,
                    'error': 'pdf2image not available - cannot process PDF files',
                    'text': '',
                    'confidence': 0.0,
                    'pages': 0
                }

            # Convert PDF to images
            images = convert_from_bytes(
                pdf_content,
                dpi=300,  # High resolution for better OCR
                first_page=1,
                last_page=10  # Limit to first 10 pages
            )

            all_text = []
            total_confidence = 0

            for i, image in enumerate(images):
                logger.info(f"Processing page {i + 1}/{len(images)}")

                # Preprocess image for better OCR
                processed_image = self._preprocess_image(image)

                # Extract text with confidence
                ocr_data = pytesseract.image_to_data(
                    processed_image,
                    output_type=pytesseract.Output.DICT,
                    config='--psm 6'  # Uniform block of text
                )

                # Extract text and calculate confidence
                page_text = self._extract_text_from_ocr_data(ocr_data)
                page_confidence = self._calculate_confidence(ocr_data)

                all_text.append(f"--- Page {i + 1} ---\n{page_text}")
                total_confidence += page_confidence

            # Combine all pages
            combined_text = "\n\n".join(all_text)
            avg_confidence = total_confidence / len(images) if images else 0

            return {
                'success': True,
                'text': combined_text,
                'confidence': avg_confidence,
                'pages': len(images),
                'file_type': 'pdf'
            }

        except Exception as e:
            logger.error(f"PDF processing error: {e}")
            return {
                'success': False,
                'error': str(e),
                'text': '',
                'confidence': 0.0,
                'pages': 0
            }

    def _process_image(self, image_content, filename):
        """Extract text from image using OCR"""
        logger.info(f"Processing image: {filename}")

        try:
            # Load image
            image = Image.open(io.BytesIO(image_content))

            # Preprocess image
            processed_image = self._preprocess_image(image)

            # Extract text with confidence
            ocr_data = pytesseract.image_to_data(
                processed_image,
                output_type=pytesseract.Output.DICT,
                config='--psm 6'
            )

            # Extract text and confidence
            text = self._extract_text_from_ocr_data(ocr_data)
            confidence = self._calculate_confidence(ocr_data)

            return {
                'success': True,
                'text': text,
                'confidence': confidence,
                'pages': 1,
                'file_type': 'image'
            }

        except Exception as e:
            logger.error(f"Image processing error: {e}")
            return {
                'success': False,
                'error': str(e),
                'text': '',
                'confidence': 0.0,
                'pages': 0
            }

    def _process_csv(self, csv_content, filename):
        """Process CSV file (no OCR needed)"""
        logger.info(f"Processing CSV: {filename}")

        try:
            # Decode CSV content
            text_content = csv_content.decode('utf-8')

            return {
                'success': True,
                'text': text_content,
                'confidence': 1.0,  # CSV is already text
                'pages': 1,
                'file_type': 'csv'
            }

        except Exception as e:
            logger.error(f"CSV processing error: {e}")
            return {
                'success': False,
                'error': str(e),
                'text': '',
                'confidence': 0.0,
                'pages': 0
            }

    def _preprocess_image(self, image):
        """Preprocess image for better OCR results using PIL only"""
        try:
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Convert to grayscale
            gray_image = image.convert('L')

            # Enhance contrast
            enhancer = ImageEnhance.Contrast(gray_image)
            contrast_image = enhancer.enhance(2.0)

            # Enhance sharpness
            enhancer = ImageEnhance.Sharpness(contrast_image)
            sharp_image = enhancer.enhance(1.5)

            # Auto-level (similar to histogram equalization)
            auto_image = ImageOps.autocontrast(sharp_image)

            # Apply slight blur to reduce noise
            smooth_image = auto_image.filter(ImageFilter.GaussianBlur(radius=0.5))

            return smooth_image

        except Exception as e:
            logger.warning(f"Image preprocessing failed, using original: {e}")
            return image

    def _extract_text_from_ocr_data(self, ocr_data):
        """Extract clean text from OCR data"""
        words = []

        for i in range(len(ocr_data['text'])):
            confidence = int(ocr_data['conf'][i])
            text = ocr_data['text'][i].strip()

            # Only include words with reasonable confidence
            if confidence > 30 and text:
                words.append(text)

        # Join words and clean up
        full_text = ' '.join(words)

        # Basic text cleaning
        full_text = self._clean_text(full_text)

        return full_text

    def _calculate_confidence(self, ocr_data):
        """Calculate average confidence score"""
        confidences = []

        for i in range(len(ocr_data['conf'])):
            conf = int(ocr_data['conf'][i])
            text = ocr_data['text'][i].strip()

            if conf > 0 and text:  # Only count valid words
                confidences.append(conf)

        if confidences:
            return sum(confidences) / len(confidences) / 100.0  # Normalize to 0-1
        else:
            return 0.0

    def _clean_text(self, text):
        """Clean and normalize extracted text"""
        # Remove extra whitespace
        text = ' '.join(text.split())

        # Remove common OCR artifacts
        text = text.replace('|', 'I')  # Common misread

        # Basic formatting
        text = text.strip()

        return text


def get_tesseract_path():
    """Get the appropriate Tesseract path for different OS and cloud environments"""
    import platform
    import shutil

    # First, try to find tesseract in PATH
    tesseract_cmd = shutil.which('tesseract')
    if tesseract_cmd:
        return tesseract_cmd

    system = platform.system().lower()

    # Cloud environment paths (Linux)
    cloud_paths = [
        '/usr/bin/tesseract',
        '/usr/local/bin/tesseract',
        '/app/.apt/usr/bin/tesseract'  # Streamlit Cloud specific
    ]

    for path in cloud_paths:
        if os.path.exists(path):
            return path

    # Local environment paths
    if system == 'windows':
        windows_paths = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Users\{}\AppData\Local\Tesseract-OCR\tesseract.exe'.format(os.getenv('USERNAME', 'user'))
        ]
        for path in windows_paths:
            if os.path.exists(path):
                return path

    elif system == 'darwin':  # macOS
        mac_paths = [
            '/opt/homebrew/bin/tesseract',
            '/usr/local/bin/tesseract'
        ]
        for path in mac_paths:
            if os.path.exists(path):
                return path

    # Fallback
    return 'tesseract'