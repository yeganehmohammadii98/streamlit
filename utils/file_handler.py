import os
import tempfile
import shutil
from typing import BinaryIO, Dict, Any
import logging

logger = logging.getLogger(__name__)


class FileHandler:
    """Handles file operations and validation"""

    SUPPORTED_TYPES = {
        'application/pdf': 'pdf',
        'image/png': 'png',
        'image/jpeg': 'jpg',
        'image/jpg': 'jpg',
        'text/csv': 'csv'
    }

    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

    def __init__(self, upload_dir='uploads'):
        """Initialize file handler

        Args:
            upload_dir: Directory to store uploaded files
        """
        self.upload_dir = upload_dir
        os.makedirs(upload_dir, exist_ok=True)

    def validate_file(self, uploaded_file) -> Dict[str, Any]:
        """Validate uploaded file

        Args:
            uploaded_file: Streamlit uploaded file object

        Returns:
            dict: Validation result
        """
        try:
            # Check file type
            if uploaded_file.type not in self.SUPPORTED_TYPES:
                return {
                    'valid': False,
                    'error': f"Unsupported file type: {uploaded_file.type}"
                }

            # Check file size
            if uploaded_file.size > self.MAX_FILE_SIZE:
                return {
                    'valid': False,
                    'error': f"File too large: {uploaded_file.size / 1024 / 1024:.1f}MB (max: {self.MAX_FILE_SIZE / 1024 / 1024}MB)"
                }

            # Check if file is not empty
            if uploaded_file.size == 0:
                return {
                    'valid': False,
                    'error': "File is empty"
                }

            return {
                'valid': True,
                'file_type': self.SUPPORTED_TYPES[uploaded_file.type],
                'size': uploaded_file.size
            }

        except Exception as e:
            logger.error(f"File validation error: {e}")
            return {
                'valid': False,
                'error': f"Validation error: {str(e)}"
            }

    def save_uploaded_file(self, uploaded_file) -> str:
        """Save uploaded file to disk

        Args:
            uploaded_file: Streamlit uploaded file object

        Returns:
            str: Path to saved file
        """
        try:
            # Create safe filename
            safe_filename = self._create_safe_filename(uploaded_file.name)
            file_path = os.path.join(self.upload_dir, safe_filename)

            # Save file
            with open(file_path, 'wb') as f:
                shutil.copyfileobj(uploaded_file, f)

            logger.info(f"File saved: {file_path}")
            return file_path

        except Exception as e:
            logger.error(f"Error saving file: {e}")
            raise

    def get_file_content(self, uploaded_file) -> bytes:
        """Get file content as bytes

        Args:
            uploaded_file: Streamlit uploaded file object

        Returns:
            bytes: File content
        """
        try:
            # Reset file pointer to beginning
            uploaded_file.seek(0)
            content = uploaded_file.read()
            uploaded_file.seek(0)  # Reset again for other uses

            return content

        except Exception as e:
            logger.error(f"Error reading file content: {e}")
            raise

    def _create_safe_filename(self, filename: str) -> str:
        """Create a safe filename for saving

        Args:
            filename: Original filename

        Returns:
            str: Safe filename
        """
        import re
        from datetime import datetime

        # Remove unsafe characters
        safe_name = re.sub(r'[^\w\-_\.]', '_', filename)

        # Add timestamp to prevent conflicts
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        name, ext = os.path.splitext(safe_name)

        return f"{timestamp}_{name}{ext}"

    def cleanup_old_files(self, days_old=7):
        """Clean up old uploaded files

        Args:
            days_old: Remove files older than this many days
        """
        try:
            import time
            cutoff_time = time.time() - (days_old * 24 * 60 * 60)

            for filename in os.listdir(self.upload_dir):
                file_path = os.path.join(self.upload_dir, filename)

                if os.path.isfile(file_path):
                    if os.path.getmtime(file_path) < cutoff_time:
                        os.remove(file_path)
                        logger.info(f"Cleaned up old file: {filename}")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


def get_file_preview_info(uploaded_file) -> Dict[str, Any]:
    """Get information for file preview

    Args:
        uploaded_file: Streamlit uploaded file object

    Returns:
        dict: Preview information
    """
    info = {
        'name': uploaded_file.name,
        'type': uploaded_file.type,
        'size': uploaded_file.size,
        'size_mb': uploaded_file.size / 1024 / 1024,
        'extension': os.path.splitext(uploaded_file.name)[1].lower()
    }

    return info