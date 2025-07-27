from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

Base = declarative_base()


class Invoice(Base):
    """Table to store invoice information"""
    __tablename__ = 'invoices'

    id = Column(Integer, primary_key=True)
    filename = Column(String(255), nullable=False)
    upload_date = Column(DateTime, default=datetime.utcnow)
    file_type = Column(String(10))  # pdf, png, jpg, csv
    processing_status = Column(String(20), default='pending')  # pending, processed, error

    # Extracted fields
    invoice_number = Column(String(100))
    invoice_date = Column(String(50))
    supplier_name = Column(String(200))
    total_amount = Column(Float)
    currency = Column(String(10))
    vat_amount = Column(Float)

    # Confidence scores (0-1)
    confidence_invoice_number = Column(Float, default=0.0)
    confidence_date = Column(Float, default=0.0)
    confidence_supplier = Column(Float, default=0.0)
    confidence_total = Column(Float, default=0.0)

    # Raw extracted text
    raw_text = Column(Text)


class UserCorrection(Base):
    """Table to store user corrections for training"""
    __tablename__ = 'user_corrections'

    id = Column(Integer, primary_key=True)
    invoice_id = Column(Integer)
    field_name = Column(String(50))  # which field was corrected
    original_value = Column(Text)  # what AI extracted
    corrected_value = Column(Text)  # what user corrected to
    correction_date = Column(DateTime, default=datetime.utcnow)
    confidence_before = Column(Float)  # confidence before correction


class ModelVersion(Base):
    """Table to track model versions"""
    __tablename__ = 'model_versions'

    id = Column(Integer, primary_key=True)
    version_name = Column(String(50), unique=True)
    creation_date = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=False)
    accuracy_score = Column(Float)
    description = Column(Text)


class OCRResult(Base):
    """Table to store OCR processing results"""
    __tablename__ = 'ocr_results'

    id = Column(Integer, primary_key=True)
    invoice_id = Column(Integer)  # Foreign key to invoices table
    extracted_text = Column(Text)
    confidence_score = Column(Float)
    processing_time = Column(Float)  # seconds
    ocr_method = Column(String(50))  # tesseract, manual, etc.
    pages_processed = Column(Integer, default=1)
    created_date = Column(DateTime, default=datetime.utcnow)

    # Technical details
    image_preprocessing = Column(Boolean, default=False)
    tesseract_config = Column(String(200))
    error_message = Column(Text)


class FieldExtraction(Base):
    """Table to store extracted fields from invoices"""
    __tablename__ = 'field_extractions'

    id = Column(Integer, primary_key=True)
    invoice_id = Column(Integer)

    # Extracted fields (AI predictions)
    invoice_number_extracted = Column(String(100))
    invoice_date_extracted = Column(String(50))
    supplier_name_extracted = Column(String(200))
    total_amount_extracted = Column(Float)
    currency_extracted = Column(String(10))
    vat_amount_extracted = Column(Float)

    # User corrections
    invoice_number_corrected = Column(String(100))
    invoice_date_corrected = Column(String(50))
    supplier_name_corrected = Column(String(200))
    total_amount_corrected = Column(Float)
    currency_corrected = Column(String(10))
    vat_amount_corrected = Column(Float)

    # Feedback metadata
    feedback_provided = Column(Boolean, default=False)
    correction_count = Column(Integer, default=0)
    feedback_date = Column(DateTime)
    user_notes = Column(Text)

    created_date = Column(DateTime, default=datetime.utcnow)


class UserFeedback(Base):
    """Table to store user feedback for model improvement"""
    __tablename__ = 'user_feedback'

    id = Column(Integer, primary_key=True)
    invoice_id = Column(Integer)
    field_name = Column(String(50))
    original_value = Column(Text)
    corrected_value = Column(Text)
    feedback_type = Column(String(20))  # 'correction', 'confirmation', 'flag'
    confidence_before = Column(Float)
    user_rating = Column(Integer)  # 1-5 star rating
    feedback_date = Column(DateTime, default=datetime.utcnow)
    is_used_for_training = Column(Boolean, default=False)

# Database setup function
def init_database():
    """Initialize the database and create tables"""
    db_path = 'database/invoice_system.db'

    # Create database directory if it doesn't exist
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    # Create engine and tables
    engine = create_engine(f'sqlite:///{db_path}')
    Base.metadata.create_all(engine)

    # Create session maker
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    return engine, SessionLocal


# Helper function to get database session
def get_db_session():
    """Get a database session"""
    _, SessionLocal = init_database()
    return SessionLocal()