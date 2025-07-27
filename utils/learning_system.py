import logging
from typing import Dict, List, Any
from database.models import get_db_session, UserFeedback, FieldExtraction, Invoice
from datetime import datetime

logger = logging.getLogger(__name__)


class LearningSystem:
    """Learning system that improves extractions based on user feedback"""

    def __init__(self):
        self.improvement_threshold = 0.8
        self.learning_rate = 0.1

    def get_field_statistics(self) -> Dict:
        """Get statistics about field extraction performance"""
        try:
            db_session = get_db_session()

            # Get all field extractions
            extractions = db_session.query(FieldExtraction).all() if db_session else []

            if not extractions:
                return {
                    'total_extractions': 0,
                    'total_corrections': 0,
                    'accuracy_rate': 0.0,
                    'most_problematic_fields': [],
                    'improvement_trend': 'No data'
                }

            total_extractions = len(extractions)
            total_corrections = sum(e.correction_count for e in extractions)

            # Calculate accuracy rate
            accuracy_rate = 1.0 - (total_corrections / (total_extractions * 8))  # 8 main fields
            accuracy_rate = max(0.0, accuracy_rate)  # Ensure non-negative

            # Find most problematic fields
            field_errors = {}
            for extraction in extractions:
                if extraction.invoice_number_extracted != extraction.invoice_number_corrected:
                    field_errors['invoice_number'] = field_errors.get('invoice_number', 0) + 1
                if extraction.invoice_date_extracted != extraction.invoice_date_corrected:
                    field_errors['date'] = field_errors.get('date', 0) + 1
                if extraction.supplier_name_extracted != extraction.supplier_name_corrected:
                    field_errors['supplier'] = field_errors.get('supplier', 0) + 1
                if abs(extraction.total_amount_extracted - extraction.total_amount_corrected) > 0.01:
                    field_errors['total_amount'] = field_errors.get('total_amount', 0) + 1

            # Sort by error count
            problematic_fields = sorted(field_errors.items(), key=lambda x: x[1], reverse=True)

            return {
                'total_extractions': total_extractions,
                'total_corrections': total_corrections,
                'accuracy_rate': accuracy_rate,
                'most_problematic_fields': problematic_fields[:3],
                'improvement_trend': 'Improving' if accuracy_rate > 0.7 else 'Needs attention'
            }

        except Exception as e:
            logger.error(f"Error getting field statistics: {e}")
            return {
                'total_extractions': 0,
                'total_corrections': 0,
                'accuracy_rate': 0.0,
                'most_problematic_fields': [],
                'improvement_trend': 'Error loading data'
            }
        finally:
            if 'db_session' in locals() and db_session:
                db_session.close()

    def apply_learned_patterns(self, field_extractor, filename):
        """Apply learned patterns to improve extraction"""
        # For now, return the extractor as-is
        # This would be enhanced with actual machine learning in production
        logger.info(f"Applied learned patterns for {filename}")
        return field_extractor

    def save_field_corrections(self, invoice_id: int, original_fields: Dict, corrected_fields: Dict) -> bool:
        """Save user corrections for future learning"""
        try:
            db_session = get_db_session()
            if not db_session:
                return False

            corrections_made = 0

            # Compare each field and save corrections
            for field_name in original_fields.keys():
                original_value = str(original_fields[field_name]['value']).strip()
                corrected_value = str(corrected_fields.get(field_name, original_value)).strip()

                # Check if user made a correction
                if original_value != corrected_value:
                    corrections_made += 1

                    # Save individual field correction
                    feedback = UserFeedback(
                        invoice_id=invoice_id,
                        field_name=field_name,
                        original_value=original_value,
                        corrected_value=corrected_value,
                        feedback_type='correction',
                        confidence_before=original_fields[field_name]['confidence'],
                        user_rating=None,
                        is_used_for_training=True
                    )

                    db_session.add(feedback)

            # Save complete field extraction record
            field_extraction = FieldExtraction(
                invoice_id=invoice_id,
                invoice_number_extracted=str(original_fields.get('invoice_number', {}).get('value', '')),
                invoice_date_extracted=str(original_fields.get('date', {}).get('value', '')),
                supplier_name_extracted=str(original_fields.get('supplier', {}).get('value', '')),
                total_amount_extracted=float(original_fields.get('total', {}).get('value', 0)),
                vat_amount_extracted=float(original_fields.get('vat', {}).get('value', 0)),
                invoice_number_corrected=str(corrected_fields.get('invoice_number', original_fields.get('invoice_number', {}).get('value', ''))),
                invoice_date_corrected=str(corrected_fields.get('date', original_fields.get('date', {}).get('value', ''))),
                supplier_name_corrected=str(corrected_fields.get('supplier', original_fields.get('supplier', {}).get('value', ''))),
                total_amount_corrected=float(corrected_fields.get('total', original_fields.get('total', {}).get('value', 0))),
                vat_amount_corrected=float(corrected_fields.get('vat', original_fields.get('vat', {}).get('value', 0))),
                feedback_provided=True,
                correction_count=corrections_made,
                feedback_date=datetime.utcnow()
            )

            db_session.add(field_extraction)
            db_session.commit()

            logger.info(f"Saved {corrections_made} corrections for invoice {invoice_id}")
            return True

        except Exception as e:
            logger.error(f"Error saving corrections: {e}")
            if 'db_session' in locals() and db_session:
                db_session.rollback()
            return False
        finally:
            if 'db_session' in locals() and db_session:
                db_session.close()

    def get_learning_patterns(self) -> Dict:
        """Analyze correction patterns to improve extraction"""
        try:
            db_session = get_db_session()
            if not db_session:
                return {'error': 'Database not available'}

            # Get all corrections
            corrections = db_session.query(UserFeedback).filter(
                UserFeedback.feedback_type == 'correction'
            ).all()

            # Analyze patterns
            patterns = {
                'total_corrections': len(corrections),
                'field_accuracy': {},
                'common_mistakes': {},
                'improvement_suggestions': []
            }

            # Calculate accuracy per field
            field_counts = {}
            field_correct = {}

            for correction in corrections:
                field_name = correction.field_name
                field_counts[field_name] = field_counts.get(field_name, 0) + 1

                # If original was wrong (needed correction), mark as incorrect
                if correction.original_value != correction.corrected_value:
                    field_correct[field_name] = field_correct.get(field_name, 0)
                else:
                    field_correct[field_name] = field_correct.get(field_name, 0) + 1

            # Calculate accuracy rates
            for field_name in field_counts:
                if field_counts[field_name] > 0:
                    accuracy = field_correct.get(field_name, 0) / field_counts[field_name]
                    patterns['field_accuracy'][field_name] = accuracy

            return patterns

        except Exception as e:
            logger.error(f"Error analyzing learning patterns: {e}")
            return {'error': str(e)}
        finally:
            if 'db_session' in locals() and db_session:
                db_session.close()