import logging
import re
from typing import Dict, List, Any
from database.models import get_db_session, UserFeedback, FieldExtraction, Invoice
from datetime import datetime

logger = logging.getLogger(__name__)


class LearningSystem:
    """Enhanced learning system that improves extractions in real-time"""

    def __init__(self):
        self.improvement_threshold = 0.8
        self.learning_rate = 0.1
        self.learned_patterns = {}
        self._load_learned_patterns()

    def _load_learned_patterns(self):
        """Load previously learned patterns from user corrections"""
        try:
            db_session = get_db_session()
            if not db_session:
                return

            # Get all corrections to learn from
            corrections = db_session.query(UserFeedback).filter(
                UserFeedback.feedback_type == 'correction'
            ).all()

            # Group corrections by field
            field_corrections = {}
            for correction in corrections:
                field_name = correction.field_name
                if field_name not in field_corrections:
                    field_corrections[field_name] = []

                field_corrections[field_name].append({
                    'original': correction.original_value,
                    'corrected': correction.corrected_value,
                    'confidence': correction.confidence_before
                })

            # Learn patterns from corrections
            for field_name, corrections_list in field_corrections.items():
                self.learned_patterns[field_name] = self._extract_patterns_from_corrections(
                    corrections_list
                )

            logger.info(f"Loaded learned patterns for {len(self.learned_patterns)} fields")

        except Exception as e:
            logger.error(f"Error loading learned patterns: {e}")
        finally:
            if 'db_session' in locals() and db_session:
                db_session.close()

    def _extract_patterns_from_corrections(self, corrections_list):
        """Extract patterns from user corrections"""
        patterns = {
            'common_corrections': {},
            'improved_patterns': [],
            'confidence_boosters': [],
            'correction_count': len(corrections_list)
        }

        for correction in corrections_list:
            original = correction['original'].strip()
            corrected = correction['corrected'].strip()

            if original != corrected:
                # Store common correction mapping
                patterns['common_corrections'][original.lower()] = corrected

                # If original was empty but user provided value, learn the pattern
                if not original and corrected:
                    patterns['confidence_boosters'].append(corrected)

                # Learn pattern improvements for specific field types
                if len(corrected) > 2:
                    patterns['improved_patterns'].append(corrected)

        return patterns

    def apply_learned_patterns(self, field_extractor, filename):
        """Apply learned patterns to improve extraction before processing"""

        # Enhance field extractor with learned patterns
        enhanced_extractor = EnhancedFieldExtractor(field_extractor, self.learned_patterns)

        logger.info(f"Applied learned patterns for {filename}")
        return enhanced_extractor

    def save_field_corrections(self, invoice_id: int, original_fields: Dict, corrected_fields: Dict) -> bool:
        """Save user corrections and immediately update learning patterns"""

        try:
            db_session = get_db_session()
            if not db_session:
                return False

            corrections_made = 0
            new_learnings = []

            # Compare each field and save corrections
            for field_name in original_fields.keys():
                original_value = str(original_fields[field_name]['value']).strip()
                corrected_value = str(corrected_fields.get(field_name, original_value)).strip()

                # Determine feedback type
                if original_value != corrected_value:
                    feedback_type = 'correction'
                    corrections_made += 1
                else:
                    feedback_type = 'confirmation'

                # Save individual field feedback
                feedback = UserFeedback(
                    invoice_id=invoice_id,
                    field_name=field_name,
                    original_value=original_value,
                    corrected_value=corrected_value,
                    feedback_type=feedback_type,
                    confidence_before=original_fields[field_name]['confidence'],
                    user_rating=5 if feedback_type == 'confirmation' else 3,  # Higher rating for confirmations
                    is_used_for_training=True
                )

                db_session.add(feedback)

                # Record the learning for immediate application
                if feedback_type == 'correction':
                    new_learnings.append({
                        'field': field_name,
                        'original': original_value,
                        'corrected': corrected_value,
                        'confidence': original_fields[field_name]['confidence']
                    })

            # Save complete field extraction record
            field_extraction = FieldExtraction(
                invoice_id=invoice_id,

                # Original extractions
                invoice_number_extracted=str(original_fields.get('invoice_number', {}).get('value', '')),
                invoice_date_extracted=str(original_fields.get('date', {}).get('value', '')),
                supplier_name_extracted=str(original_fields.get('supplier', {}).get('value', '')),
                total_amount_extracted=float(original_fields.get('total', {}).get('value', 0)),
                vat_amount_extracted=float(original_fields.get('vat', {}).get('value', 0)),

                # User corrections
                invoice_number_corrected=str(
                    corrected_fields.get('invoice_number', original_fields.get('invoice_number', {}).get('value', ''))),
                invoice_date_corrected=str(
                    corrected_fields.get('date', original_fields.get('date', {}).get('value', ''))),
                supplier_name_corrected=str(
                    corrected_fields.get('supplier', original_fields.get('supplier', {}).get('value', ''))),
                total_amount_corrected=float(
                    corrected_fields.get('total', original_fields.get('total', {}).get('value', 0))),
                vat_amount_corrected=float(corrected_fields.get('vat', original_fields.get('vat', {}).get('value', 0))),

                # Metadata
                feedback_provided=True,
                correction_count=corrections_made,
                feedback_date=datetime.utcnow()
            )

            db_session.add(field_extraction)
            db_session.commit()

            # Immediately update learned patterns with new corrections
            if new_learnings:
                self._update_learned_patterns(new_learnings)

            logger.info(f"Saved {corrections_made} corrections for invoice {invoice_id} and updated learning patterns")
            return True

        except Exception as e:
            logger.error(f"Error saving corrections: {e}")
            if 'db_session' in locals() and db_session:
                db_session.rollback()
            return False
        finally:
            if 'db_session' in locals() and db_session:
                db_session.close()

    def _update_learned_patterns(self, new_learnings):
        """Update learned patterns with new corrections immediately"""

        for learning in new_learnings:
            field_name = learning['field']
            original = learning['original']
            corrected = learning['corrected']

            # Initialize field patterns if not exists
            if field_name not in self.learned_patterns:
                self.learned_patterns[field_name] = {
                    'common_corrections': {},
                    'improved_patterns': [],
                    'confidence_boosters': [],
                    'correction_count': 0
                }

            patterns = self.learned_patterns[field_name]
            patterns['correction_count'] += 1

            # Add to common corrections
            if original:
                patterns['common_corrections'][original.lower()] = corrected

            # Add to confidence boosters if it was a new detection
            if not original and corrected:
                if corrected not in patterns['confidence_boosters']:
                    patterns['confidence_boosters'].append(corrected)

            # Add to improved patterns
            if len(corrected) > 2 and corrected not in patterns['improved_patterns']:
                patterns['improved_patterns'].append(corrected)

        logger.info(f"Updated learned patterns with {len(new_learnings)} new corrections")

    def get_field_statistics(self) -> Dict:
        """Get comprehensive statistics about field extraction performance"""
        try:
            db_session = get_db_session()
            if not db_session:
                return self._empty_stats()

            # Get all field extractions
            extractions = db_session.query(FieldExtraction).all()
            feedbacks = db_session.query(UserFeedback).all()

            if not extractions:
                return self._empty_stats()

            total_extractions = len(extractions)
            total_corrections = sum(e.correction_count for e in extractions)
            total_confirmations = len([f for f in feedbacks if f.feedback_type == 'confirmation'])

            # Calculate accuracy rate (improved formula)
            total_fields_processed = total_extractions * 8  # 8 main fields per invoice
            correct_fields = total_fields_processed - total_corrections + total_confirmations
            accuracy_rate = correct_fields / total_fields_processed if total_fields_processed > 0 else 0.0
            accuracy_rate = max(0.0, min(1.0, accuracy_rate))  # Clamp between 0 and 1

            # Find most problematic fields
            field_errors = {}
            field_totals = {}

            for extraction in extractions:
                # Count corrections for each field
                fields_to_check = [
                    ('invoice_number', extraction.invoice_number_extracted, extraction.invoice_number_corrected),
                    ('date', extraction.invoice_date_extracted, extraction.invoice_date_corrected),
                    ('supplier', extraction.supplier_name_extracted, extraction.supplier_name_corrected),
                    ('total_amount', extraction.total_amount_extracted, extraction.total_amount_corrected),
                    ('vat_amount', extraction.vat_amount_extracted, extraction.vat_amount_corrected)
                ]

                for field_name, extracted, corrected in fields_to_check:
                    field_totals[field_name] = field_totals.get(field_name, 0) + 1

                    # Check if correction was needed
                    if str(extracted).strip() != str(corrected).strip():
                        field_errors[field_name] = field_errors.get(field_name, 0) + 1

            # Sort by error count
            problematic_fields = sorted(field_errors.items(), key=lambda x: x[1], reverse=True)

            return {
                'total_extractions': total_extractions,
                'total_corrections': total_corrections,
                'total_confirmations': total_confirmations,
                'accuracy_rate': accuracy_rate,
                'most_problematic_fields': problematic_fields[:3],
                'improvement_trend': 'Improving' if accuracy_rate > 0.7 else 'Needs attention',
                'field_totals': field_totals,
                'field_errors': field_errors
            }

        except Exception as e:
            logger.error(f"Error getting field statistics: {e}")
            return self._empty_stats()
        finally:
            if 'db_session' in locals() and db_session:
                db_session.close()

    def _empty_stats(self):
        """Return empty statistics when no data available"""
        return {
            'total_extractions': 0,
            'total_corrections': 0,
            'total_confirmations': 0,
            'accuracy_rate': 0.0,
            'most_problematic_fields': [],
            'improvement_trend': 'No data',
            'field_totals': {},
            'field_errors': {}
        }

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

            confirmations = db_session.query(UserFeedback).filter(
                UserFeedback.feedback_type == 'confirmation'
            ).all()

            # Analyze patterns
            patterns = {
                'total_corrections': len(corrections),
                'total_confirmations': len(confirmations),
                'field_accuracy': {},
                'common_mistakes': {},
                'improvement_suggestions': []
            }

            # Calculate accuracy per field
            field_counts = {}
            field_errors = {}

            # Count total attempts per field
            for correction in corrections + confirmations:
                field_name = correction.field_name
                field_counts[field_name] = field_counts.get(field_name, 0) + 1

            # Count errors (corrections needed)
            for correction in corrections:
                field_name = correction.field_name
                field_errors[field_name] = field_errors.get(field_name, 0) + 1

            # Calculate accuracy rates
            for field_name in field_counts:
                total_attempts = field_counts[field_name]
                errors = field_errors.get(field_name, 0)
                accuracy = (total_attempts - errors) / total_attempts if total_attempts > 0 else 0.0
                patterns['field_accuracy'][field_name] = accuracy

            # Identify common mistakes
            mistake_patterns = {}
            for correction in corrections:
                original = correction.original_value.strip().lower()
                corrected = correction.corrected_value.strip()

                if original and corrected:
                    mistake_key = f"{correction.field_name}:{original}"
                    if mistake_key not in mistake_patterns:
                        mistake_patterns[mistake_key] = {
                            'field': correction.field_name,
                            'wrong': original,
                            'correct': corrected,
                            'count': 0
                        }
                    mistake_patterns[mistake_key]['count'] += 1

            # Get most common mistakes
            common_mistakes = sorted(mistake_patterns.values(), key=lambda x: x['count'], reverse=True)[:5]
            patterns['common_mistakes'] = common_mistakes

            return patterns

        except Exception as e:
            logger.error(f"Error analyzing learning patterns: {e}")
            return {'error': str(e)}
        finally:
            if 'db_session' in locals() and db_session:
                db_session.close()


class EnhancedFieldExtractor:
    """Enhanced field extractor that uses learned patterns"""

    def __init__(self, base_extractor, learned_patterns):
        self.base_extractor = base_extractor
        self.learned_patterns = learned_patterns

    def extract_all_fields(self, text: str) -> Dict:
        """Extract fields using base extractor + learned patterns"""

        # Get base extraction
        base_results = self.base_extractor.extract_all_fields(text)

        # Enhance with learned patterns
        enhanced_results = {}

        for field_name, field_data in base_results.items():
            enhanced_results[field_name] = self._enhance_field_extraction(
                field_name, field_data, text
            )

        return enhanced_results

    def _enhance_field_extraction(self, field_name, field_data, text):
        """Enhance single field extraction with learned patterns"""

        if field_name not in self.learned_patterns:
            return field_data

        patterns = self.learned_patterns[field_name]
        original_value = str(field_data['value']).strip()
        original_confidence = field_data['confidence']

        # Apply common corrections
        if original_value.lower() in patterns['common_corrections']:
            corrected_value = patterns['common_corrections'][original_value.lower()]
            return {
                'value': corrected_value,
                'confidence': min(0.95, original_confidence + 0.3),  # Boost confidence
                'method': 'learned_correction'
            }

        # If original extraction failed, try confidence boosters
        if not original_value and patterns['confidence_boosters']:
            # Look for any of the learned patterns in the text
            text_lower = text.lower()
            for booster in patterns['confidence_boosters']:
                if booster.lower() in text_lower:
                    # Extract surrounding context
                    context = self._extract_context(text, booster)
                    if context:
                        return {
                            'value': context,
                            'confidence': 0.75,  # Medium confidence for learned pattern
                            'method': 'learned_pattern'
                        }

        # If still no good result, try improved patterns
        if original_confidence < 0.5 and patterns['improved_patterns']:
            text_lower = text.lower()
            for pattern in patterns['improved_patterns']:
                if pattern.lower() in text_lower:
                    return {
                        'value': pattern,
                        'confidence': 0.70,
                        'method': 'learned_improvement'
                    }

        return field_data

    def _extract_context(self, text, pattern):
        """Extract context around a found pattern"""
        try:
            # Find the pattern in text (case insensitive)
            pattern_lower = pattern.lower()
            text_lower = text.lower()

            start_idx = text_lower.find(pattern_lower)
            if start_idx == -1:
                return None

            # Extract the actual case-preserved text
            end_idx = start_idx + len(pattern)
            return text[start_idx:end_idx].strip()

        except Exception:
            return pattern