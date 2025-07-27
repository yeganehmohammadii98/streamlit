import re
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class FieldExtractor:
    """Extract specific fields from invoice text using rule-based approaches"""

    def __init__(self):
        """Initialize field extraction patterns and rules"""
        self.invoice_patterns = [
            r'(?:invoice|inv|bill|receipt)[\s#:]*([a-z0-9-]+)',
            r'#[\s]*([a-z0-9-]+)',
            r'invoice[\s]*number[\s]*:[\s]*([a-z0-9-]+)',
            r'inv[\s]*#[\s]*([a-z0-9-]+)'
        ]

        self.date_patterns = [
            r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            r'(\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{2,4})',
            r'(?:date|issued|created)[\s:]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            r'(\d{4}-\d{1,2}-\d{1,2})'
        ]

        self.amount_patterns = [
            r'(?:total|amount|sum|due)[\s:$]*([0-9,]+\.?[0-9]*)',
            r'\$[\s]*([0-9,]+\.?[0-9]*)',
            r'([0-9,]+\.?[0-9]*)[\s]*(?:usd|dollar|\$)',
            r'(?:grand[\s]*total|final[\s]*amount)[\s:$]*([0-9,]+\.?[0-9]*)'
        ]

        self.supplier_patterns = [
            r'(?:from|supplier|vendor|company)[\s:]*([a-z\s&,.-]+?)(?:\n|address|phone|email)',
            r'^([a-z\s&,.-]+?)(?:\n|address|phone|email)',
            r'(?:bill[\s]*from|sold[\s]*by)[\s:]*([a-z\s&,.-]+?)(?:\n|address)',
        ]

        self.vat_patterns = [
            r'(?:vat|tax|gst)[\s:$]*([0-9,]+\.?[0-9]*)',
            r'(?:sales[\s]*tax|value[\s]*added[\s]*tax)[\s:$]*([0-9,]+\.?[0-9]*)',
        ]

    def extract_all_fields(self, text: str) -> Dict:
        """Extract all fields from invoice text

        Args:
            text: Extracted text from invoice

        Returns:
            dict: Extracted fields with confidence scores
        """
        text_clean = self._clean_text(text)
        text_lower = text_clean.lower()

        results = {
            'invoice_number': self._extract_invoice_number(text_lower),
            'date': self._extract_date(text_lower),
            'supplier': self._extract_supplier(text_clean),
            'customer': self._extract_customer(text_clean),
            'total': self._extract_total_amount(text_lower),
            'subtotal': self._extract_subtotal(text_lower),
            'vat': self._extract_vat_amount(text_lower),
            'line_items_count': self._extract_line_items_count(text_lower),
            'line_items_subtotal': self._extract_line_items_subtotal(text_lower)
        }

        return results

    def _extract_invoice_number(self, text: str) -> Dict:
        """Extract invoice number with confidence"""
        for pattern in self.invoice_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                # Take the first match, clean it up
                invoice_num = matches[0].strip().upper()
                confidence = 0.85 if 'invoice' in pattern else 0.75

                return {
                    'value': invoice_num,
                    'confidence': confidence,
                    'method': 'rule_based_pattern'
                }

        # Fallback: look for standalone alphanumeric codes
        fallback_matches = re.findall(r'\b[A-Z0-9]{3,}-?[0-9]+\b', text.upper())
        if fallback_matches:
            return {
                'value': fallback_matches[0],
                'confidence': 0.60,
                'method': 'fallback_pattern'
            }

        return {'value': '', 'confidence': 0.0, 'method': 'not_found'}

    def _extract_date(self, text: str) -> Dict:
        """Extract invoice date with confidence"""
        for pattern in self.date_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                date_str = matches[0].strip()
                confidence = 0.85 if 'date' in pattern else 0.75

                # Try to standardize the date format
                standardized_date = self._standardize_date(date_str)

                return {
                    'value': standardized_date,
                    'confidence': confidence,
                    'method': 'rule_based_pattern'
                }

        return {'value': '', 'confidence': 0.0, 'method': 'not_found'}

    def _extract_supplier(self, text: str) -> Dict:
        """Extract supplier name with confidence"""
        lines = text.split('\n')

        # Look for supplier in first few lines (common invoice format)
        for i, line in enumerate(lines[:5]):
            line_clean = line.strip()
            if len(line_clean) > 3 and not re.match(r'^\d+', line_clean):
                # Skip obvious non-supplier lines
                skip_keywords = ['invoice', 'bill', 'receipt', 'date', 'total', 'address']
                if not any(keyword in line_clean.lower() for keyword in skip_keywords):
                    return {
                        'value': line_clean[:50],  # Limit length
                        'confidence': 0.80 if i == 0 else 0.70,
                        'method': 'position_based'
                    }

        return {'value': '', 'confidence': 0.0, 'method': 'not_found'}

    def _extract_customer(self, text: str) -> Dict:
        """Extract customer name (simplified)"""
        # Look for "bill to", "customer", etc.
        customer_patterns = [
            r'(?:bill[\s]*to|customer|client)[\s:]*([a-z\s&,.-]+?)(?:\n|address)',
        ]

        for pattern in customer_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            if matches:
                customer = matches[0].strip()[:50]
                return {
                    'value': customer,
                    'confidence': 0.75,
                    'method': 'pattern_based'
                }

        return {'value': '', 'confidence': 0.0, 'method': 'not_found'}

    def _extract_total_amount(self, text: str) -> Dict:
        """Extract total amount with confidence"""
        for pattern in self.amount_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                amount_str = matches[0].replace(',', '')
                try:
                    amount = float(amount_str)
                    confidence = 0.85 if 'total' in pattern else 0.75

                    return {
                        'value': amount,
                        'confidence': confidence,
                        'method': 'rule_based_pattern'
                    }
                except ValueError:
                    continue

        return {'value': 0.0, 'confidence': 0.0, 'method': 'not_found'}

    def _extract_subtotal(self, text: str) -> Dict:
        """Extract subtotal amount"""
        subtotal_patterns = [
            r'(?:subtotal|sub[\s]*total)[\s:$]*([0-9,]+\.?[0-9]*)',
            r'(?:net[\s]*amount|before[\s]*tax)[\s:$]*([0-9,]+\.?[0-9]*)'
        ]

        for pattern in subtotal_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                amount_str = matches[0].replace(',', '')
                try:
                    amount = float(amount_str)
                    return {
                        'value': amount,
                        'confidence': 0.80,
                        'method': 'rule_based_pattern'
                    }
                except ValueError:
                    continue

        return {'value': 0.0, 'confidence': 0.0, 'method': 'not_found'}

    def _extract_vat_amount(self, text: str) -> Dict:
        """Extract VAT/tax amount"""
        for pattern in self.vat_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                amount_str = matches[0].replace(',', '')
                try:
                    amount = float(amount_str)
                    return {
                        'value': amount,
                        'confidence': 0.80,
                        'method': 'rule_based_pattern'
                    }
                except ValueError:
                    continue

        return {'value': 0.0, 'confidence': 0.0, 'method': 'not_found'}

    def _extract_line_items_count(self, text: str) -> Dict:
        """Extract number of line items (basic implementation)"""
        return {'value': 0, 'confidence': 0.0, 'method': 'not_implemented'}

    def _extract_line_items_subtotal(self, text: str) -> Dict:
        """Extract line items subtotal (basic implementation)"""
        return {'value': 0.0, 'confidence': 0.0, 'method': 'not_implemented'}

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text for better extraction"""
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text

    def _standardize_date(self, date_str: str) -> str:
        """Standardize date format"""
        try:
            # Try different date formats
            formats = ['%m/%d/%Y', '%d/%m/%Y', '%Y-%m-%d', '%m-%d-%Y']

            for fmt in formats:
                try:
                    parsed_date = datetime.strptime(date_str, fmt)
                    return parsed_date.strftime('%m/%d/%Y')
                except ValueError:
                    continue

            # If parsing fails, return original
            return date_str

        except Exception:
            return date_str


def calculate_field_confidence_score(extracted_fields: Dict) -> float:
    """Calculate overall confidence score for all extracted fields"""
    confidences = []

    for field_name, field_data in extracted_fields.items():
        if isinstance(field_data, dict) and 'confidence' in field_data:
            confidences.append(field_data['confidence'])

    if confidences:
        return sum(confidences) / len(confidences)
    else:
        return 0.0