"""
Data Type Detection Model
Detects various data types like phone numbers, passports, emails, etc.
"""

import json
import numpy as np
import triton_python_backend_utils as pb_utils
import re
from typing import List, Dict, Any
import phonenumbers
from datetime import datetime


class TritonPythonModel:
    """Data type detection model using regex patterns and validation"""

    def initialize(self, args):
        self.model_config = json.loads(args['model_config'])
        
        # Initialize patterns for various data types
        self.patterns = {
            'email': {
                'pattern': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
                'confidence': 0.95
            },
            'url': {
                'pattern': r'https?://(?:[-\w.])+(?::\d+)?(?:[/\w\-._~:/?#[\]@!$&\'()*+,;=.]+)?',
                'confidence': 0.9
            },
            'ipv4': {
                'pattern': r'^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$',
                'confidence': 0.95
            },
            'ipv6': {
                'pattern': r'^(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$',
                'confidence': 0.95
            },
            'mac_address': {
                'pattern': r'^([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})$',
                'confidence': 0.95
            },
            'credit_card': {
                'pattern': r'^(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3(?:0[0-5]|[68][0-9])[0-9]{11}|6(?:011|5[0-9]{2})[0-9]{12})$',
                'confidence': 0.85
            },
            'ssn_us': {
                'pattern': r'^\d{3}-\d{2}-\d{4}$',
                'confidence': 0.9
            },
            'date_iso': {
                'pattern': r'^\d{4}-\d{2}-\d{2}$',
                'confidence': 0.85
            },
            'datetime_iso': {
                'pattern': r'^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}',
                'confidence': 0.85
            },
            'uuid': {
                'pattern': r'^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$',
                'confidence': 0.95
            },
            'hex_color': {
                'pattern': r'^#(?:[0-9a-fA-F]{3}){1,2}$',
                'confidence': 0.9
            },
            'bitcoin_address': {
                'pattern': r'^[13][a-km-zA-HJ-NP-Z1-9]{25,34}$',
                'confidence': 0.85
            },
            'ethereum_address': {
                'pattern': r'^0x[a-fA-F0-9]{40}$',
                'confidence': 0.9
            }
        }
        
        # Passport patterns for different countries
        self.passport_patterns = {
            'uk': {
                'pattern': r'^[0-9]{9}$|^[A-Z]{2}[0-9]{6}$',
                'name': 'British Passport',
                'confidence': 0.8
            },
            'us': {
                'pattern': r'^[A-Z][0-9]{8}$|^[0-9]{9}$',
                'name': 'US Passport',
                'confidence': 0.8
            },
            'canada': {
                'pattern': r'^[A-Z]{2}[0-9]{6}$',
                'name': 'Canadian Passport',
                'confidence': 0.8
            },
            'australia': {
                'pattern': r'^[A-Z][0-9]{7}$|^[A-Z]{2}[0-9]{6}$',
                'name': 'Australian Passport',
                'confidence': 0.8
            },
            'india': {
                'pattern': r'^[A-Z][0-9]{7}$',
                'name': 'Indian Passport',
                'confidence': 0.8
            },
            'germany': {
                'pattern': r'^[A-Z][0-9A-Z]{8}$',
                'name': 'German Passport',
                'confidence': 0.8
            },
            'france': {
                'pattern': r'^[0-9]{2}[A-Z]{2}[0-9]{5}$',
                'name': 'French Passport',
                'confidence': 0.8
            }
        }
        
        # Driver's license patterns
        self.license_patterns = {
            'uk_driving': {
                'pattern': r'^[A-Z]{2,5}[0-9]{6}[A-Z]{2}[0-9A-Z]{3}$',
                'name': 'UK Driving License',
                'confidence': 0.75
            },
            'us_driving_ca': {
                'pattern': r'^[A-Z][0-9]{7}$',
                'name': 'California Driver License',
                'confidence': 0.7
            }
        }
        
        # National ID patterns
        self.national_id_patterns = {
            'uk_nin': {
                'pattern': r'^[A-Z]{2}[0-9]{6}[A-Z]$',
                'name': 'UK National Insurance Number',
                'confidence': 0.85
            },
            'canada_sin': {
                'pattern': r'^\d{3}-\d{3}-\d{3}$',
                'name': 'Canadian SIN',
                'confidence': 0.85
            }
        }

    def execute(self, requests):
        responses = []
        
        for request in requests:
            # Get input tensor
            in_0 = pb_utils.get_input_tensor_by_name(request, "text")
            texts = in_0.as_numpy().tolist()
            
            detection_results = []
            
            for text_bytes in texts:
                if isinstance(text_bytes, bytes):
                    text = text_bytes.decode('utf-8')
                else:
                    text = str(text_bytes)
                
                # Detect data types
                result = self._detect_data_types(text)
                detection_results.append(json.dumps(result))
            
            # Create output tensor
            out_tensor = pb_utils.Tensor(
                "detection_result",
                np.array(detection_results, dtype=np.object_))
            
            # Create response
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor])
            responses.append(inference_response)
        
        return responses
    
    def _detect_data_types(self, text: str) -> Dict[str, Any]:
        """Detect various data types in the text"""
        text = text.strip()
        detections = []
        
        # Check for phone numbers first (special handling)
        phone_result = self._detect_phone_number(text)
        if phone_result:
            detections.append(phone_result)
        
        # Check general patterns
        for dtype, config in self.patterns.items():
            if re.match(config['pattern'], text, re.IGNORECASE):
                detections.append({
                    'type': dtype,
                    'confidence': config['confidence'],
                    'value': text,
                    'category': 'identifier'
                })
        
        # Check passport patterns
        for country, config in self.passport_patterns.items():
            if re.match(config['pattern'], text.replace(' ', ''), re.IGNORECASE):
                detections.append({
                    'type': 'passport',
                    'subtype': config['name'],
                    'country': country,
                    'confidence': config['confidence'],
                    'value': text,
                    'category': 'document'
                })
        
        # Check driver's license patterns
        for license_type, config in self.license_patterns.items():
            if re.match(config['pattern'], text.replace(' ', ''), re.IGNORECASE):
                detections.append({
                    'type': 'driving_license',
                    'subtype': config['name'],
                    'confidence': config['confidence'],
                    'value': text,
                    'category': 'document'
                })
        
        # Check national ID patterns
        for id_type, config in self.national_id_patterns.items():
            if re.match(config['pattern'], text.replace(' ', ''), re.IGNORECASE):
                detections.append({
                    'type': 'national_id',
                    'subtype': config['name'],
                    'confidence': config['confidence'],
                    'value': text,
                    'category': 'document'
                })
        
        # Check for credit card with Luhn validation
        if self._is_valid_credit_card(text):
            detections.append({
                'type': 'credit_card',
                'confidence': 0.95,
                'value': self._mask_credit_card(text),
                'category': 'financial'
            })
        
        # Check for IBAN
        iban_result = self._detect_iban(text)
        if iban_result:
            detections.append(iban_result)
        
        # If no specific type detected, try to classify general content
        if not detections:
            detections.append(self._classify_general_text(text))
        
        return {
            'text': text,
            'detections': detections,
            'primary_type': detections[0]['type'] if detections else 'unknown',
            'confidence': detections[0]['confidence'] if detections else 0.0
        }
    
    def _detect_phone_number(self, text: str) -> Dict[str, Any]:
        """Detect and validate phone numbers"""
        try:
            # Try to parse with country code
            if text.startswith('+'):
                parsed = phonenumbers.parse(text, None)
            else:
                # Try common formats
                for country in ['US', 'GB', 'CA', 'AU', 'IN', 'DE', 'FR']:
                    try:
                        parsed = phonenumbers.parse(text, country)
                        if phonenumbers.is_valid_number(parsed):
                            return {
                                'type': 'phone_number',
                                'country': country,
                                'international': phonenumbers.format_number(
                                    parsed, phonenumbers.PhoneNumberFormat.INTERNATIONAL),
                                'confidence': 0.95,
                                'value': text,
                                'category': 'contact'
                            }
                    except:
                        continue
            
            if phonenumbers.is_valid_number(parsed):
                return {
                    'type': 'phone_number',
                    'international': phonenumbers.format_number(
                        parsed, phonenumbers.PhoneNumberFormat.INTERNATIONAL),
                    'confidence': 0.95,
                    'value': text,
                    'category': 'contact'
                }
        except:
            # Check if it looks like a phone number
            phone_pattern = r'^[\+]?[(]?[0-9]{1,4}[)]?[-\s\.]?[(]?[0-9]{1,4}[)]?[-\s\.]?[0-9]{1,5}[-\s\.]?[0-9]{1,5}$'
            if re.match(phone_pattern, text):
                return {
                    'type': 'phone_number',
                    'confidence': 0.6,
                    'value': text,
                    'category': 'contact',
                    'note': 'Pattern match only, not validated'
                }
        
        return None
    
    def _is_valid_credit_card(self, number: str) -> bool:
        """Validate credit card using Luhn algorithm"""
        number = re.sub(r'\D', '', number)
        if len(number) < 13 or len(number) > 19:
            return False
        
        # Luhn algorithm
        def luhn_checksum(card_number):
            def digits_of(n):
                return [int(d) for d in str(n)]
            
            digits = digits_of(card_number)
            odd_digits = digits[-1::-2]
            even_digits = digits[-2::-2]
            
            checksum = sum(odd_digits)
            for d in even_digits:
                checksum += sum(digits_of(d * 2))
            
            return checksum % 10
        
        return luhn_checksum(number) == 0
    
    def _mask_credit_card(self, number: str) -> str:
        """Mask credit card number for security"""
        clean_number = re.sub(r'\D', '', number)
        if len(clean_number) >= 12:
            return f"****-****-****-{clean_number[-4:]}"
        return number
    
    def _detect_iban(self, text: str) -> Dict[str, Any]:
        """Detect IBAN (International Bank Account Number)"""
        iban_pattern = r'^[A-Z]{2}[0-9]{2}[A-Z0-9]{11,30}$'
        clean_text = text.replace(' ', '').upper()
        
        if re.match(iban_pattern, clean_text):
            return {
                'type': 'iban',
                'country_code': clean_text[:2],
                'confidence': 0.85,
                'value': text,
                'category': 'financial'
            }
        return None
    
    def _classify_general_text(self, text: str) -> Dict[str, Any]:
        """Classify general text when no specific pattern matches"""
        # Simple heuristics for general classification
        if text.replace('.', '').replace(',', '').replace('-', '').isdigit():
            return {
                'type': 'number',
                'confidence': 0.9,
                'value': text,
                'category': 'numeric'
            }
        
        if len(text.split()) == 1 and text.isalpha():
            return {
                'type': 'single_word',
                'confidence': 0.8,
                'value': text,
                'category': 'text'
            }
        
        if len(text.split()) > 5:
            return {
                'type': 'sentence',
                'confidence': 0.7,
                'value': text,
                'category': 'text'
            }
        
        return {
            'type': 'text',
            'confidence': 0.5,
            'value': text,
            'category': 'general'
        }
    
    def finalize(self):
        pass
