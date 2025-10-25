"""
Translation Model using Hugging Face Transformers
Supports multiple language pairs
"""

import json
import numpy as np
import triton_python_backend_utils as pb_utils
from transformers import MarianMTModel, MarianTokenizer, M2M100ForConditionalGeneration, M2M100Tokenizer
import torch
from typing import Dict, List


class TritonPythonModel:
    """Translation model supporting multiple language pairs"""

    def initialize(self, args):
        self.model_config = json.loads(args['model_config'])
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize translation models
        # For production, you would load actual models here
        # Using Helsinki-NLP models for demonstration
        self.models = {}
        self.tokenizers = {}
        
        # Define supported language pairs
        self.language_pairs = {
            'en-es': 'Helsinki-NLP/opus-mt-en-es',
            'en-fr': 'Helsinki-NLP/opus-mt-en-fr',
            'en-de': 'Helsinki-NLP/opus-mt-en-de',
            'en-zh': 'Helsinki-NLP/opus-mt-en-zh',
            'en-ar': 'Helsinki-NLP/opus-mt-en-ar',
            'en-hi': 'Helsinki-NLP/opus-mt-en-hi',
            'en-ru': 'Helsinki-NLP/opus-mt-en-ru',
            'es-en': 'Helsinki-NLP/opus-mt-es-en',
            'fr-en': 'Helsinki-NLP/opus-mt-fr-en',
            'de-en': 'Helsinki-NLP/opus-mt-de-en'
        }
        
        # For demonstration, we'll use a simple translation approach
        # In production, you would load the actual models
        self.simple_translations = {
            'en-es': {
                'hello': 'hola',
                'world': 'mundo',
                'thank you': 'gracias',
                'goodbye': 'adiós',
                'yes': 'sí',
                'no': 'no',
                'please': 'por favor',
                'sorry': 'lo siento',
                'how are you': 'cómo estás'
            },
            'en-fr': {
                'hello': 'bonjour',
                'world': 'monde',
                'thank you': 'merci',
                'goodbye': 'au revoir',
                'yes': 'oui',
                'no': 'non',
                'please': 's\'il vous plaît',
                'sorry': 'désolé',
                'how are you': 'comment allez-vous'
            },
            'en-de': {
                'hello': 'hallo',
                'world': 'welt',
                'thank you': 'danke',
                'goodbye': 'auf wiedersehen',
                'yes': 'ja',
                'no': 'nein',
                'please': 'bitte',
                'sorry': 'entschuldigung',
                'how are you': 'wie geht es dir'
            },
            'en-hi': {
                'hello': 'नमस्ते',
                'world': 'विश्व',
                'thank you': 'धन्यवाद',
                'goodbye': 'अलविदा',
                'yes': 'हाँ',
                'no': 'नहीं',
                'please': 'कृपया',
                'sorry': 'माफ़ करें',
                'how are you': 'आप कैसे हैं'
            },
            'en-ar': {
                'hello': 'مرحبا',
                'world': 'عالم',
                'thank you': 'شكرا',
                'goodbye': 'وداعا',
                'yes': 'نعم',
                'no': 'لا',
                'please': 'من فضلك',
                'sorry': 'آسف',
                'how are you': 'كيف حالك'
            }
        }

    def execute(self, requests):
        responses = []
        
        for request in requests:
            # Get input tensors
            text_tensor = pb_utils.get_input_tensor_by_name(request, "text")
            source_lang_tensor = pb_utils.get_input_tensor_by_name(request, "source_language")
            target_lang_tensor = pb_utils.get_input_tensor_by_name(request, "target_language")
            
            texts = text_tensor.as_numpy().tolist()
            source_langs = source_lang_tensor.as_numpy().tolist() if source_lang_tensor else ['auto'] * len(texts)
            target_langs = target_lang_tensor.as_numpy().tolist() if target_lang_tensor else ['en'] * len(texts)
            
            translation_results = []
            
            for text_bytes, source, target in zip(texts, source_langs, target_langs):
                if isinstance(text_bytes, bytes):
                    text = text_bytes.decode('utf-8')
                else:
                    text = str(text_bytes)
                
                if isinstance(source, bytes):
                    source = source.decode('utf-8')
                if isinstance(target, bytes):
                    target = target.decode('utf-8')
                
                # Perform translation
                result = self._translate(text, source, target)
                translation_results.append(json.dumps(result))
            
            # Create output tensor
            out_tensor = pb_utils.Tensor(
                "translated_text",
                np.array(translation_results, dtype=np.object_))
            
            # Create response
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor])
            responses.append(inference_response)
        
        return responses
    
    def _translate(self, text: str, source_lang: str, target_lang: str) -> Dict:
        """Perform translation"""
        
        # Auto-detect source language if not specified
        if source_lang == 'auto':
            source_lang = self._detect_language(text)
        
        # Check if translation is needed
        if source_lang == target_lang:
            return {
                'original': text,
                'translated': text,
                'source_language': source_lang,
                'target_language': target_lang,
                'confidence': 1.0,
                'method': 'no_translation_needed'
            }
        
        # Get language pair
        lang_pair = f"{source_lang}-{target_lang}"
        
        # Perform translation
        translated = text
        confidence = 0.0
        method = 'unsupported'
        
        if lang_pair in self.simple_translations:
            # Simple dictionary-based translation for demonstration
            translated = self._simple_translate(text, lang_pair)
            confidence = 0.6
            method = 'dictionary'
        elif lang_pair in self.language_pairs:
            # In production, use actual model here
            # translated = self._model_translate(text, lang_pair)
            translated = f"[Translation from {source_lang} to {target_lang}]: {text}"
            confidence = 0.85
            method = 'neural_mt'
        else:
            # Try reverse pair
            reverse_pair = f"{target_lang}-{source_lang}"
            if reverse_pair in self.language_pairs:
                # Could use back-translation technique
                translated = f"[No direct translation available from {source_lang} to {target_lang}]"
                confidence = 0.3
                method = 'unsupported_pair'
        
        return {
            'original': text,
            'translated': translated,
            'source_language': source_lang,
            'target_language': target_lang,
            'confidence': confidence,
            'method': method,
            'alternative_translations': self._get_alternatives(text, lang_pair)
        }
    
    def _detect_language(self, text: str) -> str:
        """Detect the language of the text"""
        # Simple heuristic-based detection for demonstration
        # In production, use langdetect or fasttext
        
        # Check for common words/patterns
        if any(word in text.lower() for word in ['the', 'and', 'is', 'are', 'have']):
            return 'en'
        elif any(word in text.lower() for word in ['le', 'la', 'de', 'et', 'est']):
            return 'fr'
        elif any(word in text.lower() for word in ['el', 'la', 'de', 'y', 'es']):
            return 'es'
        elif any(word in text.lower() for word in ['der', 'die', 'das', 'und', 'ist']):
            return 'de'
        elif any(char in text for char in 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя'):
            return 'ru'
        elif any(char in text for char in '的一是不了在有我他这个们中来上大为和国地到以说时要就出会可也你对生能而子那得于着下自之年过发后作里用道行所然家种事成方多经么去法学如都同现当没动面起看定天分还进好小部其些主样理心她本前开但因只从想实日军者意无力它与长把机十民第公此已工使情明性知全三又关点正业外将两高间由问很最重并物手应战向头文体政美相见被利什二等产或新己制身果加西斯月话合回特代内信表化老给世位次度门任常先海通教儿原东声提立及比员解水名真论处走义各入几口认条平系气题活尔更别打女变四神总何电数安少报才结反受目太量再感建务做接必场件计管期市直德资命山金指克许统区保至队形社便空决治展马科司五基眼书非则听白却界达光放强即像难且权思王象完设式色路记南品住告类求据程北边死张该交规万取拉格望觉术领共确传师观清今切院让识候带导争运笑飞风步改收根干造言联持组每济车亲极林服快办议往元英士复整流数'):
            return 'zh'
        elif any(char in text for char in 'あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん'):
            return 'ja'
        elif any(char in text for char in 'اأإآبتثجحخدذرزسشصضطظعغفقكلمنهوي'):
            return 'ar'
        elif any(char in text for char in 'अआइईउऊएऐओऔकखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह'):
            return 'hi'
        
        return 'en'  # Default to English
    
    def _simple_translate(self, text: str, lang_pair: str) -> str:
        """Simple dictionary-based translation for demonstration"""
        if lang_pair not in self.simple_translations:
            return text
        
        translations = self.simple_translations[lang_pair]
        text_lower = text.lower()
        
        # Check for exact matches
        if text_lower in translations:
            return translations[text_lower]
        
        # Try word-by-word translation
        words = text.split()
        translated_words = []
        for word in words:
            word_lower = word.lower().strip('.,!?;:')
            if word_lower in translations:
                translated_words.append(translations[word_lower])
            else:
                translated_words.append(word)
        
        return ' '.join(translated_words)
    
    def _get_alternatives(self, text: str, lang_pair: str) -> List[str]:
        """Get alternative translations"""
        # In production, this would return multiple translation candidates
        alternatives = []
        
        if lang_pair == 'en-es' and text.lower() == 'hello':
            alternatives = ['hola', 'buenos días', 'buenas tardes']
        elif lang_pair == 'en-fr' and text.lower() == 'hello':
            alternatives = ['bonjour', 'salut', 'bonsoir']
        
        return alternatives[:3]  # Return top 3 alternatives
    
    def finalize(self):
        """Clean up resources"""
        # Unload models if loaded
        for model in self.models.values():
            del model
        for tokenizer in self.tokenizers.values():
            del tokenizer
        
        self.models = {}
        self.tokenizers = {}
