import requests
import json
import base64
from pathlib import Path
from typing import Dict, List, Any, Optional
import cv2
import numpy as np
from PIL import Image
import pandas as pd

class AdvancedInvoiceOCR:
    """OCR avancé pour factures avec préservation de la structure"""
    
    def __init__(self):
        self.ocr_engines = {
            'tesseract': self._setup_tesseract,
            'paddleocr': self._setup_paddleocr,
            'easyocr': self._setup_easyocr,
            'azure_form_recognizer': self._setup_azure,
            'google_document_ai': self._setup_google,
            'aws_textract': self._setup_aws
        }
    
    # 1. TESSERACT AVEC PRÉSERVATION DE STRUCTURE
    def _setup_tesseract(self):
        """Configuration Tesseract pour structure"""
        try:
            import pytesseract
            from pytesseract import Output
            return pytesseract, Output
        except ImportError:
            raise Exception("pip install pytesseract")
    
    def extract_with_tesseract_structure(self, image_path: str) -> Dict[str, Any]:
        """
        Tesseract avec préservation de la structure des tableaux
        """
        pytesseract, Output = self._setup_tesseract()
        
        # Préprocessing de l'image
        image = cv2.imread(image_path)
        processed_img = self._preprocess_invoice_image(image)
        
        # Configuration Tesseract pour préserver la structure
        custom_config = r'--oem 3 --psm 4 -c preserve_interword_spaces=1'
        
        # Extraction avec données de position
        data = pytesseract.image_to_data(
            processed_img, 
            output_type=Output.DICT,
            config=custom_config,
            lang='fra+eng'
        )
        
        # Reconstruction de la structure
        structured_data = self._reconstruct_structure_from_tesseract(data)
        
        return {
            'engine': 'tesseract',
            'structured_text': structured_data,
            'tables': self._extract_tables_from_structure(structured_data),
            'key_value_pairs': self._extract_key_value_pairs(structured_data)
        }
    
    # 2. PADDLEOCR - EXCELLENT POUR LA STRUCTURE
    def _setup_paddleocr(self):
        """Configuration PaddleOCR"""
        try:
            from paddleocr import PaddleOCR
            return PaddleOCR(use_angle_cls=True, lang='fr', show_log=False)
        except ImportError:
            raise Exception("pip install paddleocr")
    
    def extract_with_paddleocr(self, image_path: str) -> Dict[str, Any]:
        """
        PaddleOCR avec détection de structure avancée
        """
        ocr = self._setup_paddleocr()
        
        # Extraction avec coordonnées
        result = ocr.ocr(image_path, cls=True)
        
        # Conversion en format structuré
        structured_data = []
        for line in result[0]:  # result[0] car une seule image
            bbox, (text, confidence) = line
            structured_data.append({
                'text': text,
                'confidence': confidence,
                'bbox': bbox,
                'center': self._calculate_center(bbox)
            })
        
        # Tri par position (ligne puis colonne)
        structured_data.sort(key=lambda x: (x['center'][1], x['center'][0]))
        
        return {
            'engine': 'paddleocr',
            'structured_text': structured_data,
            'tables': self._detect_tables_paddleocr(structured_data),
            'key_value_pairs': self._extract_kv_pairs_paddleocr(structured_data)
        }
    
    # 3. EASYOCR - BON POUR MULTI-LANGUES
    def _setup_easyocr(self):
        """Configuration EasyOCR"""
        try:
            import easyocr
            return easyocr.Reader(['fr', 'en'])
        except ImportError:
            raise Exception("pip install easyocr")
    
    def extract_with_easyocr(self, image_path: str) -> Dict[str, Any]:
        """EasyOCR avec structure"""
        reader = self._setup_easyocr()
        
        # Extraction
        results = reader.readtext(image_path, detail=1)
        
        structured_data = []
        for (bbox, text, confidence) in results:
            structured_data.append({
                'text': text,
                'confidence': confidence,
                'bbox': bbox,
                'center': self._calculate_center(bbox)
            })
        
        # Tri par position
        structured_data.sort(key=lambda x: (x['center'][1], x['center'][0]))
        
        return {
            'engine': 'easyocr',
            'structured_text': structured_data,
            'tables': self._detect_tables_easyocr(structured_data),
            'key_value_pairs': self._extract_kv_pairs_easyocr(structured_data)
        }
    
    # 4. AZURE FORM RECOGNIZER - LE MEILLEUR POUR LES FACTURES
    def _setup_azure(self):
        """Configuration Azure Form Recognizer"""
        # Remplacez par vos vraies clés
        self.azure_endpoint = "https://your-resource.cognitiveservices.azure.com/"
        self.azure_key = "your-api-key"
        return True
    
    def extract_with_azure_form_recognizer(self, image_path: str) -> Dict[str, Any]:
        """
        Azure Form Recognizer - Spécialisé pour les factures
        """
        self._setup_azure()
        
        # Lire l'image
        with open(image_path, 'rb') as f:
            image_data = f.read()
        
        # Headers pour l'API
        headers = {
            'Ocp-Apim-Subscription-Key': self.azure_key,
            'Content-Type': 'application/octet-stream'
        }
        
        # URL pour l'analyse de factures
        analyze_url = f"{self.azure_endpoint}formrecognizer/documentModels/prebuilt-invoice:analyze?api-version=2023-07-31"
        
        # Envoi pour analyse
        response = requests.post(analyze_url, headers=headers, data=image_data)
        
        if response.status_code == 202:
            # Récupérer l'URL de résultat
            operation_location = response.headers.get('Operation-Location')
            
            # Attendre le résultat
            import time
            while True:
                result_response = requests.get(
                    operation_location,
                    headers={'Ocp-Apim-Subscription-Key': self.azure_key}
                )
                result = result_response.json()
                
                if result['status'] == 'succeeded':
                    break
                elif result['status'] == 'failed':
                    raise Exception("Analyse échouée")
                
                time.sleep(2)
            
            return self._parse_azure_invoice_result(result)
        
        else:
            raise Exception(f"Erreur Azure: {response.status_code}")
    
    # 5. GOOGLE DOCUMENT AI - TRÈS PERFORMANT
    def extract_with_google_document_ai(self, image_path: str) -> Dict[str, Any]:
        """
        Google Document AI pour factures
        """
        try:
            from google.cloud import documentai
        except ImportError:
            raise Exception("pip install google-cloud-documentai")
        
        # Configuration
        project_id = "your-project-id"
        location = "us"  # ou "eu"
        processor_id = "your-processor-id"
        
        client = documentai.DocumentProcessorServiceClient()
        
        # Le nom complet du processeur
        name = f"projects/{project_id}/locations/{location}/processors/{processor_id}"
        
        # Lire l'image
        with open(image_path, "rb") as image:
            image_content = image.read()
        
        # Configuration de la requête
        raw_document = documentai.RawDocument(
            content=image_content,
            mime_type="image/jpeg"  # ou image/png
        )
        
        request = documentai.ProcessRequest(
            name=name,
            raw_document=raw_document
        )
        
        # Traitement
        result = client.process_document(request=request)
        document = result.document
        
        return self._parse_google_document_ai_result(document)
    
    # 6. AWS TEXTRACT - EXCELLENT POUR LES TABLEAUX
    def extract_with_aws_textract(self, image_path: str) -> Dict[str, Any]:
        """
        AWS Textract avec analyse de tableaux et formulaires
        """
        try:
            import boto3
        except ImportError:
            raise Exception("pip install boto3")
        
        # Client Textract
        textract = boto3.client('textract', region_name='us-east-1')
        
        # Lire l'image
        with open(image_path, 'rb') as document:
            img_bytes = document.read()
        
        # Analyse avec tableaux et formulaires
        response = textract.analyze_document(
            Document={'Bytes': img_bytes},
            FeatureTypes=['TABLES', 'FORMS']
        )
        
        return self._parse_aws_textract_result(response)
    
    # MÉTHODES UTILITAIRES
    def _setup_google(self):
        """Configuration Google Document AI (optionnelle)"""
        return True

    def _setup_aws(self):
        """Configuration Google Document AI (optionnelle)"""
        return True

    def _preprocess_invoice_image(self, image):
        """Préprocessing spécialisé pour factures"""
        # Conversion en niveaux de gris
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Réduction du bruit
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Amélioration du contraste
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # Binarisation adaptative
        binary = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        return binary
    
    def _calculate_center(self, bbox):
        """Calcule le centre d'une bounding box"""
        if len(bbox) == 4 and len(bbox[0]) == 2:  # Format PaddleOCR
            x_coords = [point[0] for point in bbox]
            y_coords = [point[1] for point in bbox]
            return (sum(x_coords) / 4, sum(y_coords) / 4)
        return (0, 0)
    
    def _detect_tables_paddleocr(self, structured_data: List[Dict]) -> List[Dict]:
        """Détection de tableaux dans les données PaddleOCR"""
        tables = []
        
        # Grouper les éléments par ligne (Y similaire)
        lines = {}
        for item in structured_data:
            y = int(item['center'][1] / 20) * 20  # Grouper par tranches de 20px
            if y not in lines:
                lines[y] = []
            lines[y].append(item)
        
        # Détecter les lignes de tableau (plusieurs colonnes alignées)
        table_lines = []
        for y, items in lines.items():
            if len(items) >= 3:  # Au moins 3 colonnes
                # Trier par X
                items.sort(key=lambda x: x['center'][0])
                table_lines.append({
                    'y': y,
                    'items': items,
                    'columns': len(items)
                })
        
        # Grouper les lignes consécutives en tableaux
        if table_lines:
            current_table = [table_lines[0]]
            
            for i in range(1, len(table_lines)):
                if abs(table_lines[i]['y'] - table_lines[i-1]['y']) < 100:
                    current_table.append(table_lines[i])
                else:
                    if len(current_table) >= 2:
                        tables.append(self._format_table(current_table))
                    current_table = [table_lines[i]]
            
            if len(current_table) >= 2:
                tables.append(self._format_table(current_table))
        
        return tables
    
    def _format_table(self, table_lines: List[Dict]) -> Dict:
        """Formate les lignes de tableau en structure tabulaire"""
        # Déterminer le nombre de colonnes
        max_cols = max(line['columns'] for line in table_lines)
        
        # Créer la matrice du tableau
        table_data = []
        for line in table_lines:
            row = [item['text'] for item in line['items']]
            # Compléter avec des cellules vides si nécessaire
            while len(row) < max_cols:
                row.append('')
            table_data.append(row)
        
        return {
            'type': 'table',
            'rows': len(table_data),
            'columns': max_cols,
            'data': table_data,
            'dataframe': pd.DataFrame(table_data[1:], columns=table_data[0]) if table_data else None
        }
    
    def _extract_kv_pairs_paddleocr(self, structured_data: List[Dict]) -> Dict[str, str]:
        """Extraction de paires clé-valeur"""
        kv_pairs = {}
        
        # Mots-clés de factures
        invoice_keywords = [
            'numéro', 'number', 'facture', 'invoice', 'date', 'total', 
            'montant', 'amount', 'tva', 'tax', 'client', 'customer',
            'fournisseur', 'vendor', 'supplier'
        ]
        
        for i, item in enumerate(structured_data):
            text = item['text'].lower().strip()
            
            # Chercher les mots-clés
            for keyword in invoice_keywords:
                if keyword in text:
                    # Chercher la valeur à proximité
                    value = self._find_nearby_value(structured_data, i, item['center'])
                    if value:
                        kv_pairs[keyword] = value
                        break
        
        return kv_pairs
    
    def _find_nearby_value(self, data: List[Dict], current_idx: int, center: tuple) -> Optional[str]:
        """Trouve la valeur la plus proche d'une clé"""
        min_distance = float('inf')
        closest_value = None
        
        for i, item in enumerate(data):
            if i == current_idx:
                continue
            
            # Calculer la distance
            distance = ((item['center'][0] - center[0]) ** 2 + 
                       (item['center'][1] - center[1]) ** 2) ** 0.5
            
            # Privilégier les éléments à droite ou en dessous
            if (item['center'][0] > center[0] or item['center'][1] > center[1]) and distance < min_distance:
                min_distance = distance
                closest_value = item['text']
        
        return closest_value
    
    # Méthodes pour les autres engines (similaires)
    def _detect_tables_easyocr(self, structured_data):
        return self._detect_tables_paddleocr(structured_data)
    
    def _extract_kv_pairs_easyocr(self, structured_data):
        return self._extract_kv_pairs_paddleocr(structured_data)
    
    def _reconstruct_structure_from_tesseract(self, data):
        """Reconstruit la structure à partir des données Tesseract"""
        structured = []
        n_boxes = len(data['text'])
        
        for i in range(n_boxes):
            if int(data['conf'][i]) > 30:  # Seuil de confiance
                structured.append({
                    'text': data['text'][i],
                    'confidence': data['conf'][i],
                    'bbox': (data['left'][i], data['top'][i], 
                            data['width'][i], data['height'][i]),
                    'center': (data['left'][i] + data['width'][i]//2,
                              data['top'][i] + data['height'][i]//2)
                })
        
        return structured
    
    def _extract_tables_from_structure(self, structured_data):
        return self._detect_tables_paddleocr(structured_data)
    
    def _extract_key_value_pairs(self, structured_data):
        return self._extract_kv_pairs_paddleocr(structured_data)
    
    # MÉTHODE PRINCIPALE
    def extract_invoice_structure(self, image_path: str, engine: str = 'paddleocr') -> Dict[str, Any]:
        """
        Méthode principale pour extraire la structure d'une facture
        
        Args:
            image_path: Chemin vers l'image
            engine: Engine OCR à utiliser
        """
        if engine == 'tesseract':
            return self.extract_with_tesseract_structure(image_path)
        elif engine == 'paddleocr':
            return self.extract_with_paddleocr(image_path)
        elif engine == 'easyocr':
            return self.extract_with_easyocr(image_path)
        elif engine == 'azure':
            return self.extract_with_azure_form_recognizer(image_path)
        elif engine == 'google':
            return self.extract_with_google_document_ai(image_path)
        elif engine == 'aws':
            return self.extract_with_aws_textract(image_path)
        else:
            raise ValueError(f"Engine {engine} non supporté")

# EXEMPLE D'UTILISATION
if __name__ == "__main__":
    ocr = AdvancedInvoiceOCR()
    
    # Test avec différents engines
    image_path = "/workspace/Gitpod-Ubuntu-Server/test.png"
    
    try:
        print("=== PADDLEOCR (Recommandé pour la structure) ===")
        result_paddle = ocr.extract_invoice_structure(image_path, 'paddleocr')
        
        print(f"Texte structuré: {len(result_paddle['structured_text'])} éléments")
        print(f"Tableaux détectés: {len(result_paddle['tables'])}")
        print(f"Paires clé-valeur: {len(result_paddle['key_value_pairs'])}")
        
        # Afficher les tableaux
        for i, table in enumerate(result_paddle['tables']):
            print(f"\nTableau {i+1}: {table['rows']}x{table['columns']}")
            if table['dataframe'] is not None and not table['dataframe'].empty:
                print(table['dataframe'].head())
        
        # Afficher les paires clé-valeur
        print("\nPaires clé-valeur détectées:")
        for key, value in result_paddle['key_value_pairs'].items():
            print(f"- {key}: {value}")
    
    except Exception as e:
        print(f"Erreur PaddleOCR: {e}")
    
    try:
        print("\n=== TESSERACT avec structure ===")
        result_tesseract = ocr.extract_invoice_structure(image_path, 'tesseract')
        print(f"Éléments extraits: {len(result_tesseract['structured_text'])}")
        
    except Exception as e:
        print(f"Erreur Tesseract: {e}")