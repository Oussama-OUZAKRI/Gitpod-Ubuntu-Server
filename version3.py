import cv2
import numpy as np
from PIL import Image
import pandas as pd
from typing import List, Dict, Tuple, Any
import re
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class TextElement:
    """Élément de texte avec position et métadonnées"""
    text: str
    x: int
    y: int
    width: int
    height: int
    confidence: float
    center_x: int
    center_y: int
    
    def __post_init__(self):
        if not hasattr(self, 'center_x'):
            self.center_x = self.x + self.width // 2
        if not hasattr(self, 'center_y'):
            self.center_y = self.y + self.height // 2

class StructuredTextExtractor:
    """Extracteur de texte avec préservation de structure pour factures"""
    
    def __init__(self, engine='paddleocr'):
        self.engine = engine
        self.setup_engine()
        
        # Configuration pour la structuration - Ajustée pour les factures
        self.line_tolerance = 20  # Augmenté pour capturer plus de variations
        self.column_tolerance = 50  # Plus flexible pour l'alignement
        self.table_detection_threshold = 2  # Réduit à 2 colonnes minimum
        
    def setup_engine(self):
        """Configure l'engine OCR choisi"""
        if self.engine == 'paddleocr':
            try:
                from paddleocr import PaddleOCR
                self.ocr = PaddleOCR(use_angle_cls=True, lang='fr', show_log=False)
            except ImportError:
                raise Exception("pip install paddleocr")
                
        elif self.engine == 'easyocr':
            try:
                import easyocr
                self.ocr = easyocr.Reader(['fr', 'en'])
            except ImportError:
                raise Exception("pip install easyocr")
                
        elif self.engine == 'tesseract':
            try:
                import pytesseract
                from pytesseract import Output
                self.pytesseract = pytesseract
                self.Output = Output
            except ImportError:
                raise Exception("pip install pytesseract")
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """Préprocessing optimisé pour la lecture de texte"""
        # Charger l'image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Impossible de charger l'image: {image_path}")
        
        # Conversion en niveaux de gris
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Réduction du bruit avec filtre bilatéral
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Amélioration du contraste avec CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # Binarisation adaptative pour améliorer la lisibilité
        binary = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        return binary
    
    def extract_text_elements(self, image_path: str) -> List[TextElement]:
        """Extrait les éléments de texte avec leurs positions"""
        elements = []
        
        if self.engine == 'paddleocr':
            elements = self._extract_with_paddleocr(image_path)
        elif self.engine == 'easyocr':
            elements = self._extract_with_easyocr(image_path)
        elif self.engine == 'tesseract':
            elements = self._extract_with_tesseract(image_path)
        
        # Trier par position (Y puis X)
        elements.sort(key=lambda e: (e.y, e.x))
        
        return elements
    
    def _extract_with_paddleocr(self, image_path: str) -> List[TextElement]:
        """Extraction avec PaddleOCR"""
        result = self.ocr.ocr(image_path, cls=True)
        elements = []
        
        for line in result[0]:
            bbox, (text, confidence) = line
            
            # Calculer les coordonnées du rectangle
            x_coords = [point[0] for point in bbox]
            y_coords = [point[1] for point in bbox]
            
            x = int(min(x_coords))
            y = int(min(y_coords))
            width = int(max(x_coords) - min(x_coords))
            height = int(max(y_coords) - min(y_coords))
            
            elements.append(TextElement(
                text=text.strip(),
                x=x, y=y, width=width, height=height,
                confidence=confidence,
                center_x=x + width // 2,
                center_y=y + height // 2
            ))
        
        return elements
    
    def _extract_with_easyocr(self, image_path: str) -> List[TextElement]:
        """Extraction avec EasyOCR"""
        results = self.ocr.readtext(image_path, detail=1)
        elements = []
        
        for (bbox, text, confidence) in results:
            # Calculer les coordonnées
            x_coords = [point[0] for point in bbox]
            y_coords = [point[1] for point in bbox]
            
            x = int(min(x_coords))
            y = int(min(y_coords))
            width = int(max(x_coords) - min(x_coords))
            height = int(max(y_coords) - min(y_coords))
            
            elements.append(TextElement(
                text=text.strip(),
                x=x, y=y, width=width, height=height,
                confidence=confidence,
                center_x=x + width // 2,
                center_y=y + height // 2
            ))
        
        return elements
    
    def _extract_with_tesseract(self, image_path: str) -> List[TextElement]:
        """Extraction avec Tesseract"""
        # Préprocessing
        processed_img = self.preprocess_image(image_path)
        
        # Configuration Tesseract
        custom_config = r'--oem 3 --psm 6 -c preserve_interword_spaces=1'
        
        # Extraction avec données de position
        data = self.pytesseract.image_to_data(
            processed_img, 
            output_type=self.Output.DICT,
            config=custom_config,
            lang='fra+eng'
        )
        
        elements = []
        n_boxes = len(data['text'])
        
        for i in range(n_boxes):
            confidence = int(data['conf'][i])
            text = data['text'][i].strip()
            
            if confidence > 30 and text:  # Filtrer les éléments de faible confiance
                elements.append(TextElement(
                    text=text,
                    x=data['left'][i],
                    y=data['top'][i],
                    width=data['width'][i],
                    height=data['height'][i],
                    confidence=confidence / 100.0,
                    center_x=data['left'][i] + data['width'][i] // 2,
                    center_y=data['top'][i] + data['height'][i] // 2
                ))
        
        return elements
    
    def group_into_lines(self, elements: List[TextElement]) -> List[List[TextElement]]:
        """Groupe les éléments en lignes basées sur la position Y"""
        if not elements:
            return []
        
        lines = []
        current_line = [elements[0]]
        
        for element in elements[1:]:
            # Si l'élément est sur la même ligne (Y similaire)
            if abs(element.center_y - current_line[-1].center_y) <= self.line_tolerance:
                current_line.append(element)
            else:
                # Trier la ligne actuelle par X et commencer une nouvelle ligne
                current_line.sort(key=lambda e: e.x)
                lines.append(current_line)
                current_line = [element]
        
        # Ajouter la dernière ligne
        if current_line:
            current_line.sort(key=lambda e: e.x)
            lines.append(current_line)
        
        return lines
    
    def detect_table_structure(self, lines: List[List[TextElement]]) -> Dict[str, Any]:
        """Détecte la structure des tableaux dans les lignes - Version améliorée"""
        table_info = {
            'has_tables': False,
            'table_sections': [],
            'regular_sections': []
        }
        
        # Première passe : identifier toutes les lignes potentielles de tableau
        table_candidates = []
        for line_idx, line in enumerate(lines):
            if self._is_table_line(line, lines):
                table_candidates.append(line_idx)
        
        if not table_candidates:
            # Aucun tableau détecté, tout est du texte
            table_info['regular_sections'].append({
                'type': 'text',
                'lines': lines,
                'start_line': 0,
                'end_line': len(lines) - 1
            })
            return table_info
        
        # Grouper les lignes de tableau consécutives ou proches
        table_groups = []
        current_group = [table_candidates[0]]
        
        for i in range(1, len(table_candidates)):
            # Si les lignes sont proches (avec quelques lignes de texte entre)
            if table_candidates[i] - table_candidates[i-1] <= 3:
                current_group.append(table_candidates[i])
            else:
                table_groups.append(current_group)
                current_group = [table_candidates[i]]
        
        if current_group:
            table_groups.append(current_group)
        
        # Construire les sections en alternant texte et tableau
        current_line = 0
        
        for group in table_groups:
            start_table = min(group)
            end_table = max(group)
            
            # Ajouter le texte avant le tableau
            if current_line < start_table:
                table_info['regular_sections'].append({
                    'type': 'text',
                    'lines': lines[current_line:start_table],
                    'start_line': current_line,
                    'end_line': start_table - 1
                })
            
            # Ajouter le tableau (inclure quelques lignes autour pour le contexte)
            table_start = max(0, start_table - 1)  # Inclure la ligne précédente si possible
            table_end = min(len(lines), end_table + 2)  # Inclure une ligne suivante
            
            table_lines = lines[table_start:table_end]
            
            table_info['table_sections'].append({
                'type': 'table',
                'lines': table_lines,
                'start_line': table_start,
                'end_line': table_end - 1,
                'structure': self._analyze_table_structure(table_lines)
            })
            table_info['has_tables'] = True
            
            current_line = table_end
        
        # Ajouter le texte restant après le dernier tableau
        if current_line < len(lines):
            table_info['regular_sections'].append({
                'type': 'text',
                'lines': lines[current_line:],
                'start_line': current_line,
                'end_line': len(lines) - 1
            })
        
        return table_info
    
    def _is_table_line(self, line: List[TextElement], all_lines: List[List[TextElement]]) -> bool:
        """Détermine si une ligne fait partie d'un tableau"""
        # Critères améliorés pour identifier une ligne de tableau :
        
        # 1. Détecter les en-têtes de tableau typiques
        line_text = ' '.join([elem.text.lower() for elem in line])
        table_headers = ['qty', 'rate', 'amount', 'description', 'quantity', 'price', 'total', 'item']
        if any(header in line_text for header in table_headers) and len(line) >= 2:
            return True
        
        # 2. Détecter les lignes avec des patterns numériques (montants, quantités)
        numeric_pattern = r'\d+[.,]\d+|\$\d+|\d+%|\d+\.\d+'
        numeric_elements = sum(1 for elem in line if re.search(numeric_pattern, elem.text))
        if numeric_elements >= 2 and len(line) >= 3:
            return True
        
        # 3. Lignes avec au moins 3 colonnes et distribution horizontale
        if len(line) >= 3:
            # Vérifier la distribution des éléments sur la largeur
            x_positions = [elem.center_x for elem in line]
            x_span = max(x_positions) - min(x_positions)
            
            # Si les éléments sont bien répartis horizontalement
            if x_span > 300:  # Large distribution
                return True
        
        # 4. Détecter par alignement avec d'autres lignes similaires
        similar_lines = 0
        for other_line in all_lines:
            if other_line != line and len(other_line) >= 2:
                # Comparer les positions X approximatives
                alignment_score = self._calculate_column_alignment(line, other_line)
                if alignment_score > 0.6:  # 60% d'alignement
                    similar_lines += 1
        
        if similar_lines >= 1 and len(line) >= 2:
            return True
        
    def _calculate_column_alignment(self, line1: List[TextElement], line2: List[TextElement]) -> float:
        """Calcule le score d'alignement entre deux lignes"""
        if not line1 or not line2:
            return 0.0
        
        aligned_columns = 0
        total_comparisons = min(len(line1), len(line2))
        
        for i in range(total_comparisons):
            # Tolérance d'alignement vertical
            x_diff = abs(line1[i].center_x - line2[i].center_x)
            if x_diff < self.column_tolerance:
                aligned_columns += 1
        
        return aligned_columns / max(len(line1), len(line2)) if max(len(line1), len(line2)) > 0 else 0.0
    
    def _analyze_table_structure(self, table_lines: List[List[TextElement]]) -> Dict[str, Any]:
        """Analyse la structure détaillée d'un tableau"""
        if not table_lines:
            return {}
        
        # Déterminer le nombre de colonnes
        max_cols = max(len(line) for line in table_lines)
        
        # Analyser les positions des colonnes
        column_positions = []
        for col_idx in range(max_cols):
            positions = []
            for line in table_lines:
                if col_idx < len(line):
                    positions.append(line[col_idx].x)
            
            if positions:
                column_positions.append({
                    'index': col_idx,
                    'avg_x': sum(positions) / len(positions),
                    'min_x': min(positions),
                    'max_x': max(positions)
                })
        
        # Créer la matrice du tableau
        table_matrix = []
        for line in table_lines:
            row = []
            for col_idx in range(max_cols):
                if col_idx < len(line):
                    row.append(line[col_idx].text)
                else:
                    row.append('')
            table_matrix.append(row)
        
        return {
            'rows': len(table_lines),
            'columns': max_cols,
            'column_positions': column_positions,
            'matrix': table_matrix
        }
    
    def format_structured_text(self, image_path: str) -> str:
        """Formate le texte extrait en préservant la structure"""
        # Extraire les éléments
        elements = self.extract_text_elements(image_path)
        
        if not elements:
            return ""
        
        # Grouper en lignes
        lines = self.group_into_lines(elements)
        
        # Détecter la structure des tableaux
        structure_info = self.detect_table_structure(lines)
        
        # Construire le texte formaté en respectant l'ordre
        formatted_sections = []
        
        # Combiner et trier toutes les sections par position
        all_sections = []
        
        for section in structure_info['regular_sections']:
            all_sections.append((section['start_line'], 'text', section))
        
        for section in structure_info['table_sections']:
            all_sections.append((section['start_line'], 'table', section))
        
        # Trier par ligne de début
        all_sections.sort(key=lambda x: x[0])
        
        # Formatter chaque section dans l'ordre
        for _, section_type, section in all_sections:
            if section_type == 'text':
                section_text = self._format_text_section(section['lines'])
                if section_text.strip():
                    formatted_sections.append(section_text)
            else:  # table
                table_text = self._format_table_section(section)
                if table_text.strip():
                    formatted_sections.append("\n" + table_text + "\n")  # Espacement autour des tableaux
        
        return '\n\n'.join(formatted_sections)
    
    def _format_text_section(self, lines: List[List[TextElement]]) -> str:
        """Formate une section de texte normal"""
        section_lines = []
        
        for line in lines:
            # Joindre les éléments de la ligne avec des espaces appropriés
            line_text = self._join_line_elements(line)
            if line_text.strip():
                section_lines.append(line_text)
        
        return '\n'.join(section_lines)
    
    def _format_table_section(self, table_section: Dict[str, Any]) -> str:
        """Formate une section de tableau"""
        structure = table_section['structure']
        matrix = structure['matrix']
        
        if not matrix:
            return ""
        
        # Calculer la largeur de chaque colonne
        col_widths = []
        for col_idx in range(structure['columns']):
            max_width = 0
            for row in matrix:
                if col_idx < len(row):
                    max_width = max(max_width, len(str(row[col_idx])))
            col_widths.append(max(max_width, 10))  # Minimum 10 caractères
        
        # Formater le tableau
        formatted_rows = []
        
        # En-tête de tableau (première ligne souvent)
        if matrix:
            header_row = []
            for col_idx, cell in enumerate(matrix[0]):
                if col_idx < len(col_widths):
                    header_row.append(str(cell).ljust(col_widths[col_idx]))
            formatted_rows.append('\t'.join(header_row))
            
            # Ligne de séparation (optionnelle)
            # separator = '\t'.join(['-' * width for width in col_widths])
            # formatted_rows.append(separator)
        
        # Autres lignes
        for row in matrix[1:]:
            formatted_row = []
            for col_idx, cell in enumerate(row):
                if col_idx < len(col_widths):
                    formatted_row.append(str(cell).ljust(col_widths[col_idx]))
            formatted_rows.append('\t'.join(formatted_row))
        
        return '\n'.join(formatted_rows)
    
    def _join_line_elements(self, line: List[TextElement]) -> str:
        """Joint les éléments d'une ligne avec un espacement intelligent"""
        if not line:
            return ""
        
        if len(line) == 1:
            return line[0].text
        
        # Calculer les espacements entre les éléments
        result = [line[0].text]
        
        for i in range(1, len(line)):
            prev_element = line[i-1]
            curr_element = line[i]
            
            # Distance entre les éléments
            gap = curr_element.x - (prev_element.x + prev_element.width)
            
            # Ajouter des espaces proportionnels à la distance
            if gap > 100:  # Grande distance - plusieurs espaces/tab
                result.append('  ')  # Double espace
            elif gap > 30:  # Distance moyenne - espace simple
                result.append(' ')
            else:  # Petite distance - collé
                pass  # Pas d'espace supplémentaire
            
            result.append(curr_element.text)
        
        return ''.join(result)
    
    def extract_with_debug(self, image_path: str) -> Dict[str, Any]:
        """Extraction avec informations de debug"""
        elements = self.extract_text_elements(image_path)
        lines = self.group_into_lines(elements)
        structure_info = self.detect_table_structure(lines)
        formatted_text = self.format_structured_text(image_path)
        
        return {
            'formatted_text': formatted_text,
            'elements_count': len(elements),
            'lines_count': len(lines),
            'tables_detected': len(structure_info['table_sections']),
            'structure_info': structure_info,
            'debug_info': {
                'elements': [{'text': e.text, 'x': e.x, 'y': e.y, 'confidence': e.confidence} for e in elements[:10]],  # Premiers 10 éléments
                'avg_confidence': sum(e.confidence for e in elements) / len(elements) if elements else 0
            }
        }

# EXEMPLE D'UTILISATION
if __name__ == "__main__":
    # Test avec différents engines
    engines = ['paddleocr', 'easyocr', 'tesseract']
    image_path = "/workspace/Gitpod-Ubuntu-Server/test.png"
    
    for engine in engines:
        try:
            print(f"\n{'='*60}")
            print(f"TEST AVEC {engine.upper()}")
            print(f"{'='*60}")
            
            extractor = StructuredTextExtractor(engine=engine)
            
            # Extraction simple
            formatted_text = extractor.format_structured_text(image_path)
            print("TEXTE FORMATÉ:")
            print(formatted_text)
            
            # Extraction avec debug
            debug_result = extractor.extract_with_debug(image_path)
            print(f"\nINFOS DEBUG:")
            print(f"- Éléments extraits: {debug_result['elements_count']}")
            print(f"- Lignes détectées: {debug_result['lines_count']}")
            print(f"- Tableaux détectés: {debug_result['tables_detected']}")
            print(f"- Confiance moyenne: {debug_result['debug_info']['avg_confidence']:.2f}")
            
        except Exception as e:
            print(f"Erreur avec {engine}: {e}")
    
    # Test de comparaison
    print(f"\n{'='*60}")
    print("COMPARAISON DES ENGINES")
    print(f"{'='*60}")
    
    results = {}
    for engine in ['paddleocr', 'easyocr']:  # Les plus fiables
        try:
            extractor = StructuredTextExtractor(engine=engine)
            debug_result = extractor.extract_with_debug(image_path)
            results[engine] = {
                'text_length': len(debug_result['formatted_text']),
                'elements': debug_result['elements_count'],
                'confidence': debug_result['debug_info']['avg_confidence'],
                'tables': debug_result['tables_detected']
            }
        except:
            results[engine] = None
    
    for engine, result in results.items():
        if result:
            print(f"{engine}: {result['elements']} éléments, "
                  f"{result['confidence']:.2f} confiance, "
                  f"{result['tables']} tableaux")