import cv2
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass
import logging
from sklearn.cluster import DBSCAN

# Configuration de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TextElement:
    """Élément de texte avec position"""
    text: str
    x: int
    y: int
    width: int
    height: int
    confidence: float
    center_x: int
    center_y: int
    
    def __post_init__(self):
        self.center_x = self.x + self.width // 2
        self.center_y = self.y + self.height // 2

class IntelligentTextStructurer:
    """Structureur de texte intelligent pour factures"""
    
    def __init__(self):
        self.setup_ocr()
        
        # Configuration pour le regroupement spatial
        self.line_tolerance = 25  # Tolérance pour les lignes
        self.column_tolerance = 40  # Tolérance pour les colonnes
        self.section_gap_threshold = 50  # Seuil pour détecter les sections
        
    def setup_ocr(self):
        """Configure PaddleOCR français"""
        try:
            from paddleocr import PaddleOCR
            self.ocr = PaddleOCR(
                use_angle_cls=True, 
                lang='en',
                show_log=False,
                det_limit_side_len=1920,
                use_gpu=False
            )
            logger.info("PaddleOCR initialisé en français")
        except ImportError:
            raise Exception("Installation requise: pip install paddleocr")
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """Préprocessing de l'image"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Impossible de charger l'image: {image_path}")
        
        # Redimensionnement si nécessaire
        height, width = img.shape[:2]
        if width > 2000:
            scale = 2000 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Conversion et amélioration
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        return enhanced
    
    def extract_text_elements(self, image_path: str) -> List[TextElement]:
        """Extrait les éléments de texte avec positions"""
        try:
            result = self.ocr.ocr(image_path, cls=True)
            if not result or not result[0]:
                logger.warning("Aucun texte détecté")
                return []
            
            elements = []
            for line in result[0]:
                bbox, (text, confidence) = line
                
                # Calculer les coordonnées
                x_coords = [point[0] for point in bbox]
                y_coords = [point[1] for point in bbox]
                
                x = int(min(x_coords))
                y = int(min(y_coords))
                width = int(max(x_coords) - min(x_coords))
                height = int(max(y_coords) - min(y_coords))
                
                cleaned_text = text.strip()
                if not cleaned_text or confidence < 0.5:
                    continue
                
                elements.append(TextElement(
                    text=cleaned_text,
                    x=x, y=y, width=width, height=height,
                    confidence=confidence,
                    center_x=x + width // 2,
                    center_y=y + height // 2
                ))
            
            # Trier par position (Y puis X)
            elements.sort(key=lambda e: (e.y, e.x))
            logger.info(f"Extraction: {len(elements)} éléments trouvés")
            return elements
            
        except Exception as e:
            logger.error(f"Erreur OCR: {e}")
            return []
    
    def group_elements_into_lines(self, elements: List[TextElement]) -> List[List[TextElement]]:
        """Groupe les éléments en lignes basées sur la position Y"""
        if not elements:
            return []
        
        # Clustering spatial sur l'axe Y
        y_positions = np.array([[elem.center_y] for elem in elements])
        clustering = DBSCAN(eps=self.line_tolerance, min_samples=1)
        line_clusters = clustering.fit_predict(y_positions)
        
        # Organiser par clusters (lignes)
        lines_dict = {}
        for elem, cluster in zip(elements, line_clusters):
            if cluster not in lines_dict:
                lines_dict[cluster] = []
            lines_dict[cluster].append(elem)
        
        # Trier chaque ligne par X et les lignes par Y moyen
        lines = []
        for cluster_id, line_elements in lines_dict.items():
            # Trier les éléments de la ligne par position X
            line_elements.sort(key=lambda e: e.x)
            # Calculer la position Y moyenne de la ligne
            avg_y = sum(e.center_y for e in line_elements) / len(line_elements)
            lines.append((avg_y, line_elements))
        
        # Trier les lignes par position Y
        lines.sort(key=lambda x: x[0])
        return [line[1] for line in lines]
    
    def detect_table_structure(self, lines: List[List[TextElement]]) -> Dict[str, List[int]]:
        """Détecte quelles lignes appartiennent à des tableaux"""
        table_lines = []
        regular_lines = []
        
        for line_idx, line in enumerate(lines):
            is_table_line = self._is_table_line(line, lines)
            
            if is_table_line:
                table_lines.append(line_idx)
            else:
                regular_lines.append(line_idx)
        
        return {
            'table_lines': table_lines,
            'regular_lines': regular_lines
        }
    
    def _is_table_line(self, line: List[TextElement], all_lines: List[List[TextElement]]) -> bool:
        """Détermine si une ligne fait partie d'un tableau"""
        # Critères pour identifier une ligne de tableau :
        
        # 1. Ligne avec au moins 3 éléments bien espacés
        if len(line) >= 3:
            # Vérifier la distribution horizontale
            x_positions = [elem.center_x for elem in line]
            x_span = max(x_positions) - min(x_positions)
            if x_span > 300:  # Large distribution
                return True
        
        # 2. Ligne contenant des nombres (montants, quantités)
        has_numbers = any(any(char.isdigit() for char in elem.text) for elem in line)
        if has_numbers and len(line) >= 2:
            return True
        
        # 3. Alignement avec d'autres lignes similaires
        similar_lines = 0
        for other_line in all_lines:
            if other_line != line and len(other_line) >= 2:
                alignment_score = self._calculate_alignment_score(line, other_line)
                if alignment_score > 0.6:
                    similar_lines += 1
        
        if similar_lines >= 1 and len(line) >= 2:
            return True
        
        return False
    
    def _calculate_alignment_score(self, line1: List[TextElement], line2: List[TextElement]) -> float:
        """Calcule le score d'alignement entre deux lignes"""
        if not line1 or not line2:
            return 0.0
        
        aligned_columns = 0
        max_comparisons = min(len(line1), len(line2))
        
        for i in range(max_comparisons):
            x_diff = abs(line1[i].center_x - line2[i].center_x)
            if x_diff < self.column_tolerance:
                aligned_columns += 1
        
        return aligned_columns / max(len(line1), len(line2))
    
    def detect_sections(self, lines: List[List[TextElement]]) -> List[Dict[str, Any]]:
        """Détecte les sections du document"""
        if not lines:
            return []
        
        sections = []
        current_section = {'type': 'header', 'lines': [], 'start_y': lines[0][0].center_y}
        
        for line_idx, line in enumerate(lines):
            current_y = line[0].center_y if line else 0
            
            # Détecter les grandes séparations (nouvelles sections)
            if line_idx > 0:
                prev_y = lines[line_idx - 1][0].center_y if lines[line_idx - 1] else 0
                gap = current_y - prev_y
                
                if gap > self.section_gap_threshold:
                    # Finaliser la section actuelle
                    if current_section['lines']:
                        sections.append(current_section)
                    
                    # Déterminer le type de la nouvelle section
                    section_type = self._determine_section_type(line_idx, lines)
                    current_section = {
                        'type': section_type,
                        'lines': [],
                        'start_y': current_y
                    }
            
            current_section['lines'].append(line)
        
        # Ajouter la dernière section
        if current_section['lines']:
            sections.append(current_section)
        
        return sections
    
    def _determine_section_type(self, line_idx: int, lines: List[List[TextElement]]) -> str:
        """Détermine le type de section basé sur la position et le contenu"""
        total_lines = len(lines)
        
        # En-tête : premières lignes
        if line_idx < total_lines * 0.2:
            return 'header'
        
        # Pied de page : dernières lignes
        elif line_idx > total_lines * 0.8:
            return 'footer'
        
        # Tableau : au milieu avec structure tabulaire
        elif self._is_table_line(lines[line_idx], lines):
            return 'table'
        
        # Corps : le reste
        else:
            return 'body'
    
    def format_structured_text(self, image_path: str) -> str:
        """Formate le texte de manière structurée pour LLM"""
        # Extraire les éléments
        elements = self.extract_text_elements(image_path)
        if not elements:
            return "Aucun texte détecté dans l'image."
        
        # Grouper en lignes
        lines = self.group_elements_into_lines(elements)
        
        # Détecter les sections
        sections = self.detect_sections(lines)
        
        # Détecter la structure des tableaux
        table_structure = self.detect_table_structure(lines)
        
        # Formater le texte final
        formatted_sections = []
        
        for section in sections:
            section_text = self._format_section(section, table_structure)
            if section_text.strip():
                formatted_sections.append(section_text)
        
        return '\n\n'.join(formatted_sections)
    
    def _format_section(self, section: Dict[str, Any], table_structure: Dict[str, List[int]]) -> str:
        """Formate une section spécifique"""
        lines = section['lines']
        
        formatted_lines = []
        
        for line in lines:
            line_text = self._format_line(line)
            if line_text.strip():
                formatted_lines.append(line_text)
        
        return '\n'.join(formatted_lines)
    
    def _format_line(self, line: List[TextElement]) -> str:
        """Formate une ligne d'éléments"""
        if not line:
            return ""
        
        if len(line) == 1:
            return line[0].text
        
        # Pour les lignes multi-éléments, ajouter des espaces intelligents
        formatted_parts = [line[0].text]
        
        for i in range(1, len(line)):
            prev_elem = line[i-1]
            curr_elem = line[i]
            
            # Calculer l'espacement basé sur la distance
            gap = curr_elem.x - (prev_elem.x + prev_elem.width)
            
            if gap > 100:  # Grande distance
                formatted_parts.append("  |  ")  # Séparateur de colonne
            elif gap > 30:  # Distance moyenne
                formatted_parts.append("  ")  # Double espace
            else:  # Petite distance
                formatted_parts.append(" ")  # Espace simple
            
            formatted_parts.append(curr_elem.text)
        
        return ''.join(formatted_parts)
    
    def extract_debug_info(self, image_path: str) -> Dict[str, Any]:
        """Extraction avec informations de debug"""
        elements = self.extract_text_elements(image_path)
        lines = self.group_elements_into_lines(elements)
        sections = self.detect_sections(lines)
        table_structure = self.detect_table_structure(lines)
        formatted_text = self.format_structured_text(image_path)
        
        return {
            'formatted_text': formatted_text,
            'debug_info': {
                'total_elements': len(elements),
                'total_lines': len(lines),
                'total_sections': len(sections),
                'table_lines_count': len(table_structure['table_lines']),
                'avg_confidence': sum(e.confidence for e in elements) / len(elements) if elements else 0,
                'sections_types': [s['type'] for s in sections]
            },
            'raw_text': ' '.join([e.text for e in elements])
        }

# EXEMPLE D'UTILISATION
if __name__ == "__main__":
    # Initialisation
    structurer = IntelligentTextStructurer()
    
    # Test avec image
    image_path = "/workspace/Gitpod-Ubuntu-Server/test2.png"
    
    try:
        print("Structuration du texte en cours...")
        
        # Extraction avec debug
        result = structurer.extract_debug_info(image_path)
        
        print("\nTEXTE STRUCTURÉ POUR LLM:")
        print("=" * 80)
        print(result['formatted_text'])
        print("=" * 80)
        
        print(f"\nINFORMATIONS DE DEBUG:")
        debug = result['debug_info']
        print(f"- Éléments détectés: {debug['total_elements']}")
        print(f"- Lignes regroupées: {debug['total_lines']}")
        print(f"- Sections détectées: {debug['total_sections']}")
        print(f"- Lignes de tableau: {debug['table_lines_count']}")
        print(f"- Confiance moyenne: {debug['avg_confidence']:.2f}")
        print(f"- Types de sections: {', '.join(debug['sections_types'])}")
        
        print(f"\nTEXTE BRUT (POUR COMPARAISON):")
        print("-" * 80)
        print(result['raw_text'])
        print("-" * 80)
        
        print("\nStructuration terminée avec succès!")
        
    except Exception as e:
        logger.error(f"❌ Erreur: {e}")
        print(f"Erreur: {e}")
