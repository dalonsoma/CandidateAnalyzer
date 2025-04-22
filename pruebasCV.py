import os
import re
import PyPDF2
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from typing import Dict, List, Optional


class CVEvaluator:
    def __init__(self, job_description: str, profile_requirements: dict=None):
        """
        Inicializa el evaluador con:
        - job_description: Descripción textual del puesto
        - profile_requirements: Diccionario con requisitos específicos del perfil
        """
        self.nlp = spacy.load("es_core_news_sm")  # o "en_core_web_sm" para inglés
        self.job_description = job_description
        self.profile_requirements = profile_requirements
        #self.required_keywords = self.extract_keywords(job_description)
        self.required_keywords = self.extract_keywords(job_description)

        # Palabras clave específicas del perfil (seguridad OT en este caso)
        self.profile_keywords = self._get_profile_keywords()

    def _get_profile_keywords(self) -> List[str]:
        """Extrae y combina palabras clave específicas del perfil"""
        keywords = []
        for category, requirements in self.profile_requirements.items():
            if 'keywords' in requirements:
                keywords.extend(requirements['keywords'])
        return list(set(keywords))

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extrae texto de un archivo PDF"""
        text = ""
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text()
        return text

    def preprocess_text(self, text: str) -> str:
        """Limpia y normaliza el texto"""
        text = text.lower()
        text = re.sub(r'[^a-zA-Záéíóúñ\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def extract_keywords(self, text: str) -> List[str]:
        """Extrae palabras clave del texto usando NLP"""
        doc = self.nlp(text)
        keywords = [
            token.lemma_ for token in doc
            if not token.is_stop and not token.is_punct and token.is_alpha
        ]
        return list(set(keywords))

    def calculate_match_score(self, cv_text: str) -> float:
        """Calcula el puntaje de coincidencia entre el CV y los requisitos"""
        cv_processed = self.preprocess_text(cv_text)
        jd_processed = self.preprocess_text(self.job_description)

        # Vectorización TF-IDF
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([jd_processed, cv_processed])

        # Similitud del coseno (50% del score)
        similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0] * 50

        # Puntaje por palabras clave específicas del perfil (50% del score)
        profile_match_score = self.calculate_profile_specific_score(cv_text)

        return similarity_score + profile_match_score

    def calculate_profile_specific_score(self, cv_text: str) -> float:
        """Calcula el puntaje de coincidencia con el perfil de forma robusta"""
        try:
            date_ranges = []
            total_score = 0.0
            max_possible = 100  # Puntaje máximo total (no 50 como antes)
            cv_text = str(cv_text).lower() if cv_text else ""

            # 1. Puntaje por keywords (si existe el campo)
            if "keywords" in self.profile_requirements:
                keywords = self.profile_requirements["keywords"].get("keywords", [])
                weight = self.profile_requirements["keywords"].get("weight", 0.0)

                if keywords:
                    found = sum(1 for kw in keywords if str(kw).lower() in cv_text)
                    total_score += (found / len(keywords)) * weight * max_possible

            # 2. Puntaje por certificaciones (si existe el campo)
            if "certifications" in self.profile_requirements:
                certifications = self.profile_requirements["certifications"].get("items", [])
                weight = self.profile_requirements["certifications"].get("weight", 0.0)

                if certifications:
                    found = sum(1 for cert in certifications if str(cert).lower() in cv_text)
                    total_score += (found / len(certifications)) * weight * max_possible

            # 3. Puntaje por experiencia (si existe el campo)
            if "experience" in self.profile_requirements:
                exp_field = str(self.profile_requirements["experience"].get("field", "")).lower()
                min_years_raw = self.profile_requirements.get("experience", {}).get("min_years")
                try:
                    min_years = float(min_years_raw) if min_years_raw is not None else 1.0
                except (ValueError, TypeError):
                    print("ERROR EN MIN_YEARS")
                    min_years = 1.0

                weight = self.profile_requirements["experience"].get("weight", 0.0)

                exp_years = 0.0
                date_ranges = []

                match = re.search(r'(\d+)\s*(años?|years?)[\s\w]*' + exp_field, cv_text, re.IGNORECASE)
                if not match:
                    match = re.search(r'(\d+)\s*(años?|years?)\s+.*?(experiencia|laboral|work)', cv_text, re.IGNORECASE)
                if not match:
                    date_ranges = re.findall(r'(\d{4})\s*[-–]\s*(\d{4}|actual)', cv_text)
                    exp_years = len(date_ranges) * 1.5

                if match and not date_ranges:
                    try:
                        exp_years = float(match.group(1) or 0.0)
                    except (ValueError, TypeError):
                        exp_years = 0.0

                if min_years > 0:
                    exp_ratio = min(exp_years / min_years, 1.0)
                    total_score += exp_ratio * weight * max_possible

            return round(min(total_score, max_possible), 2)

        except Exception as e:
            print(f"Error seguro: {str(e)}")
            return 0.0





    def extract_name2(self, text: str) -> str:
        """Intenta extraer el nombre del CV"""
        doc = self.nlp(text)
        for ent in doc.ents:
            if ent.label_ == "PER":
                return ent.text
        return "Nombre no encontrado"

    def extract_name(self, text: str) -> str:
        """Extrae un posible nombre del CV usando varias heurísticas robustas"""
        # 1. NLP en primeras líneas
        first_lines = "\n".join(text.strip().split("\n")[:10])
        doc = self.nlp(first_lines[:400])  # spaCy o similar

        for ent in doc.ents:
            if ent.label_ in ("PER", "PERSON") and 2 <= len(ent.text.split()) <= 4:
                if not re.search(r'\b(curriculum|vitae|analyst|developer|security|ingeniero|ciberseguridad)\b',
                                 ent.text, re.IGNORECASE):
                    return ent.text.strip()

        # 2. Buscar línea con formato de nombre clásico
        name_regex = r'^([A-ZÁÉÍÓÚÑÜ][a-záéíóúñü]+(?:\s+(de\s|del\s|y\s)?[A-ZÁÉÍÓÚÑÜ][a-záéíóúñü]+){1,3})$'
        for line in text.split('\n')[:15]:
            clean_line = line.strip()
            if re.match(name_regex, clean_line, re.UNICODE):
                return clean_line

        # 3. Último recurso: palabras capitalizadas en primeras líneas
        capitalized = [w for w in text.split() if w.istitle() and len(w) > 2]
        if len(capitalized) >= 2:
            return " ".join(capitalized[:3])

        return "Nombre no detectado"

    def extract_name4(self, text: str) -> str:
        """Extrae nombres de CVs en español/inglés sin depender de datos fijos"""
        # 1. Patrón para líneas divisorias después del nombre (común en CVs)
        name_divider = re.search(
            r'^([A-ZÁÉÍÓÚÑÜ][A-Za-záéíóúñü]+\s+[A-ZÁÉÍÓÚÑÜ][A-Za-záéíóúñü]+(?:\s+[A-ZÁÉÍÓÚÑÜ][A-Za-záéíóúñü]+)?)\s*\n[-=~]{3,}',
            text,
            re.MULTILINE
        )
        if name_divider:
            return name_divider.group(1).strip()
        match = re.search(r'^([A-ZÁÉÍÓÚÑ][a-záéíóúñü]+(?:\s(?:de\s?)?[A-ZÁÉÍÓÚÑ][a-záéíóúñü]+){1,3})', text,
                          re.MULTILINE)

        if match:
            candidato = match.group(1)
            # Filtro contextual combinado
            if not re.search(r'\b(en|security|titulo|carrera|licenciado|licenciada|grado|curso|experiencia|analista)\b', candidato, re.I):
                     # Usando diccionario opcional
                return candidato.strip()
        # 2. Detección de formato "# Nombre Apellido"
        hash_format = re.search(r'^#\s+([A-ZÁÉÍÓÚÑÜ][a-záéíóúñü]+\s+[A-ZÁÉÍÓÚÑÜ][a-záéíóúñü]+)', text)
        if hash_format:
            return hash_format.group(1)
        name_match = re.search(
            r'^(?!.*\b(grado|master|curso|ingenier[ií]a)\b)'  # Excluye estas palabras
            r'([A-ZÁÉÍÓÚÑ][a-záéíóúñü]+(?:\s(?:de\s)?[A-ZÁÉÍÓÚÑ][a-záéíóúñü]+){1,3})',
            text,
            re.IGNORECASE | re.MULTILINE
        )
        if name_match:
            return name_match.group(0).strip() if name_match else "Nombre no detectado"

        # 3. NLP con filtros estrictos (solo primeras líneas)
        first_paragraph = text.split('\n')[0] if text else ""
        doc = self.nlp(first_paragraph[:150])  # Analiza solo el inicio
        for ent in doc.ents:
            if ent.label_ == "PER" and 2 <= len(ent.text.split()) <= 3:
                if not any(
                        kw in ent.text.lower()
                        for kw in ["curriculum", "vitae", "analyst", "developer"]
                ):
                    return ent.text

        # 4. Último recurso: Primeras palabras capitalizadas
        words = [
            w for w in text.split()
            if w.istitle() and len(w) > 2
               and not w.lower() in ["ingeniero", "soc", "phd"]
        ]
        if len(words) >= 2:
            return " ".join(words[:2])

        return "Nombre no detectado (formato no compatible)"

    def extract_name3(self, text: str) -> str:
        """Extrae el nombre usando 4 técnicas consecutivas con tolerancia a fallos"""

        # 1. Detección por formato estructurado (CV profesionales)
        structured_formats = [
            r'^(?:#|CV|RESUME?)\s*[\-\:]?\s*([A-ZÁÉÍÓÚÑÜ][a-záéíóúñü]+(?:\s+[A-ZÁÉÍÓÚÑÜ][a-záéíóúñü]+)+)',
            # "# Nombre Apellido"
            r'^(?:Nombre|Name)\s*[\:\-]\s*([^\n]+)',  # "Nombre: Diego Alonso"
            r'^([A-ZÁÉÍÓÚÑÜ\s]+\n[-=~]{5,})'  # "DIEGO MARTIN\n------"
        ]

        for pattern in structured_formats:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).strip()

        # 2. Heurística de línea inicial (para CVs simples)
        first_line = text.split('\n')[0].strip()
        if re.fullmatch(r'^([A-ZÁÉÍÓÚÑÜ][a-záéíóúñü]+\s+){1,3}[A-ZÁÉÍÓÚÑÜ][a-záéíóúñü]+$', first_line):
            return first_line

        # 3. NLP con spaCy (para formatos libres)
        doc = self.nlp(text)
        for ent in doc.ents:
            if ent.label_ == "PER" and len(ent.text.split()) >= 2:  # Filtra nombres completos
                return ent.text

        # 4. Fallback: Extrae el primer bloque de 2-4 palabras capitalizadas
        words = re.findall(r'\b[A-ZÁÉÍÓÚÑÜ][a-záéíóúñü]+\b', text)
        if len(words) >= 2:
            return ' '.join(words[:3 if len(words) >= 3 else 2])

        return "Nombre no detectado (formato no soportado)"

    def extract_email(self, text: str) -> str:
        """Extrae el email del CV"""
        email = re.search(r'[\w\.-]+@[\w\.-]+', text)
        return email.group(0) if email else "Email no encontrado"

    def extract_profile_specific_data2(self, cv_text: str) -> Dict:
        """Extrae datos específicos del perfil solicitado"""
        results = {}

        # Extraer certificaciones
        if "certifications" in self.profile_requirements:
            certs = []
            for cert in self.profile_requirements['certifications']['items']:
                if cert.lower() in cv_text.lower():
                    certs.append(cert)
            results['certifications'] = certs

        # Extraer años de experiencia en el campo
        if "experience" in self.profile_requirements:
            exp_field = self.profile_requirements['experience']['field']
            exp_pattern = r'(\d+)\s*(año|años|year|years)\s+.*?' + exp_field
            match = re.search(exp_pattern, cv_text, re.IGNORECASE)
            results['years_experience'] = int(match.group(1)) if match else 0

        return results

    def extract_profile_specific_data2(self, cv_text: str) -> Dict:
        """Extrae datos específicos del perfil solicitado"""
        results = {}

        # Extraer certificaciones

        if "certifications" in self.profile_requirements:
            certs = []
            items = self.profile_requirements['certifications'].get('items', [])
            for cert in items:
                if cert and cert.lower() in cv_text.lower():
                    certs.append(cert)
            results['certifications'] = certs

        # Extraer años de experiencia en el campo
        if "experience" in self.profile_requirements:
            exp_field = self.profile_requirements['experience'].get('field', '')
            exp_pattern = r'(\d+)\s*(año|años|year|years)\s+.*?' + re.escape(exp_field)
            match = re.search(exp_pattern, cv_text, re.IGNORECASE)
            years = 0
            if match:
                try:
                    years = int(match.group(1))
                except (ValueError, TypeError):
                    years = 0
            results['years_experience'] = years

        return results

    def extract_profile_specific_data(self, cv_text: str) -> Dict:
        """Extrae nombre, email, años de experiencia y certificaciones de un CV"""
        results = {}

        # --- Extraer nombre con NLP o heurística ---
        def extract_name(text: str) -> str:
            # 1. NLP si está disponible
            if hasattr(self, "nlp"):
                doc = self.nlp("\n".join(text.strip().split("\n")[:10])[:400])
                for ent in doc.ents:
                    if ent.label_ in ("PER", "PERSON") and 2 <= len(ent.text.split()) <= 4:
                        if not re.search(r'\b(curriculum|vitae|analyst|developer|security|ingeniero|ciberseguridad)\b',
                                         ent.text, re.I):
                            return ent.text.strip()

            # 2. Línea con capitalización clásica
            name_regex = r'^([A-ZÁÉÍÓÚÑÜ][a-záéíóúñü]+(?:\s+(de\s|del\s|y\s)?[A-ZÁÉÍÓÚÑÜ][a-záéíóúñü]+){1,3})$'
            for line in text.split('\n')[:15]:
                clean_line = line.strip()
                if re.match(name_regex, clean_line, re.UNICODE):
                    return clean_line

            # 3. Primeras palabras capitalizadas
            capitalized = [w for w in text.split() if w.istitle() and len(w) > 2]
            if len(capitalized) >= 2:
                return " ".join(capitalized[:3])

            return "Nombre no detectado"

        # --- Extraer email ---
        def extract_email(text: str) -> str:
            match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
            return match.group(0) if match else "Correo no detectado"

        # --- Extraer certificaciones ---
        certs = []
        if "certifications" in self.profile_requirements:
            items = self.profile_requirements['certifications'].get('items', [])
            for cert in items:
                print(cert)
                if cert and cert.lower() in cv_text.lower():
                    certs.append(cert)
        results['certifications'] = certs

        # --- Extraer años de experiencia (varios patrones) ---
        years = 0
        #years = None
        exp_field = ""

        # Primero verificamos si alguna de las claves existe
        if "experience" in self.profile_requirements:
            exp_data = self.profile_requirements["experience"]
            exp_field = exp_data.get('field', '')
        elif "experiencia" in self.profile_requirements:
            exp_data = self.profile_requirements["experiencia"]
            exp_field = exp_data.get('field', '')

        # Patrones para detectar años de experiencia
        patterns = [
            r'más de\s+(\d+)\s+(años|year|years)',
            r'over\s+(\d+)\s+(years?)',
            r'(\d+)\s+(años|year|years?)\s+de\s+experiencia',
            r'con\s+(\d+)\s+(años|year|years?)\s+de\s+experiencia',
            r'(\d+)\s+(años|year|years?)\s+en\s+' + (re.escape(exp_field) if exp_field else r'\w+')
        ]

        for pat in patterns:
            matches = re.findall(pat, cv_text, re.IGNORECASE)
            for match in matches:
                try:
                    # El primer elemento del match debe ser el número
                    candidate_years = int(match[0] if isinstance(match, tuple) else match)
                    if years is None or candidate_years > years:
                        years = candidate_years
                except (ValueError, TypeError, IndexError):
                    continue

        results = {}
        results['years_experience'] = years if years is not None else 0

        # --- Añadir nombre y email ---
        results['name'] = extract_name(cv_text)
        results['email'] = extract_email(cv_text)

        return results
    def evaluate_cv(self, pdf_path: str) -> Optional[Dict]:
        """Evalúa un CV y devuelve un dict con los resultados"""
        try:
            try:
                cv_text = self.extract_text_from_pdf(pdf_path)
                score = self.calculate_match_score(cv_text)
            except Exception as e:
                print("EY")
                print(e)

            # Extraer información básica
            name = self.extract_name(cv_text)
            email = self.extract_email(cv_text)

            # Extraer información específica del perfil
            try:
                profile_data = self.extract_profile_specific_data(cv_text)
            except Exception as e1:
                print("EY1")
                print(e1)

            return {
                'name': name,
                'email': email,
                'score': round(score, 2),
                'file': os.path.basename(pdf_path),
                **profile_data  # Incluir los datos específicos del perfil
            }
        except Exception as e:
            print(f"Error procesando {pdf_path}: {str(e)}")
            return None

    def evaluate_directory(self, dir_path: str) -> List[Dict]:
        """Evalúa todos los PDFs en un directorio"""
        results = []
        for filename in os.listdir(dir_path):
            if filename.lower().endswith('.pdf'):
                filepath = os.path.join(dir_path, filename)
                result = self.evaluate_cv(filepath)
                if result:
                    results.append(result)

        # Ordenar resultados por puntaje
        return sorted(results, key=lambda x: x['score'], reverse=True)


# Perfil de ejemplo para Seguridad OT (Tecnología Operacional)
OT_SECURITY_PROFILE = {
    "keywords": {
        "keywords": [
            "seguridad ot", "ics", "scada", "plc", "industrial",
            "ciber seguridad", "redes industriales", "modbus",
            "dnp3", "protección crítica", "firewall industrial"
        ],
        "weight": 0.4  # 40% de este grupo
    },
    "certifications": {
        "items": [
            "ISA/IEC 62443", "GIAC GICSP", "Certified SCADA Security Architect",
            "OSCP", "CISSP", "CRISC"
        ],
        "weight": 0.3  # 30% de este grupo
    },
    "experience": {
        "field": "seguridad ot|seguridad industrial|ciber seguridad industrial",
        "min_years": 3,
        "weight": 0.3  # 30% de este grupo
    }
}

# Ejemplo de uso con perfil de Seguridad OT
if __name__ == '__main__':


    # Descripción del puesto de trabajo
    job_desc = """
    Buscamos especialista en Seguridad OT con experiencia en entornos industriales.
    Requisitos:
    - Conocimientos profundos de redes industriales (SCADA, ICS)
    - Experiencia con protocolos industriales (Modbus, DNP3)
    - Certificaciones relevantes (ISA/IEC 62443, GICSP)
    - 3+ años de experiencia en ciberseguridad industrial
    """
    # Ejemplo con perfil personalizado
    mi_perfil = {
        "keywords": {
            "keywords": ["python", "seguridad", "redes"],
            "weight": 0.5
        }
    }
    OT_SECURITY_PROFILE = {
        "keywords": {
            "keywords": [
                # Tecnologías OT/ICS
                "scada", "ics", "plc", "rtu", "hmi", "dmz industrial",
                "modbus", "dnp3", "opc ua", "bacnet", "profibus",
                # Seguridad específica
                "seguridad ot", "redes industriales", "firewall industrial",
                "zona desmilitarizada", "nerc cip", "isa/iec 62443",
                "seguridad entornos críticos", "protocolos industriales",
                # Habilidades técnicas
                "análisis de tráfico industrial", "hardening dispositivos",
                "monitoreo pasivo", "segmentación red", "backup configuraciones",
                "respuesta a incidentes OT", "redundancia sistemas"
            ],
            "weight": 0.45  # 45% del score total
        },
        "certifications": {
            "items": [
                "ISA/IEC 62443 Cybersecurity Fundamentals Specialist",
                "GIAC GICSP (Global Industrial Cyber Security Professional)",
                "Certified SCADA Security Architect (CSSA)",
                "SANS ICS410: ICS/SCADA Security Essentials",
                "CRISC (Certified in Risk and Information Systems Control)",
                "OSHA 30-Hour General Industry (Seguridad física)",
                "CCNA Industrial (Cisco Certified Network Associate)"
            ],
            "weight": 0.25  # 25% del score
        },
        "experience": {
            "requirements": {
                "min_years": 3,
                "fields": [
                    "entornos industriales (energía, agua, manufactura)",
                    "implantación dmz industriales",
                    "gestión de vulnerabilidades en sistemas legacy",
                    "integración sistemas it/ot",
                    "forensia digital industrial"
                ]
            },
            "weight": 0.30  # 30% del score
        },
        "tools": {
            "items": [
                "Claroty", "Nozomi Networks", "Tenable.ot",
                "Dragos Platform", "Siemens Siveillance",
                "PAS Cyber Integrity", "Waterfall Security",
                "Wireshark con plugins industriales"
            ],
            "weight": 0.15  # 15% (parte del 45% de keywords)
        },
        "regulatory": {
            "items": [
                "NERC CIP", "NIST SP 800-82", "IEC 62443",
                "RG 5.71 (Nuclear Regulatory Commission)",
                "Directiva NIS 2 (UE)", "ANSI/ISA-95"
            ],
            "weight": 0.10  # 10% (parte del 45% keywords)
        }
    }

    OT_SECURITY_PROFILE_CORREGIDO = {
        "keywords": {
            "keywords": [
                "scada", "ics", "plc", "rtu", "hmi", "dmz industrial",
                "modbus", "dnp3", "opc ua", "seguridad ot", "redes industriales", "ot security"
            ],
            "weight": 0.45
        },
        "certifications": {
            "items": ["ISA/IEC 62443", "GICSP", "CSSA", "CRISC","XSOAR","PCI DSS"],
            "weight": 0.25
        },
        "experience": {
            "field": "seguridad ot|redes industriales|scada|ics|plc",
            "min_years": 3,
            "weight": 0.30
        }
    }

    # Inicialización CORRECTA con argumentos
    evaluator = CVEvaluator(job_description="Buscamos experto en seguridad",profile_requirements=OT_SECURITY_PROFILE_CORREGIDO)

    # Crear evaluador con el perfil de Seguridad OT
    #evaluator = CVEvaluator(job_desc, OT_SECURITY_PROFILE)

    # Evaluar todos los CVs en un directorio
    results = evaluator.evaluate_directory(r"C:\Users\diego\Desktop\CVsTest")

    # Mostrar resultados
    df = pd.DataFrame(results)
    print(df.head(10))


    # Guardar resultados en CSV
    df.to_csv("resultados_seguridad_ot.csv", index=False)