import os
import re
from typing import List, Dict, Any
from pathlib import Path
from pypdf import PdfReader

# Line-anchored Major Section patterns
SECTION_LINE_PATTERNS = [
    (re.compile(r"^\s*SECTION\s+A\.?\s*DEFINITIONS", re.I), "SECTION A. DEFINITIONS"),
    (re.compile(r"^\s*SECTION\s+B\.?\s*BENEFITS", re.I), "SECTION B. BENEFITS"),
    (re.compile(r"^\s*SECTION\s+C\.?\s*WAITING\s*PERIOD\s*AND\s*EXCLUSIONS", re.I), "SECTION C. WAITING PERIOD AND EXCLUSIONS"),
    (re.compile(r"^\s*SECTION\s+D\.?\s*GENERAL\s*TERMS\s*AND\s*CLAUSES", re.I), "SECTION D. GENERAL TERMS AND CLAUSES"),
    (re.compile(r"^\s*SECTION\s+E\.?\s*OTHER\s*TERMS\s*AND\s*CLAUSES", re.I), "SECTION E. OTHER TERMS AND CLAUSES"),
    (re.compile(r"^\s*Annexure\s+A\b", re.I), "Annexure A - Contact details of Offices of Insurance Ombudsman"),
    (re.compile(r"^\s*Annexure\s+B\s*[-–]\s*Items\s*for\s*which\s*Coverage\s*is\s*not\s*available", re.I), "Annexure B - Non-Medical Expenses (Items Not Covered)"),
    (re.compile(r"^\s*Annexure\s+C\s*[-–]\s*Plan\s*Chart", re.I), "Annexure C - Plan Chart: Schedule of Benefits"),
    (re.compile(r"^\s*Add\s*on\s*[-–]\s*Covers:", re.I), "Add on - Covers"),
]

# Specific Clause and Heading patterns
CLAUSE_LINE_PATTERNS = [
    # Section A Definitions
    (re.compile(r"^\s*1\.1\.\s*Standard\s*Definitions", re.I), "Section 1.1: Standard Definitions"),
    (re.compile(r"^\s*1\.2\.\s*Specific\s*Definitions", re.I), "Section 1.2: Specific Definitions"),
    (re.compile(r"^\s*Def\.\s*\d+\.\s*([A-Za-z\s\(\)]+?)(?=\s+means|\s+is|\s+refers|\s*\n)", re.I), "Definition"),

    # Section B Benefits (Base Covers)
    (re.compile(r"^\s*1\.\s*Base\s*Coverage", re.I), "Section B.1: Base Coverage"),
    (re.compile(r"^\s*1\.1\.\s*Hospitalization\s*Expenses", re.I), "Clause B.1.1: Hospitalization Expenses (Room Rent & ICU)"),
    (re.compile(r"^\s*1\.1\.1\.\s*Other\s*Expenses", re.I), "Clause B.1.1.1: Other Expenses (Road Ambulance & Day Care)"),
    (re.compile(r"^\s*1\.2\.\s*Home\s*Health\s*Care", re.I), "Clause B.1.2: Home Health Care"),
    (re.compile(r"^\s*1\.3\.\s*Domiciliary\s*Hospitalization", re.I), "Clause B.1.3: Domiciliary Hospitalization"),
    (re.compile(r"^\s*1\.4\.\s*AYUSH\s*Treatment", re.I), "Clause B.1.4: AYUSH Treatment"),
    (re.compile(r"^\s*1\.5\.\s*Pre-Hospitalization\s*Expenses", re.I), "Clause B.1.5: Pre-Hospitalization Expenses"),
    (re.compile(r"^\s*1\.6\.\s*Post-Hospitalization\s*Expenses", re.I), "Clause B.1.6: Post-Hospitalization Expenses"),
    (re.compile(r"^\s*1\.7\.\s*Organ\s*Donor\s*Expenses", re.I), "Clause B.1.7: Organ Donor Expenses"),
    (re.compile(r"^\s*1\.8\.\s*Cumulative\s*Bonus", re.I), "Clause B.1.8: Cumulative Bonus"),

    # Section B Optional Covers
    (re.compile(r"^\s*2\.\s*Optional\s*Covers", re.I), "Section B.2: Optional Covers Overview"),
    (re.compile(r"^\s*2\.1\.\s*Emergency\s*Air\s*Ambulance", re.I), "Clause B.2.1: Emergency Air Ambulance"),
    (re.compile(r"^\s*2\.2\.\s*Daily\s*Cash\s*for\s*Shared\s*Room", re.I), "Clause B.2.2: Daily Cash for Shared Room"),
    (re.compile(r"^\s*2\.3\.\s*Protect\s*Benefit", re.I), "Clause B.2.3: Protect Benefit"),
    (re.compile(r"^\s*2\.4\.\s*Plus\s*Benefit", re.I), "Clause B.2.4: Plus Benefit"),
    (re.compile(r"^\s*2\.5\.\s*Secure\s*Benefit", re.I), "Clause B.2.5: Secure Benefit"),
    (re.compile(r"^\s*2\.6\.\s*Automatic\s*Restore\s*Benefit", re.I), "Clause B.2.6: Automatic Restore Benefit"),
    (re.compile(r"^\s*2\.7\.\s*Aggregate\s*Deductible", re.I), "Clause B.2.7: Aggregate Deductible"),
    (re.compile(r"^\s*2\.8\.\s*E-Opinion\s*for\s*Critical\s*Illness", re.I), "Clause B.2.8: E-Opinion for Critical Illness"),
    (re.compile(r"^\s*2\.9\.\s*Global\s*Health\s*Cover\s*\(Emergency", re.I), "Clause B.2.9: Global Health Cover (Emergency Only)"),
    (re.compile(r"^\s*2\.10\.\s*Global\s*Health\s*Cover\s*\(Emergency\s*&\s*Planned", re.I), "Clause B.2.10: Global Health Cover (Emergency & Planned)"),
    (re.compile(r"^\s*2\.11\.\s*Overseas\s*Travel\s*Secure", re.I), "Clause B.2.11: Overseas Travel Secure"),
    (re.compile(r"^\s*2\.12\s*PED\s*waiting\s*period\s*modification", re.I), "Clause B.2.12: PED Waiting Period Modification"),
    (re.compile(r"^\s*2\.13\s*Modification\s*of\s*Room\s*Rent", re.I), "Clause B.2.13: Modification of Room Rent"),
    (re.compile(r"^\s*2\.14\s*Modification\s*of\s*Pre-Hospitalization", re.I), "Clause B.2.14: Modification of Pre-Hospitalization Days"),
    (re.compile(r"^\s*2\.15\s*Modification\s*of\s*Post-Hospitalization", re.I), "Clause B.2.15: Modification of Post-Hospitalization Days"),
    (re.compile(r"^\s*2\.16\s*Modification\s*of\s*Cumulative\s*Bonus", re.I), "Clause B.2.16: Modification of Cumulative Bonus"),
    (re.compile(r"^\s*3\.\s*Renewal\s*Benefit\s*[-–]\s*Preventive\s*Health\s*Check-up", re.I), "Clause B.3: Preventive Health Check-up"),

    # Section C Waiting Periods & Exclusions
    (re.compile(r"^\s*1\.\s*Waiting\s*Periods", re.I), "Section C.1: Waiting Periods Overview"),
    (re.compile(r"^\s*a\.\s*Pre-Existing\s*Diseases:\s*Code\s*[-–]\s*Excl01", re.I), "Clause C.1.a: Pre-Existing Diseases (Code: Excl01)"),
    (re.compile(r"^\s*b\.\s*Specified\s*Disease/Procedure\s*waiting\s*period:\s*Code\s*[-–]\s*Excl02", re.I), "Clause C.1.b: Specified Disease/Procedure Waiting Period (Code: Excl02)"),
    (re.compile(r"^\s*c\.\s*30-day\s*waiting\s*period:\s*Code\s*[-–]\s*Excl03", re.I), "Clause C.1.c: 30-Day Waiting Period (Code: Excl03)"),
    (re.compile(r"^\s*2\.\s*Standard\s*Exclusions", re.I), "Section C.2: Standard Exclusions Overview"),
    (re.compile(r"^\s*a\.\s*Investigation\s*&\s*Evaluation:\s*Code\s*[-–]?\s*Excl04", re.I), "Clause C.2.a: Investigation & Evaluation (Code: Excl04)"),
    (re.compile(r"^\s*b\.\s*Rest\s*Cure.*?:Code\s*[-–]\s*Excl05", re.I), "Clause C.2.b: Rest Cure and Rehabilitation (Code: Excl05)"),
    (re.compile(r"^\s*c\.\s*Obesity/Weight\s*control:\s*Code\s*[-–]\s*Excl06", re.I), "Clause C.2.c: Obesity/Weight Control (Code: Excl06)"),
    (re.compile(r"^\s*d\.\s*Change-of-Gender.*?:Code\s*[-–]\s*Excl07", re.I), "Clause C.2.d: Change-of-Gender Treatments (Code: Excl07)"),
    (re.compile(r"^\s*e\.\s*Cosmetic\s*or\s*plastic\s*Surgery:\s*Code\s*[-–]\s*Excl08", re.I), "Clause C.2.e: Cosmetic or Plastic Surgery (Code: Excl08)"),
    (re.compile(r"^\s*f\.\s*Hazardous\s*or\s*Adventure\s*Sports:\s*Code\s*[-–]\s*Excl09", re.I), "Clause C.2.f: Hazardous Sports (Code: Excl09)"),
    (re.compile(r"^\s*g\.\s*Breach\s*of\s*Law:\s*Code\s*[-–]\s*Excl10", re.I), "Clause C.2.g: Breach of Law (Code: Excl10)"),
    (re.compile(r"^\s*h\.\s*Excluded\s*Providers:\s*Code\s*[-–]\s*Excl11", re.I), "Clause C.2.h: Excluded Providers (Code: Excl11)"),
    (re.compile(r"^\s*i\.\s*Treatment\s*for\s*Alcoholism.*?:Code\s*[-–]\s*Excl12", re.I), "Clause C.2.i: Alcoholism and Drug Abuse (Code: Excl12)"),
    (re.compile(r"^\s*j\.\s*Treatments\s*received\s*in\s*health\s*hydros.*?:Code\s*[-–]\s*Excl13", re.I), "Clause C.2.j: Health Hydros and Spas (Code: Excl13)"),
    (re.compile(r"^\s*k\.\s*Dietary\s*supplements.*?:Code\s*[-–]\s*Excl14", re.I), "Clause C.2.k: Dietary Supplements (Code: Excl14)"),
    (re.compile(r"^\s*l\.\s*Refractive\s*Error:\s*Code\s*[-–]\s*Excl15", re.I), "Clause C.2.l: Refractive Error (Code: Excl15)"),
    (re.compile(r"^\s*m\.\s*Unproven\s*Treatments:\s*Code\s*[-–]\s*Excl16", re.I), "Clause C.2.m: Unproven Treatments (Code: Excl16)"),
    (re.compile(r"^\s*n\.\s*Sterility\s*and\s*Infertility:\s*Code\s*[-–]\s*Excl17", re.I), "Clause C.2.n: Sterility and Infertility (Code: Excl17)"),
    (re.compile(r"^\s*o\.\s*Maternity:\s*Code\s*[-–]\s*Excl18", re.I), "Clause C.2.o: Maternity (Code: Excl18)"),
    (re.compile(r"^\s*3\.\s*Specific\s*Exclusions", re.I), "Section C.3: Specific Exclusions"),

    # Section D General Terms & Clauses
    (re.compile(r"^\s*1\.1\.\s*Disclosure\s*of\s*Information", re.I), "Clause D.1.1: Disclosure of Information"),
    (re.compile(r"^\s*1\.2\.\s*Condition\s*Precedent", re.I), "Clause D.1.2: Condition Precedent to Admission of Liability"),
    (re.compile(r"^\s*1\.3\.\s*Claim\s*Settlement", re.I), "Clause D.1.3: Claim Settlement (Penal Interest)"),
    (re.compile(r"^\s*1\.5\.\s*Multiple\s*Policies", re.I), "Clause D.1.5: Multiple Policies"),
    (re.compile(r"^\s*1\.6\.\s*Moratorium\s*Period", re.I), "Clause D.1.6: Moratorium Period"),
    (re.compile(r"^\s*1\.7\.\s*Fraud", re.I), "Clause D.1.7: Fraud"),
    (re.compile(r"^\s*1\.8\.\s*Free\s*look\s*Period", re.I), "Clause D.1.8: Free Look Period"),
    (re.compile(r"^\s*1\.9\.\s*Renewal\s*of\s*Policy", re.I), "Clause D.1.9: Renewal of Policy"),
    (re.compile(r"^\s*1\.10\.\s*Portability", re.I), "Clause D.1.10: Portability"),
    (re.compile(r"^\s*1\.11\.\s*Migration", re.I), "Clause D.1.11: Migration"),
    (re.compile(r"^\s*1\.12\.\s*Cancellation", re.I), "Clause D.1.12: Cancellation"),
    (re.compile(r"^\s*1\.13\.\s*Premium\s*Payment\s*in\s*Instalments", re.I), "Clause D.1.13: Premium Payment in Instalments"),
    (re.compile(r"^\s*1\.17\.\s*Redressal\s*of\s*Grievance", re.I), "Clause D.1.17: Redressal of Grievance"),
    (re.compile(r"^\s*1\.18\.\s*Non-Disclosure\s*or\s*Misrepresentation\s*of\s*Pre-Existing", re.I), "Clause D.1.18: Non-Disclosure of Pre-Existing Disease"),
    (re.compile(r"^\s*1\.19\.\s*Utilization\s*of\s*Sum\s*Insured", re.I), "Clause D.1.19: Utilization of Sum Insured"),
    (re.compile(r"^\s*1\.20\.\s*Geography", re.I), "Clause D.1.20: Geography"),
    (re.compile(r"^\s*1\.24\.\s*Premium\s*Tier", re.I), "Clause D.1.24: Premium Tier (Tier 1 & Tier 2 Cities)"),

    # Section E Claims Procedure
    (re.compile(r"^\s*1\.1\.\s*Notification\s*of\s*a\s*Claim", re.I), "Clause E.1.1: Notification of a Claim"),
    (re.compile(r"^\s*1\.2\.\s*Procedure\s*for\s*Cashless\s*Claims\s*In\s*India", re.I), "Clause E.1.2: Procedure for Cashless Claims In India"),
    (re.compile(r"^\s*1\.6\.\s*Procedure\s*for\s*Reimbursement\s*Claims", re.I), "Clause E.1.6: Procedure for Reimbursement Claims"),
    (re.compile(r"^\s*1\.7\.\s*List\s*of\s*documents\s*required\s*for\s*a\s*Claim", re.I), "Clause E.1.7: List of Documents Required for a Claim"),
    (re.compile(r"^\s*2\.\s*Contact\s*Us", re.I), "Section E.2: Contact Us & Claims Submission Addresses"),

    # Annexures
    (re.compile(r"^\s*Annexure\s+A\b", re.I), "Annexure A: Insurance Ombudsman Contact Details"),
    (re.compile(r"^\s*Annexure\s+B\s*[-–]\s*Items\s*for\s*which\s*Coverage", re.I), "Annexure B: Non-Medical Expenses (Items Not Covered)"),
    (re.compile(r"^\s*Annexure\s+C\s*[-–]\s*Plan\s*Chart", re.I), "Annexure C: Plan Chart - Schedule of Benefits"),
    (re.compile(r"^\s*Add\s*on\s*[-–]\s*Covers:", re.I), "Add-on Covers Overview"),
]

def clean_text(text: str) -> str:
    """Normalize whitespace and remove repeating header/footer noise."""
    lines = text.splitlines()
    cleaned_lines = []
    for line in lines:
        line_str = line.strip()
        if not line_str:
            continue
        if any(term in line_str for term in ["General Insurance Company Limited", "Insurance Company Limited"]):
            continue
        if "HDFHLIP" in line_str or "my: Optima Secure" in line_str:
            continue
        if line_str.isdigit() and len(line_str) <= 3:
            continue
        cleaned_lines.append(line_str)
    return "\n".join(cleaned_lines)

def parse_policy_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Parses the policy PDF page-by-page, strictly detecting section
    and clause headers on line starts to prevent false mid-sentence triggers.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF document not found at: {pdf_path}")

    reader = PdfReader(pdf_path)
    chunks: List[Dict[str, Any]] = []

    current_section = "Preamble & Operative Clause"
    current_clause = "Preamble & Operative Provisions"

    for idx, page in enumerate(reader.pages):
        page_num = idx + 1
        raw_text = page.extract_text() or ""
        cleaned = clean_text(raw_text)

        if not cleaned:
            continue

        if page_num == 1:
            chunks.append({
                "page": 1,
                "section": "Table of Contents",
                "clause": "Table of Contents",
                "text": cleaned.strip()
            })
            continue

        lines = cleaned.splitlines()
        current_chunk_lines = []

        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue

            # Check if this line starts a new Major Section
            for pattern, sec_name in SECTION_LINE_PATTERNS:
                if pattern.search(line_str):
                    current_section = sec_name
                    # Reset current clause to section default when section changes
                    current_clause = sec_name
                    break

            # Check if this line starts a new Clause or Definition
            for pattern, clause_name in CLAUSE_LINE_PATTERNS:
                match = pattern.search(line_str)
                if match:
                    # Flush accumulated chunk under prior clause
                    if current_chunk_lines and len("\n".join(current_chunk_lines)) > 150:
                        chunks.append({
                            "page": page_num,
                            "section": current_section,
                            "clause": current_clause,
                            "text": "\n".join(current_chunk_lines).strip()
                        })
                        current_chunk_lines = []

                    if clause_name == "Definition":
                        def_term = match.group(1).strip()
                        current_clause = f"Definition: {def_term}"
                    else:
                        current_clause = clause_name
                    break

            current_chunk_lines.append(line_str)

            # Flush when buffer reaches target size (~1000 chars)
            if len("\n".join(current_chunk_lines)) >= 1000:
                chunks.append({
                    "page": page_num,
                    "section": current_section,
                    "clause": current_clause,
                    "text": "\n".join(current_chunk_lines).strip()
                })
                current_chunk_lines = []

        # Flush remainder for this page
        if current_chunk_lines:
            chunks.append({
                "page": page_num,
                "section": current_section,
                "clause": current_clause,
                "text": "\n".join(current_chunk_lines).strip()
            })

    return chunks
