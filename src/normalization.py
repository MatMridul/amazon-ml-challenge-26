"""
ML Challenge 2026 — Text Normalization & Field Extraction (Version 1.1)

Multi-view representations:
- Unicode NFC normalization
- Language-aware punctuation stripping preserving combining marks (accents, matras)
- Order- and position-independent legal entity token removal (English, Hindi, French)
- Postal / PIN / ZIP extraction (India 6-digit, US 5-digit, France 5-digit)
- Numeric token extraction (house/plot/street numbers)
- Token sets & 3-gram generators
"""

import re
import sys
import unicodedata
from typing import List, Set, Tuple, Optional

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Legal entity tokens to remove regardless of position/order
LEGAL_TOKENS = {
    # English
    "private", "pvt", "limited", "ltd", "inc", "incorporated",
    "corp", "corporation", "llc", "llp", "co", "company", "ms",
    # Hindi transliterations & abbreviations
    "प्राइवेट", "लिमिटेड", "प्रा", "लि", "एंटरप्राइजेज", "एंटरप्राइज", "इंडस्ट्रीज",
    # French
    "sarl", "sas", "sasu", "sa", "eurl", "snc", "sci"
}

# Regex to strip M/s prefix
MS_PREFIX_REGEX = re.compile(r"\bm\s*[/.]?\s*s\b", re.IGNORECASE)

# Address abbreviation map
ADDRESS_ABBR = {
    "rd": "road",
    "st": "street",
    "ave": "avenue",
    "dr": "drive",
    "blvd": "boulevard",
    "ln": "lane",
    "pl": "place",
    "ct": "court",
    "pkwy": "parkway",
    "hwy": "highway",
    "fl": "floor",
    "bldg": "building",
    "ste": "suite",
    "apt": "apartment",
    "no": "number",
    "opp": "opposite",
    "nr": "near",
    "tq": "taluk",
    "dist": "district",
}

# Postal code patterns
PIN_INDIA_REGEX = re.compile(r"\b([1-9][0-9]{5})\b")
ZIP_US_REGEX = re.compile(r"\b([0-9]{5})(?:-[0-9]{4})?\b")
POSTAL_FR_REGEX = re.compile(r"\b([0-9]{5})\b")


def strip_punctuation_unicode(text: str) -> str:
    """
    Strips punctuation and symbols while preserving:
    - Letters (all alphabets: Latin, Devanagari, etc.)
    - Numbers
    - Nonspacing combining marks (Devanagari matras, French accents)
    """
    if not text:
        return ""
    chars = []
    for c in text:
        cat = unicodedata.category(c)
        if cat.startswith(("P", "S")) and c != "&":
            chars.append(" ")
        else:
            chars.append(c)
    res = "".join(chars)
    return re.sub(r"\s+", " ", res).strip()


def clean_text(text: Optional[str]) -> str:
    """Basic unicode normalization and whitespace stripping."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", str(text))
    text = re.sub(r"[\t\r\n]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_name(name: Optional[str]) -> Tuple[str, str, str]:
    """
    Returns:
        norm_name: lowercased, punctuation-cleaned name
        core_name: name with legal entity terms stripped (position-independent)
        compact_name: purely alphanumeric characters
    """
    if not name:
        return "", "", ""
    cleaned = clean_text(name).lower()
    cleaned = cleaned.replace("&", " and ")
    
    # Strip M/s prefix
    cleaned = MS_PREFIX_REGEX.sub(" ", cleaned)
    
    punct_cleaned = strip_punctuation_unicode(cleaned)

    # Token-level legal entity stripping (position independent)
    words = punct_cleaned.split()
    clean_words = [w for w in words if w not in LEGAL_TOKENS]
    if clean_words:
        core = " ".join(clean_words)
    else:
        core = punct_cleaned

    # Compact alphanumeric (strips spaces and remaining non-letters/numbers)
    compact = "".join(
        c for c in punct_cleaned
        if not unicodedata.category(c).startswith(("Z", "C", "P", "S"))
    )

    return punct_cleaned, core, compact


def normalize_address(address: Optional[str], country: Optional[str] = None) -> Tuple[str, Optional[str], List[str], List[str]]:
    """
    Returns:
        norm_address: normalized address string
        postal_code: extracted postal/PIN/ZIP code (or None)
        numeric_tokens: list of numeric strings found in address
        address_tokens: cleaned list of word tokens
    """
    if not address:
        return "", None, [], []

    cleaned = clean_text(address).lower()
    cleaned = cleaned.replace("&", " and ")
    
    # Extract postal code based on country
    postal_code = None
    if country == "India":
        m = PIN_INDIA_REGEX.findall(cleaned)
        if m:
            postal_code = m[-1]
    elif country == "US":
        m = ZIP_US_REGEX.findall(cleaned)
        if m:
            postal_code = m[-1]
    elif country == "France":
        m = POSTAL_FR_REGEX.findall(cleaned)
        if m:
            postal_code = m[-1]
    else:
        m = re.findall(r"\b([0-9]{5,6})\b", cleaned)
        if m:
            postal_code = m[-1]

    # Clean punctuation
    punct_cleaned = strip_punctuation_unicode(cleaned)
    words = punct_cleaned.split()
    expanded_words = [ADDRESS_ABBR.get(w, w) for w in words]
    norm_address = " ".join(expanded_words)

    # Extract numeric tokens (digits, plot numbers like 570/13 or 448a)
    num_tokens = re.findall(r"\b\d+[a-z]?\b|\b\d+/\d+\b", norm_address)
    unique_nums = list(dict.fromkeys(num_tokens))

    # Clean word tokens (length >= 2, non-numeric)
    token_words = [w for w in expanded_words if len(w) >= 2 and not w.isdigit()]

    return norm_address, postal_code, unique_nums, token_words


def get_character_ngrams(text: str, n: int = 3) -> Set[str]:
    """Generates character n-grams from text."""
    if not text:
        return set()
    compact = "".join(
        c for c in text.lower()
        if not unicodedata.category(c).startswith(("Z", "C", "P", "S"))
    )
    if len(compact) < n:
        return {compact} if compact else set()
    return {compact[i:i+n] for i in range(len(compact) - n + 1)}


DUMMY_NUMS = {"0000000000", "1111111111", "9999999999", "1234567890", "0123456789", "9876543210"}
PHONE_REGEX = re.compile(r'(?:\+?91[\s-]?)?([6-9]\d{9})\b|\b(\d{3}[-.\s]\d{3}[-.\s]\d{4})\b')
GSTIN_REGEX = re.compile(r'\b([0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1})\b')
CIN_REGEX = re.compile(r'\b([LU][0-9]{5}[A-Z]{2}[0-9]{4}[A-Z]{3}[0-9]{6})\b')
SIREN_REGEX = re.compile(r'\b([0-9]{9})\b')
STOP_WORDS_2G = {"the", "and", "of", "for", "in", "at", "to", "on", "by", "with", "a", "an", "de", "la", "le", "et", "du", "des"}


def extract_identifiers(text: Optional[str], country: str = "India") -> Tuple[Set[str], Set[str]]:
    """
    Extracts high-confidence structured identifiers from free text:
    Returns (phones, corporate_ids)
    """
    if not text:
        return set(), set()
    text_upper = text.upper()
    phones = set()
    corp_ids = set()

    # Phones
    for match in PHONE_REGEX.finditer(text):
        m = match.group(1) or match.group(2)
        if m:
            clean_digits = re.sub(r"\D", "", m)
            if len(clean_digits) == 10 and clean_digits not in DUMMY_NUMS:
                phones.add(clean_digits)

    # Corporate IDs
    if country == "India":
        for match in GSTIN_REGEX.finditer(text_upper):
            corp_ids.add(match.group(1))
        for match in CIN_REGEX.finditer(text_upper):
            corp_ids.add(match.group(1))
    elif country == "France":
        for match in SIREN_REGEX.finditer(text):
            digits = match.group(1)
            if digits not in DUMMY_NUMS and not digits.startswith("00"):
                corp_ids.add(digits)

    return phones, corp_ids


def extract_name_2grams(core_name: str) -> List[str]:
    """
    Extracts order-independent token pairs from core name:
    e.g. 'hendricks flowers' -> ['flowers_hendricks']
    """
    if not core_name:
        return []
    tokens = [t for t in core_name.split() if len(t) >= 3 and t not in STOP_WORDS_2G]
    if len(tokens) < 2:
        return []
    grams = []
    tokens = tokens[:5]  # first 5 significant tokens
    for i in range(len(tokens)):
        for j in range(i + 1, len(tokens)):
            pair = sorted([tokens[i], tokens[j]])
            grams.append(f"{pair[0]}_{pair[1]}")
    return grams


if __name__ == "__main__":
    test_cases = [
        ("Orelee's Barbershop", "1795 Westchester Drive, High Point, NC 27262", "US"),
        ("राम मार्केटिंग प्राइवेट लिमिटेड", "KH NO. -570/13, NEW DELHI, WEST DELHI, Delhi 110041", "India"),
        ("M/s PERFECT TRADING PRIVATE LIMITED", "Plot 448A, Udyog Vihar Phase V, Gurugram, HR", "India"),
        ("Apex India Private Ltd", "Patna, Bihar", "India"),
        ("Pvt. Modi Hospitality Limited", "Mumbai, Maharashtra", "India"),
        ("Boulangerie-Pâtisserie SARL", "12 Rue de la Paix, 75002 Paris", "France")
    ]
    print("Testing normalization V1.1:")
    for name, addr, country in test_cases:
        norm_name, core_name, compact = normalize_name(name)
        norm_addr, postal, nums, tokens = normalize_address(addr, country)
        print(f"[{country}] Name: '{name}' -> Core: '{core_name}' | Compact: '{compact}'")
    print("Normalization tests completed.")
