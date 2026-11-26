import re
import csv
import io
import os
import logging
import unicodedata
from collections import Counter 

Malayalam_regex = re.compile(r'[\u0D00-\u0D7F]+') # malayalam unicode block validation
Required_cols = ("review_text", "domain",'label')
valid_labels = {'positive', 'negative', 'neutral'}

MIN_RATIO = 0.3  # minimum malayalam ratio threshold
BAD_CHAR = '\ufffd'  # replacement character for invalid UTF-8 bytes

class IngestionError(Exception):
    """fatal problem it can be missing file,empty file,wrong header"""

class RowError(Exception):
    """one row had problem """
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def is_malayalam(text):
    """
    Check if the given text contains Malayalam characters.
    args: text(str), returns: bool
    """
    if not text or not str(text).strip():
        return False
    return bool(Malayalam_regex.search(text))

def malayalam_ratio(text):
    """ anything in the malayalam block count as letter """
    mal = sum(len(m) for m in Malayalam_regex.findall(text))
    other = sum(1 for c in text if c.isalpha() and not Malayalam_regex.match(c))
    return mal / (mal + other) if (mal + other) else 0

def validate_row(row, min_ratio):
    """raises row error with reason is the row is bad"""
    for col in Required_cols:
        if row.get(col) is None:
            raise RowError("MISSING_FIELD",f"Missing Value for: {col}")

    text = row['review_text']
    if not text.strip():
        raise RowError("EMPTY_TEXT", "Empty review_text field")
    if BAD_CHAR in text:
        raise RowError("ENCODING_ERROR", f"invalid UTF-8 bytes")
    if not is_malayalam(text):
        raise RowError("MISSING_MALAYALAM", "Missing valid malayalam unicode")
    ratio = malayalam_ratio(text)
    if ratio < min_ratio:
        raise RowError("LOW_MALAYALAM_RATIO", f"Malayalam ratio {ratio:.2f} is below threshold {min_ratio:.2f}")

    if row['label'].strip().lower() not in valid_labels:
        raise RowError("INVALID_LABEL", f"Invalid label: {row['label']}")

    domain_text = row.get('domain')
    if not domain_text or not str(domain_text).strip():
        raise RowError("EMPTY_DOMAIN", "Empty domain field")
    
def make_parent_dir(filepath):
    parent = os.path.dirname(filepath)
    if parent:
        os.makedirs(parent, exist_ok=True)
        
def ingest_data(raw_filepath, output_filepath, log_filepath, min_ratio=MIN_RATIO):

    """
    read raw data line by line, validate malayalam unicode, exception handling,export clean CSV
    """
    #to ensure the output and log directories exist
    make_parent_dir(output_filepath)
    make_parent_dir(log_filepath)

    logger = logging.getLogger("ingestion_audit")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(log_filepath, 'w', encoding='utf-8')
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)

    valid_record = []
    reasons = Counter()
    total_count = 0


    print(f"Starting ingestion from {raw_filepath}...")
    logger.info(f"Audit started for raw file: {raw_filepath}")


    try:
        try:
            with open(raw_filepath, 'rb') as infile:
                raw_bytes = infile.read()

        except OSError as e:
            logger.error(f"Can't open input file: {e}")
            raise IngestionError(f"Can't read the filepath: {raw_filepath}. Error: {e}")


        content = raw_bytes.decode('utf-8-sig', errors='replace')
        if not content.strip():
            logger.error("Input file is empty after decoding.")
            raise IngestionError(f"Input file  is empty ")


        reader = csv.DictReader(io.StringIO(content,newline=''))
        missing = [c for c in Required_cols if c not in (reader.fieldnames or [])]

        if missing:
            logger.error(f"Missing required columns: {missing}")
            raise IngestionError(f"Missing required columns: {missing}")


        for row_n, row in enumerate(reader, start=1):
            total_count += 1
            try:
                validate_row(row, min_ratio)
                text =unicodedata.normalize('NFC', row['review_text'].strip())
                valid_record.append({ 'review_text': text, 'domain': row['domain'].strip().lower(), 'label': row['label'].strip().lower(), 'token_count': len(text.split()) ,'char_count': len(text) })
            except RowError as e:
                reasons[e.code] += 1
                logger.warning ("Row %d | %s | %s | Raw Data: %s", row_n, e.code, e, row)
            except Exception as e:
                reasons["UNEXPECTED_ERROR"] += 1
                logger.error("Row %d | Unexpected error: %s | Raw Data: %s", row_n, e, row)

        fieldnames = ['review_text', 'domain', 'label', 'token_count', 'char_count']
        with open(output_filepath, 'w', encoding='utf-8', newline='') as outfile:
            writer = csv.DictWriter(outfile,fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(valid_record)

        rejected_count = total_count - len(valid_record)

        logger.info("Total rows: %d | Valid: %d | Rejected: %d", total_count, len(valid_record), rejected_count)
        for code, n in sorted(reasons.items()):
            logger.info("  %s: %d", code, n)

        print("Ingestion Summary\n")
        print(f"Total Rows processed: {total_count}")
        print(f"Valid Rows: {len(valid_record)}")
        print(f"Rejected Rows: {rejected_count}")
        for code, n in sorted(reasons.items()):
            print(f"  {code}: {n}")

        return{'total': total_count, 'valid': len(valid_record), 'rejected': rejected_count, 'reasons': dict(reasons)}

    finally:
        handler.close()
        logger.handlers = []

if __name__ == "__main__":
    ingest_data(
        "data/raw/indic_sentiment_malayalam_raw.csv", 
        "data/processed/clean_malayalam_sentiment.csv", 
        "data/processed/audit.log"
    )