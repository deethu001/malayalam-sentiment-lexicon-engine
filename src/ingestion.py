import re
import csv
import os

Malayalam_regex = re.compile(r'[\u0D00-\u0D7F]+') # malayalam unicode block validation

def is_malayalam(text):
    """
    Check if the given text contains Malayalam characters.
    args: text(str), returns: bool
    """
    if not text or not str(text).strip():
        return False
    return bool(Malayalam_regex.search(text))

def ingest_data(raw_filepath, output_filepath, log_filepath):

    """
    read raw data line by line, validate malayalam unicode, exception handling,export clean CSV
    """
    #to ensure the output and log directories exist
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    os.makedirs(os.path.dirname(log_filepath), exist_ok=True)

    valid_record = []
    rejected_count = 0
    total_count = 0

    print(f"Starting ingestion from {raw_filepath}...")

    try:
        with open(raw_filepath, 'r', encoding='utf-8') as infile, \
             open(log_filepath,'w', encoding='utf-8') as log_file:
            reader = csv.DictReader(infile)
            text_col = 'review_text' if 'review_text' in reader.fieldnames else reader.fieldnames[0]  # default to first column if 'review_text' not found
            for row_n, row in enumerate(reader, start=1):
                total_count += 1
                try:
                    text = row[text_col]
                    if not is_malayalam(text):
                        raise ValueError("Missing valid malayalam unicode")

                    valid_record.append(row)
                except(ValueError, TypeError)as e:
                    rejected_count += 1
                    log_file.write(f"Row {row_n}: {e} | Raw Data: {row}\n")
                except Exception as e:
                    rejected_count += 1
                    log_file.write(f"Row {row_n}: Unexpected error: {e} | Raw Data: {row}\n")
   
    except FileNotFoundError as e:
        print(f"Error: could not find the raw file at {raw_filepath}. ")
        return
    except UnicodeDecodeError as e:
        print(f"Error: File encodeing issue {e}")
        return

    if valid_record:
        with open(output_filepath, 'w', encoding='utf-8', newline='') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=valid_record[0].keys())
            writer.writeheader()
            writer.writerows(valid_record)

    print("Ingestion Summary\n")
    print(f"Total Rows processed: {total_count}")
    print(f"Valid Rows: {len(valid_record)}")
    print(f"Rejected Rows: {rejected_count}")