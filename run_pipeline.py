import argparse
import os
import sys
from src.ingestion import ingest_data

def main():
    parser = argparse.ArgumentParser(description = "Malayalam Sentiment Pipeline")
    parser.add_argument("--input", required=True, help="Path to the raw csv")
    parser.add_argument("--output-dir",  required=True, help="Output directory")
    parser.add_argument("--report-dir",  required=True, help="Report directory")
    args = parser.parse_args()

    clean_csv_path = os.path.join(args.output_dir, "clean_malayalam_sentiment.csv")
    audit_log_path = os.path.join(args.report_dir, "sentiment_ingestion_audit.log")

    print(f"--STARTING PIPELINE--")
    print(f"Reading raw data from: {args.input}")

    ingest_data(raw_filepath = args.input,output_filepath = clean_csv_path, log_filepath = audit_log_path)

    print("---PIPELINE COMPLETED---")

    sys.exit(0)

if __name__ == "__main__":
    main()