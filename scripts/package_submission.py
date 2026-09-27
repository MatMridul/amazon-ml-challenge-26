"""
ML Challenge 2026 — Submission Packaging Script
Team: Claude's Plan

Bundles the final competition submission:
Claudes_Plan_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       ├── README.md
│       └── requirements.txt
└── Documentation_template.md
"""

import os
import sys
import shutil
import zipfile

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
SUBMISSION_ZIP = os.path.join(BASE_DIR, "Claudes_Plan_submission.zip")


def package_submission():
    print("=" * 80)
    print("PACKAGING OFFICIAL SUBMISSION ZIP: Claudes_Plan_submission.zip")
    print("=" * 80)

    # 1. Verify required files exist
    matching_tsv = os.path.join(OUTPUT_DIR, "matching_results.tsv")
    candidate_tsv = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
    doc_template = os.path.join(BASE_DIR, "Documentation_template.md")

    if not os.path.exists(matching_tsv):
        raise FileNotFoundError(f"Missing: {matching_tsv}")
    if not os.path.exists(candidate_tsv):
        raise FileNotFoundError(f"Missing: {candidate_tsv}")
    if not os.path.exists(doc_template):
        raise FileNotFoundError(f"Missing: {doc_template}")

    # Generate pinned requirements.txt
    req_path = os.path.join(BASE_DIR, "requirements.txt")
    with open(req_path, "w", encoding="utf-8") as f:
        f.write("polars>=1.0.0\npyarrow>=15.0.0\npandas>=2.0.0\nrapidfuzz>=3.8.0\nlightgbm>=4.0.0\nscikit-learn>=1.4.0\ntqdm>=4.66.0\n")

    readme_path = os.path.join(BASE_DIR, "README.md")
    src_dir = os.path.join(BASE_DIR, "src")

    print(f"Creating zip archive: {SUBMISSION_ZIP}...")
    with zipfile.ZipFile(SUBMISSION_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        # 1. output/
        zf.write(matching_tsv, arcname="output/matching_results.tsv")
        zf.write(candidate_tsv, arcname="output/candidate_pairs.tsv")
        print("  Added: output/matching_results.tsv")
        print("  Added: output/candidate_pairs.tsv")

        # 2. Documentation_template.md
        zf.write(doc_template, arcname="Documentation_template.md")
        print("  Added: Documentation_template.md")

        # 3. code/business_entity_resolution/
        zf.write(readme_path, arcname="code/business_entity_resolution/README.md")
        zf.write(req_path, arcname="code/business_entity_resolution/requirements.txt")
        print("  Added: code/business_entity_resolution/README.md")
        print("  Added: code/business_entity_resolution/requirements.txt")

        for root, dirs, files in os.walk(src_dir):
            if "__pycache__" in root:
                continue
            for file in files:
                if file.endswith(".py"):
                    full_p = os.path.join(root, file)
                    rel_p = os.path.relpath(full_p, src_dir)
                    arc_p = f"code/business_entity_resolution/src/{rel_p.replace(os.sep, '/')}"
                    zf.write(full_p, arcname=arc_p)
                    print(f"  Added: {arc_p}")

    zip_size_mb = os.path.getsize(SUBMISSION_ZIP) / (1024 * 1024)
    print("\n" + "=" * 80)
    print(f"SUBMISSION PACKAGE READY: {SUBMISSION_ZIP} ({zip_size_mb:.2f} MB)")
    print("=" * 80)


if __name__ == "__main__":
    package_submission()
