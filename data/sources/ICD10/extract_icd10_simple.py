#!/usr/bin/env python3
"""
Simple ICD-10 Disease Code Extractor

This script extracts disease codes and Vietnamese names from the ICD-10 CSV file.
- Column Q (index 16): Disease code with dots (e.g., A00.0)
- Column R (index 17): Disease code without dots (e.g., A000)
- Column T (index 19): Vietnamese disease name
"""

import csv
import json
from pathlib import Path

def extract_icd10_codes_simple(csv_file_path: str):
    """
    Simple extraction of ICD-10 disease codes and Vietnamese names.
    """
    
    print("🏥 Extracting ICD-10 disease codes and Vietnamese names...")
    
    diseases = []
    
    try:
        with open(csv_file_path, 'r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            
            # Skip header rows (first 3 rows)
            for _ in range(3):
                next(csv_reader)
            
            disease_count = 0
            
            for row_num, row in enumerate(csv_reader, start=4):
                # Ensure row has enough columns
                if len(row) < 20:
                    continue
                
                # Extract the specific columns we need
                code_with_dots = row[16].strip()      # Column Q
                code_without_dots = row[17].strip()   # Column R  
                vietnamese_name = row[19].strip()     # Column T
                
                # Skip rows with empty essential data
                if not code_with_dots or not vietnamese_name:
                    continue
                
                disease_entry = {
                    "code_with_dots": code_with_dots,
                    "code_without_dots": code_without_dots,
                    "vietnamese_name": vietnamese_name
                }
                
                diseases.append(disease_entry)
                disease_count += 1
                
                # Progress indicator
                if disease_count % 1000 == 0:
                    print(f"  📊 Processed {disease_count} diseases...")
    
    except Exception as e:
        print(f"❌ Error reading CSV file: {e}")
        return []
    
    print(f"✅ Successfully extracted {disease_count} disease codes")
    return diseases

def save_simple_csv(diseases, filename="icd10_diseases_simple.csv"):
    """Save extracted data as simple CSV."""
    
    output_dir = Path("extracted_data")
    output_dir.mkdir(exist_ok=True)
    
    csv_file = output_dir / filename
    
    with open(csv_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow(['code_with_dots', 'code_without_dots', 'vietnamese_name'])
        
        # Write disease data
        for disease in diseases:
            writer.writerow([
                disease['code_with_dots'],
                disease['code_without_dots'], 
                disease['vietnamese_name']
            ])
    
    print(f"📊 Simple CSV saved to: {csv_file}")
    return csv_file

def save_json(diseases, filename="icd10_diseases_simple.json"):
    """Save extracted data as JSON."""
    
    output_dir = Path("extracted_data")
    output_dir.mkdir(exist_ok=True)
    
    json_file = output_dir / filename
    
    data = {
        "metadata": {
            "total_diseases": len(diseases),
            "extraction_date": "2025-11-26",
            "source": "ICD-10 Vietnam Ministry of Health"
        },
        "diseases": diseases
    }
    
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"💾 JSON saved to: {json_file}")
    return json_file

def create_lookup_dict(diseases):
    """Create simple lookup dictionaries."""
    
    code_to_name_dots = {}
    code_to_name_no_dots = {}
    name_to_codes = {}
    
    for disease in diseases:
        code_dots = disease['code_with_dots']
        code_no_dots = disease['code_without_dots']
        name = disease['vietnamese_name']
        
        if code_dots:
            code_to_name_dots[code_dots] = name
        
        if code_no_dots:
            code_to_name_no_dots[code_no_dots] = name
        
        if name:
            name_to_codes[name] = {
                'code_with_dots': code_dots,
                'code_without_dots': code_no_dots
            }
    
    lookup_data = {
        "code_with_dots_to_name": code_to_name_dots,
        "code_without_dots_to_name": code_to_name_no_dots, 
        "name_to_codes": name_to_codes,
        "statistics": {
            "total_diseases": len(diseases),
            "codes_with_dots": len(code_to_name_dots),
            "codes_without_dots": len(code_to_name_no_dots)
        }
    }
    
    return lookup_data

def print_summary(diseases):
    """Print extraction summary."""
    
    print("\n" + "="*50)
    print("📋 EXTRACTION SUMMARY")
    print("="*50)
    print(f"📈 Total Diseases: {len(diseases):,}")
    
    # Sample diseases
    print(f"\n🔍 SAMPLE DISEASES (first 5):")
    for i, disease in enumerate(diseases[:5], 1):
        print(f"  {i}. {disease['code_with_dots']} ({disease['code_without_dots']}) - {disease['vietnamese_name']}")
    
    # Code statistics
    with_dots = sum(1 for d in diseases if d['code_with_dots'])
    without_dots = sum(1 for d in diseases if d['code_without_dots'])
    
    print(f"\n📊 CODE STATISTICS:")
    print(f"  🔹 Codes with dots: {with_dots:,}")
    print(f"  🔸 Codes without dots: {without_dots:,}")

def main():
    """Main execution function."""
    
    csv_file = "attachfile001.csv"
    
    print("🚀 Starting Simple ICD-10 Disease Code Extraction")
    print(f"📁 Input file: {csv_file}")
    
    # Check if file exists
    if not Path(csv_file).exists():
        print(f"❌ Error: File '{csv_file}' not found!")
        return
    
    # Extract diseases
    diseases = extract_icd10_codes_simple(csv_file)
    
    if not diseases:
        print("❌ No diseases extracted.")
        return
    
    # Save in multiple formats
    print("\n💾 Saving extracted data...")
    save_simple_csv(diseases)
    save_json(diseases)
    
    # Create and save lookup tables
    print("🔍 Creating lookup tables...")
    lookup_data = create_lookup_dict(diseases)
    
    output_dir = Path("extracted_data")
    lookup_file = output_dir / "icd10_lookup.json"
    with open(lookup_file, 'w', encoding='utf-8') as f:
        json.dump(lookup_data, f, ensure_ascii=False, indent=2)
    print(f"🔍 Lookup tables saved to: {lookup_file}")
    
    # Print summary
    print_summary(diseases)
    
    print(f"\n✅ Extraction completed!")
    print(f"📂 Files saved in 'extracted_data/' directory")

if __name__ == "__main__":
    main()