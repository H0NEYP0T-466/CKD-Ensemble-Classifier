#!/usr/bin/env python3
"""
Robust ARFF file extraction with data cleaning for Chronic Kidney Disease dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re

def clean_arff_value(value, attr_type):
    """Clean ARFF values based on attribute type"""
    if pd.isna(value) or value == '?' or value == '':
        return np.nan

    # Decode bytes if needed
    if isinstance(value, bytes):
        value = value.decode('utf-8')

    # Clean whitespace
    value = str(value).strip()

    # Handle numeric types
    if attr_type == 'numeric':
        try:
            # Remove any non-numeric characters except ., -, and spaces
            value = re.sub(r'[^\d.\- ]', '', value)
            # Handle cases like "1.2 3.4" by taking first number
            if ' ' in value:
                value = value.split()[0]
            return float(value)
        except (ValueError, TypeError):
            return np.nan

    # Handle nominal types - clean and standardize
    if attr_type == 'nominal':
        # Standardize common variations
        value_lower = value.lower()

        # Standardize yes/no variations
        if value_lower in ['yes', 'y', '1']:
            return 'yes'
        if value_lower in ['no', 'n', '0']:
            return 'no'

        # Standardize normal/abnormal
        if value_lower in ['normal', 'norm']:
            return 'normal'
        if value_lower in ['abnormal', 'abn']:
            return 'abnormal'

        # Standardize present/notpresent
        if value_lower in ['present', 'pres', 'yes']:
            return 'present'
        if value_lower in ['notpresent', 'not', 'no', 'not present']:
            return 'notpresent'

        # For sg (specific gravity) values
        if value in ['1.005', '1.010', '1.015', '1.020', '1.025']:
            return value

        # For al, su values (0-5 scale)
        if value in ['0', '1', '2', '3', '4', '5']:
            return int(value)

        # Return original if it matches known patterns
        return value

    return value

def parse_arff_manually(arff_path):
    """Manually parse ARFF file to handle data quality issues"""
    print(f"Manually parsing ARFF file: {arff_path}")

    # Read the file
    with open(arff_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    # Parse header
    attributes = []
    attribute_types = {}
    data_start_idx = 0

    for i, line in enumerate(lines):
        line = line.strip()

        # Skip empty lines and comments
        if not line or line.startswith('%'):
            continue

        # Parse @relation
        if line.lower().startswith('@relation'):
            relation = line.split(maxsplit=1)[1] if ' ' in line else 'unknown'
            print(f"Relation: {relation}")

        # Parse @attribute
        elif line.lower().startswith('@attribute'):
            parts = line.split(maxsplit=2)
            if len(parts) >= 3:
                attr_name = parts[1].strip("'\"")
                attr_type = parts[2].lower()

                # Clean up nominal types
                if '{' in attr_type:
                    attr_type = 'nominal'
                elif 'numeric' in attr_type or 'real' in attr_type or 'integer' in attr_type:
                    attr_type = 'numeric'
                elif 'string' in attr_type or 'date' in attr_type:
                    attr_type = 'string'
                else:
                    attr_type = 'nominal'

                attributes.append(attr_name)
                attribute_types[attr_name] = attr_type

        # Find @data
        elif line.lower().startswith('@data'):
            data_start_idx = i + 1
            break

    print(f"Found {len(attributes)} attributes")
    print(f"Data starts at line {data_start_idx}")

    # Parse data
    data = []
    for i in range(data_start_idx, len(lines)):
        line = lines[i].strip()
        if not line or line.startswith('%'):
            continue

        # Split by comma, but handle quoted strings properly
        row = []
        current_field = ""
        in_quotes = False
        quote_char = None

        for char in line:
            if char in ['"', "'"] and not in_quotes:
                in_quotes = True
                quote_char = char
                current_field += char
            elif char == quote_char and in_quotes:
                in_quotes = False
                current_field += char
            elif char == ',' and not in_quotes:
                row.append(current_field.strip())
                current_field = ""
            else:
                current_field += char

        # Add last field
        if current_field.strip():
            row.append(current_field.strip())

        # Clean up each field
        if len(row) == len(attributes):
            cleaned_row = []
            for j, (value, attr_name) in enumerate(zip(row, attributes)):
                attr_type = attribute_types.get(attr_name, 'nominal')
                cleaned_value = clean_arff_value(value, attr_type)
                cleaned_row.append(cleaned_value)
            data.append(cleaned_row)

    # Create DataFrame
    df = pd.DataFrame(data, columns=attributes)

    # Convert numeric columns
    for attr_name, attr_type in attribute_types.items():
        if attr_type == 'numeric' and attr_name in df.columns:
            df[attr_name] = pd.to_numeric(df[attr_name], errors='coerce')

    return df, attributes, attribute_types

def display_dataset_info(df, attributes, attribute_types):
    """Display comprehensive dataset information"""
    if df is None:
        return

    print("\n" + "="*60)
    print("CHRONIC KIDNEY DISEASE DATASET INFORMATION")
    print("="*60)

    print(f"\nDataset Shape: {df.shape}")
    print(f"Total Samples: {len(df)}")
    print(f"Total Features: {len(attributes) - 1}")  # -1 for target
    print(f"Total Attributes: {len(attributes)}")

    # Get target column (assuming last column is the class)
    target_col = attributes[-1]
    feature_cols = attributes[:-1]

    print(f"\nTarget Variable: '{target_col}'")
    print(f"Target Type: {attribute_types.get(target_col, 'unknown')}")

    print("\n" + "="*60)
    print("TARGET DISTRIBUTION")
    print("="*60)

    if target_col in df.columns:
        target_counts = df[target_col].value_counts()
        print(target_counts)
        print(f"\nClass proportions:")
        for cls, count in target_counts.items():
            prop = (count / len(df)) * 100
            print(f"  {cls}: {count} samples ({prop:.1f}%)")

    print("\n" + "="*60)
    print("FEATURES SUMMARY")
    print("="*60)

    for i, col in enumerate(feature_cols, 1):
        if col in df.columns:
            dtype = df[col].dtype
            attr_type = attribute_types.get(col, 'unknown')

            # Get unique values for nominal features
            if dtype == 'object':
                unique_vals = df[col].dropna().unique()
                unique_count = len(unique_vals)
                unique_str = f"{unique_count} values"
                if unique_count <= 10:
                    try:
                        unique_str += f": {sorted(unique_vals)}"
                    except TypeError:
                        # Handle mixed types by converting to strings
                        unique_str += f": {sorted([str(v) for v in unique_vals])}"
            else:
                unique_count = df[col].nunique()
                unique_str = f"{unique_count} unique values"
                if not pd.api.types.is_numeric_dtype(df[col]):
                    unique_str += f" (non-numeric)"

            missing = df[col].isnull().sum()
            missing_pct = (missing / len(df)) * 100

            print(f"{i:2d}. {col:15s} | Type: {attr_type:10s} | Dtype: {str(dtype):10s} | {unique_str:30s} | Missing: {missing:3d} ({missing_pct:5.1f}%)")

    print("\n" + "="*60)
    print("MISSING VALUES BY COLUMN")
    print("="*60)

    missing_summary = df.isnull().sum()
    if missing_summary.sum() > 0:
        missing_pct = (missing_summary / len(df)) * 100
        missing_df = pd.DataFrame({
            'Missing_Count': missing_summary,
            'Missing_Percent': missing_pct
        })
        missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)
        print(missing_df)
    else:
        print("No missing values found")

    print("\n" + "="*60)
    print("FIRST 5 ROWS")
    print("="*60)
    print(df.head().to_string())

    print("\n" + "="*60)
    print("DATA TYPES SUMMARY")
    print("="*60)
    print(df.dtypes.value_counts())

def save_extracted_data(df, attributes, attribute_types):
    """Save extracted data and metadata"""
    # Create output directory
    output_dir = Path('extracted_data')
    output_dir.mkdir(exist_ok=True)

    # Save to CSV
    csv_path = output_dir / 'chronic_kidney_disease.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nData saved to: {csv_path}")

    # Save metadata
    meta_path = output_dir / 'dataset_metadata.txt'
    with open(meta_path, 'w', encoding='utf-8') as f:
        f.write("CHRONIC KIDNEY DISEASE DATASET METADATA\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total samples: {len(df)}\n")
        f.write(f"Total attributes: {len(attributes)}\n\n")

        f.write("ATTRIBUTES:\n")
        f.write("-" * 30 + "\n")
        for i, attr in enumerate(attributes, 1):
            attr_type = attribute_types.get(attr, 'unknown')
            dtype = df[attr].dtype if attr in df.columns else 'unknown'
            missing = df[attr].isnull().sum() if attr in df.columns else 0
            f.write(f"{i:2d}. {attr:15s} | Type: {attr_type:10s} | Dtype: {str(dtype):10s} | Missing: {missing:3d}\n")

        if attributes[-1] in df.columns:
            f.write(f"\nTARGET VARIABLE: {attributes[-1]}\n")
            target_counts = df[attributes[-1]].value_counts()
            for cls, count in target_counts.items():
                f.write(f"  {cls}: {count} samples\n")

    print(f"Metadata saved to: {meta_path}")

    # Save feature statistics
    stats_path = output_dir / 'feature_statistics.csv'
    stats_data = []

    for col in attributes:
        if col in df.columns:
            dtype = df[col].dtype
            attr_type = attribute_types.get(col, 'unknown')
            missing = df[col].isnull().sum()
            missing_pct = (missing / len(df)) * 100

            if pd.api.types.is_numeric_dtype(df[col]):
                stats = {
                    'Feature': col,
                    'Type': attr_type,
                    'Dtype': dtype,
                    'Missing_Count': missing,
                    'Missing_Percent': missing_pct,
                    'Mean': df[col].mean(),
                    'Std': df[col].std(),
                    'Min': df[col].min(),
                    'Max': df[col].max(),
                    'Median': df[col].median()
                }
            else:
                stats = {
                    'Feature': col,
                    'Type': attr_type,
                    'Dtype': dtype,
                    'Missing_Count': missing,
                    'Missing_Percent': missing_pct,
                    'Unique_Values': df[col].nunique(),
                    'Most_Frequent': df[col].mode().iloc[0] if not df[col].mode().empty else 'N/A'
                }

            stats_data.append(stats)

    stats_df = pd.DataFrame(stats_data)
    stats_df.to_csv(stats_path, index=False)
    print(f"✓ Statistics saved to: {stats_path}")

def main():
    """Main function"""
    print("Starting ARFF file extraction...")
    print("="*60)

    # File paths
    arff_file = 'chronic_kidney_disease.arff'

    # Check if ARFF file exists
    if not Path(arff_file).exists():
        print(f"❌ Error: ARFF file '{arff_file}' not found!")
        print("Available ARFF files:")
        for f in Path('.').glob('*.arff'):
            print(f"  - {f}")
        return

    try:
        # Parse ARFF file
        df, attributes, attribute_types = parse_arff_manually(arff_file)

        if df is not None and not df.empty:
            print(f"\nSuccessfully extracted {len(df)} rows and {len(attributes)} columns")

            # Display information
            display_dataset_info(df, attributes, attribute_types)

            # Save data
            save_extracted_data(df, attributes, attribute_types)

            print("\n" + "="*60)
            print("EXTRACTION COMPLETE!")
            print("="*60)
            print("Files created in ./extracted_data/:")
            print("  - chronic_kidney_disease.csv (main dataset)")
            print("  - dataset_metadata.txt (detailed metadata)")
            print("  - feature_statistics.csv (statistical summary)")

            print(f"\n📊 Ready for machine learning with {len(df)} samples and {len(attributes)-1} features!")

        else:
            print("Failed to extract data from ARFF file")

    except Exception as e:
        print(f"Error during extraction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()