import pandas as pd
import re
import os

def process_excel(input_file):
    print(f"Processing {input_file}...")
    
    # Load the data
    try:
        df = pd.read_excel(input_file)
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # --- Step 1: Filter rows ---
    # Delete row if "是否正数发票" is "否" AND "备注" contains "冲红"
    # Note: Handle NaN values gracefully
    def should_keep(row):
        is_positive = str(row.get('是否正数发票', '')).strip()
        remark = str(row.get('备注', ''))
        
        if is_positive == '否' and '冲红' in remark:
            return False
        return True

    df = df[df.apply(should_keep, axis=1)].copy()
    print(f"Rows after filtering: {len(df)}")

    # --- Step 2: Copy Invoice Number ---
    # If "发票号码" has a value, copy to "数电发票号码"
    # We assume 'has a value' means not NaN and not empty string
    mask = df['发票号码'].notna() & (df['发票号码'].astype(str).str.strip() != '')
    df.loc[mask, '数电发票号码'] = df.loc[mask, '发票号码']

    # --- Step 3: Date Formatting ---
    # Format "开票日期" to "xx年x月x日"
    def format_date(val):
        if pd.isna(val):
            return val
        try:
            # If it's already a datetime object
            if isinstance(val, pd.Timestamp) or hasattr(val, 'strftime'):
                return val.strftime('%Y年%m月%d日')
            
            # If it's a string, try to parse it
            dt = pd.to_datetime(val)
            return dt.strftime('%Y年%m月%d日')
        except:
            return val # Return original if parsing fails

    if '开票日期' in df.columns:
        df['开票日期'] = df['开票日期'].apply(format_date)

    # --- Step 4: Update Invoice Type (发票票种) ---
    def update_invoice_type(row):
        invoice_type = str(row.get('发票票种', '')).strip()
        tax_rate = row.get('税率')
        
        # Helper to check if tax rate > 0
        is_positive_rate = False
        if pd.isna(tax_rate):
            is_positive_rate = False
        else:
            try:
                # If string with %
                if isinstance(tax_rate, str) and '%' in tax_rate:
                    val = float(tax_rate.replace('%', ''))
                    is_positive_rate = val > 0
                # If number
                elif isinstance(tax_rate, (int, float)):
                    is_positive_rate = tax_rate > 0
                else:
                    # Try converting string to float
                    is_positive_rate = float(tax_rate) > 0
            except:
                is_positive_rate = False

        if invoice_type in ['道路通行费电子普通发票', '数电发票（通行费发票）']:
            if is_positive_rate:
                return '专票'
            else:
                return '普票'
        
        if invoice_type in ['数电发票（航空运输电子客票行程单）', '数电发票（增值税专用发票）', '专票']:
            return '专票'
            
        if invoice_type == '数电发票（铁路电子客票）':
            return '专票（铁路电子客票）'
            
        # Default for others
        return '普票'

    df['发票票种'] = df.apply(update_invoice_type, axis=1)

    # --- Step 5: Generate step_1 file ---
    step1_filename = os.path.splitext(input_file)[0] + '_step_1.xlsx'
    
    # Columns for "进项发票"
    cols_input = ['开票日期', '销方名称', '货物或应税劳务名称', '规格型号', '单位', '数量', '单价', '金额', '税率', '税额', '价税合计', '数电发票号码', '发票代码']
    # Ensure columns exist
    cols_input = [c for c in cols_input if c in df.columns]
    
    # Columns for "报销发票"
    # Note: Added '数电发票号码' because Step 6 requires it for aggregation
    cols_reimburse = ['开票日期', '发票票种', '销方名称', '金额', '税率', '税额', '价税合计', '数电发票号码', '发票代码']
    cols_reimburse = [c for c in cols_reimburse if c in df.columns]

    with pd.ExcelWriter(step1_filename, engine='openpyxl') as writer:
        df[cols_input].to_excel(writer, sheet_name='进项发票', index=False)
        df[cols_reimburse].to_excel(writer, sheet_name='报销发票', index=False)
    
    print(f"Generated {step1_filename}")

    # --- Step 6: Generate final file with aggregation ---
    final_filename = os.path.splitext(input_file)[0] + '_final.xlsx'
    
    # We need to process the dataframes again for aggregation
    df_input = df[cols_input].copy()
    df_reimburse = df[cols_reimburse].copy()

    # --- Aggregation Logic ---
    
    def aggregate_input_group(group):
        # If only one row, return it
        if len(group) == 1:
            return group.iloc[0]
        
        # (1) Merge text columns
        # “货物或应税劳务名称、规格型号、单位、数量、单价”
        merge_cols = ['货物或应税劳务名称', '规格型号', '单位', '数量', '单价', '价税合计']
        merged_text_parts = []
        for idx, row in group.iterrows():
            parts = [str(row[c]) for c in merge_cols if pd.notna(row[c])]
            merged_text_parts.append(" ".join(parts))
        
        # Put merged text into first row's "货物或应税劳务名称"
        first_row = group.iloc[0].copy()
        first_row['货物或应税劳务名称'] = "\n".join(merged_text_parts) # Use newline to separate rows
        
        # Clear other merged columns in the result row if desired? 
        # The prompt says: "将所有行的...合并成一个大的文本，将文本填入到该数电发票号码第一行的“货物或应税劳务名称”中"
        # It doesn't explicitly say to clear the others, but usually in aggregation we keep one row.
        # We will keep the values of the first row for spec, unit, etc., or maybe clear them.
        # Given the instruction is just to fill the Name column, I will leave others as per first row.
        
        # (2) Sum numeric columns
        sum_cols = ['金额', '税额', '价税合计']
        for col in sum_cols:
            if col in group.columns:
                # Convert to numeric, coercing errors
                vals = pd.to_numeric(group[col], errors='coerce').fillna(0)
                first_row[col] = vals.sum()
        
        # (3) Enum "税率" -> join with "/"
        if '税率' in group.columns:
            rates = group['税率'].astype(str).unique()
            first_row['税率'] = "/".join(rates)
            
        return first_row

    def aggregate_reimburse_group(group):
        if len(group) == 1:
            return group.iloc[0]
        
        first_row = group.iloc[0].copy()
        
        # (1) Sum numeric columns
        sum_cols = ['金额', '税额', '价税合计']
        for col in sum_cols:
            if col in group.columns:
                vals = pd.to_numeric(group[col], errors='coerce').fillna(0)
                first_row[col] = vals.sum()
                
        # (2) Enum "税率"
        if '税率' in group.columns:
            rates = group['税率'].astype(str).unique()
            first_row['税率'] = "/".join(rates)
            
        return first_row

    # Apply aggregation
    # Note: We group by '数电发票号码'. Rows with NaN/Empty invoice number are treated as unique or grouped?
    # Usually empty invoice numbers shouldn't be grouped together.
    # We will assume only non-empty invoice numbers are aggregated.
    
    def apply_aggregation(df_target, agg_func):
        # Split into rows with and without invoice number
        mask_has_no = df_target['数电发票号码'].notna() & (df_target['数电发票号码'].astype(str).str.strip() != '')
        
        df_has_no = df_target[mask_has_no].copy()
        df_no_no = df_target[~mask_has_no].copy()
        
        if not df_has_no.empty:
            # Group by invoice number and apply aggregation
            # We use groupby().apply() but need to handle the index
            df_grouped = df_has_no.groupby('数电发票号码', group_keys=False).apply(agg_func)
            # groupby().apply() with a function returning a Series (row) results in a DataFrame with unique index
            
            # Combine back
            return pd.concat([df_grouped, df_no_no], ignore_index=True)
        else:
            return df_target

    df_input_final = apply_aggregation(df_input, aggregate_input_group)
    df_reimburse_final = apply_aggregation(df_reimburse, aggregate_reimburse_group)
    
    # Remove '数电发票号码' from Reimburse sheet if it wasn't requested in final output?
    # The prompt for step 5 didn't include it, but step 6 required it for processing.
    # Prompt for Step 5: "报销发票 复制...这几列的内容" (list without Number)
    # Prompt for Step 6: "进入报销发票 sheet，如果数电发票号码 是重复的..."
    # So the final file *might* not need it visible, OR it's needed. 
    # Usually if we aggregate by it, we keep it. But if the user strictly wants the columns from Step 5 list...
    # I will keep it because otherwise the aggregation logic is invisible/unverifiable. 
    # Also, "复制第3步的excel文件" (Step 5 file) implies the columns from Step 5 are present.
    # And I added it to Step 5. So I will keep it.

    with pd.ExcelWriter(final_filename, engine='openpyxl') as writer:
        df_input_final.to_excel(writer, sheet_name='进项发票', index=False)
        df_reimburse_final.to_excel(writer, sheet_name='报销发票', index=False)
        
    print(f"Generated {final_filename}")

if __name__ == "__main__":
    process_excel('original.xlsx')
