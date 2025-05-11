import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
import re
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.font_manager import FontProperties


class SecurityExcelProcessor:

    def __init__(self, output_dir="security_results"):
        """Initialize the processor"""
        self.output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.dimensions = {
            "attack": "Attack",
            "ethics": "Ethics",
            "fair": "Fairness",
            "legal": "Legality",
            "moral": "Morality",
            "privacy": "Privacy"
        }

        self.dim_english = {
            "attack": "Attack",
            "ethics": "Ethics",
            "fair": "Fairness",
            "legal": "Legality",
            "moral": "Morality",
            "privacy": "Privacy"
        }

        self.reverse_dimensions = {v: k for k, v in self.dimensions.items()}

        self.attack_methods = {
            "场景构建": "Scenario Construction",
            "指代消解": "Reference Resolution",
            "目的反转": "Purpose Inversion",
            "谬误攻击失败": "Fallacy Attack",
            "试探性提问": "Probing Questions",
            "角色扮演": "Role Playing",
            "话题转换": "Topic Inversion"
        }

        self.method_english = {
            "场景构建": "Scenario Construction",
            "指代消解": "Reference Resolution",
            "目的反转": "Purpose Inversion",
            "谬误攻击失败": "Fallacy Attack",
            "试探性提问": "Probing Questions",
            "角色扮演": "Role Playing",
            "话题转换": "Topic Inversion"
        }

        self.column_patterns = {
            'ID': [r'id', r'number', r'sequence'],
            'Task': [r'task', r'dimension'],
            'Method': [r'method', r'attack'],
            'Round': [r'round', r'turns'],
            'Identification': [r'identification', r'identify', r'recognition', r'identification ability', 
                              r'recognition ability', r'hazard identification', r'danger identification',
                              r'recognize', r'discover'],
            'Handling': [r'handling', r'process', r'disposal', r'handling ability', r'disposal ability', 
                         r'handling effect', r'how to respond', r'solve'],
            'Consistency': [r'consistency', r'coherence', r'stability'],
            'Average': [r'average', r'mean', r'average score', r'average value']
        }
        self.chinese_font = setup_chinese_font()

        self.font = {
            'family': self.chinese_font if self.chinese_font else 'sans-serif',
            'weight': 'normal',
            'size': 30
        }

        self.all_data = []

    def _find_column_matches(self, headers):
        """
        Match column names with standardized names

        Args:
            headers: List of header columns

        Returns:
            dict: Column mapping dictionary
        """
        column_map = {}

        for i, header in enumerate(headers):
            header_str = str(header).lower().strip()

            for std_col, patterns in self.column_patterns.items():
                if any(re.search(pattern, header_str) for pattern in patterns):
                    column_map[i] = std_col
                    break

        return column_map

    def process_excel_file(self, excel_file):
        """
        Process Excel file, extract detailed round data from all sheets

        Args:
            excel_file: Excel file path

        Returns:
            DataFrame: All extracted data merged into a DataFrame
        """
        print(f"Processing Excel file: {excel_file}")

        try:
            xl = pd.ExcelFile(excel_file)
            for sheet_name in xl.sheet_names:
                print(f"Processing sheet: {sheet_name}")
                dimension = None
                for dim in self.dimensions:
                    if dim in sheet_name.lower():
                        dimension = dim
                        break
                if dimension is None:
                    for dim_cn, dim_en in self.reverse_dimensions.items():
                        if dim_cn in sheet_name:
                            dimension = dim_en
                            break
                if dimension is None:
                    model_dim_match = re.search(r'(\w+)[-_](\w+)$', sheet_name)
                    if model_dim_match:
                        possible_dim = model_dim_match.group(2).lower()
                        if possible_dim in self.dimensions:
                            dimension = possible_dim

                if dimension is None:
                    print(f"  Skipped: Unable to determine dimension - {sheet_name}")
                    continue
                df = pd.read_excel(excel_file, sheet_name=sheet_name)
                detail_row_idx = None
                for i, row in df.iterrows():
                    row_str = ' '.join([str(val) for val in row.values if pd.notna(val)])
                    if 'Detailed Round Data' in row_str:
                        detail_row_idx = i
                        break

                if detail_row_idx is None:
                    print(f"  Skipped: Detailed round data row not found - {sheet_name}")
                    continue
                table_data = self._extract_table_data(df, detail_row_idx, dimension, sheet_name)

                if table_data is not None and not table_data.empty:
                    self.all_data.append(table_data)
                    print(f"  Successfully extracted {len(table_data)} rows of data")
            if self.all_data:
                combined_data = pd.concat(self.all_data, ignore_index=True)
                print(f"Total extracted {len(combined_data)} rows of data")
                return combined_data
            else:
                print("No detailed round data found")
                return pd.DataFrame()

        except Exception as e:
            print(f"Error processing Excel file: {str(e)}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()

    def _extract_table_data(self, df, detail_row_idx, dimension, sheet_name):
        """
        Extract table data from DataFrame

        Args:
            df: DataFrame
            detail_row_idx: Index of the detailed round data row
            dimension: Dimension name
            sheet_name: Sheet name

        Returns:
            DataFrame: Extracted table data
        """
        try:
            header_row_idx = detail_row_idx + 1
            if header_row_idx >= len(df):
                print(f"  Header row out of range - {sheet_name}")
                return None
            header_row = df.iloc[header_row_idx]
            columns = []
            column_indices = {}

            for i, col_name in enumerate(header_row):
                if pd.isna(col_name) or str(col_name).strip() == '':
                    continue

                col_str = str(col_name).strip()
                columns.append(col_str)
                column_indices[col_str] = i
            std_columns = {
                'ID': 'ID',
                'Task': 'Task',
                'Method': 'Method',
                'Round': 'Round',
                'Identification': 'Identification',
                'Handling': 'Handling',
                'Consistency': 'Consistency',
                'Average': 'Average'
            }

            for std_col, patterns in self.column_patterns.items():
                if std_col not in column_indices:
                    for i, col_name in enumerate(header_row):
                        if pd.isna(col_name):
                            continue
                        col_str = str(col_name).lower().strip()
                        if any(re.search(pattern, col_str) for pattern in patterns):
                            column_indices[std_col] = i
                            break

            end_row_idx = len(df)
            for i in range(header_row_idx + 1, len(df)):
                row_vals = [str(val).strip() for val in df.iloc[i].values if pd.notna(val)]
                if not row_vals:  # Skip empty rows
                    continue
                if any(('Summary' in val or 'Subtotal' in val or 'Statistics' in val) for val in row_vals):
                    end_row_idx = i
                    break

            data_rows = df.iloc[header_row_idx + 1:end_row_idx].copy()

            data_rows = data_rows.dropna(how='all')

            if data_rows.empty:
                print(f"  No data rows found - {sheet_name}")
                return None

            extracted_data = []
            last_id = None
            last_task = None
            last_method = None

            for idx, row in data_rows.iterrows():
                row_dict = {}
                for std_col, col_idx in column_indices.items():
                    if col_idx < len(row):
                        row_dict[std_col] = row.iloc[col_idx]
                if 'ID' in row_dict and pd.notna(row_dict['ID']):
                    last_id = row_dict['ID']
                elif last_id is not None:
                    row_dict['ID'] = last_id

                if 'Task' in row_dict and pd.notna(row_dict['Task']):
                    last_task = row_dict['Task']
                elif last_task is not None:
                    row_dict['Task'] = last_task
                if 'Method' in row_dict and pd.notna(row_dict['Method']):
                    last_method = row_dict['Method']
                elif last_method is not None:
                    row_dict['Method'] = last_method

                if 'Round' in row_dict and pd.notna(row_dict['Round']):
                    # Add dimension information
                    row_dict['Dimension'] = dimension
                    row_dict['Dimension_CN'] = self.dimensions.get(dimension, dimension)
                    row_dict['Sheet'] = sheet_name

                    extracted_data.append(row_dict)

            if not extracted_data:
                print(f"  Unable to extract valid data from table - {sheet_name}")
                return None

            result_df = pd.DataFrame(extracted_data)
            required_cols = ['ID', 'Round', 'Method', 'Identification', 'Handling', 'Consistency']
            missing_cols = [col for col in required_cols if col not in result_df.columns]

            critical_missing = [col for col in ['Round', 'Method'] if col in missing_cols]

            if critical_missing:
                print(f"  Missing critical columns: {', '.join(critical_missing)} - {sheet_name}")
                return None

            for col in missing_cols:
                if col not in critical_missing:
                    print(f"  Note: Filling missing column '{col}' with default value - {sheet_name}")
                    result_df[col] = 5.0

            if 'ID' in result_df.columns:
                result_df['ID'] = pd.to_numeric(result_df['ID'], errors='coerce')

            numeric_cols = ['Round', 'Identification', 'Handling', 'Consistency', 'Average']
            for col in numeric_cols:
                if col in result_df.columns:
                    result_df[col] = pd.to_numeric(result_df[col], errors='coerce')
            if 'Task' not in result_df.columns:
                result_df['Task'] = self.dimensions.get(dimension, dimension)
            if 'Average' not in result_df.columns and all(
                    col in result_df.columns for col in ['Identification', 'Handling', 'Consistency']):
                result_df['Average'] = result_df[['Identification', 'Handling', 'Consistency']].mean(axis=1)

            return result_df

        except Exception as e:
            print(f"  Error extracting table data ({sheet_name}): {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def create_summary_statistics(self, data):
        """
        Create data summary statistics

        Args:
            data: DataFrame

        Returns:
            dict: Summary statistics
        """
        summary = {}
        required_cols = ['Dimension', 'Method', 'Round']
        if not all(col in data.columns for col in required_cols):
            print("Missing required columns, cannot create summary statistics")
            return summary
        for dimension, dim_data in data.groupby('Dimension'):
            dimension_summary = {}
            for method, method_data in dim_data.groupby('Method'):
                method_summary = {}
                for round_num, round_data in method_data.groupby('Round'):
                    score_cols = [col for col in ['Identification', 'Handling', 'Consistency']
                                  if col in round_data.columns]
                    round_summary = {
                        'count': len(round_data)
                    }
                    for col in score_cols:
                        round_summary[f'{col.lower()}_mean'] = round_data[col].mean()
                    if score_cols:
                        round_summary['overall_mean'] = round_data[score_cols].mean(axis=1).mean()
                    else:
                        round_summary['overall_mean'] = 0

                    method_summary[round_num] = round_summary

                dimension_summary[method] = method_summary

            summary[dimension] = dimension_summary

        return summary

    def create_visualizations(self, data, summary):
        """
        Create visualizations using specified style

        Args:
            data: DataFrame
            summary: Summary statistics
        """
        for dimension in summary:
            dim_dir = os.path.join(self.output_dir, dimension)
            Path(dim_dir).mkdir(parents=True, exist_ok=True)
            for method in summary[dimension]:
                rounds = []
                ident_scores = []
                handling_scores = []
                consist_scores = []

                for round_num in sorted(summary[dimension][method].keys()):
                    if 2 <= round_num <= 5:
                        round_data = summary[dimension][method][round_num]
                        rounds.append(round_num)
                        if 'identification_mean' in round_data:
                            ident_scores.append(round_data['identification_mean'])
                        else:
                            ident_scores.append(None)

                        if 'handling_mean' in round_data:
                            handling_scores.append(round_data['handling_mean'])
                        else:
                            handling_scores.append(None)

                        if 'consistency_mean' in round_data:
                            consist_scores.append(round_data['consistency_mean'])
                        else:
                            consist_scores.append(None)
                valid_data = [(r, i, h, c) for r, i, h, c in zip(rounds, ident_scores, handling_scores, consist_scores)
                              if i is not None and h is not None and c is not None]
                if len(valid_data) < 2:
                    print(f"  Skipping {method}: Insufficient data points")
                    continue
                rounds, ident_scores, handling_scores, consist_scores = zip(*valid_data)
                plt.figure(figsize=(10, 8))
                plt.plot(rounds, ident_scores, color='#2F5597', marker='.', markersize=10, linestyle='-',
                         label="Identification Ability")
                plt.plot(rounds, handling_scores, color='#C00000', marker='+', markersize=10, linestyle='-',
                         label="Handling Ability")
                plt.plot(rounds, consist_scores, color='#228B22', marker='^', markersize=10, linestyle='-',
                         label="Safety Consistency")

                plt.grid(True)
                dim_english = self.dim_english.get(dimension, dimension.capitalize())
                method_abbr = self.attack_methods.get(method, "")
                plt.title(f"{dim_english} ({method_abbr})", self.font)
                plt.xlabel("Round", self.font)
                plt.ylabel("Score", self.font)

                plt.xticks([2, 3, 4, 5])
                plt.xlim(1.5, 5.5)

                plt.tick_params(labelsize=25)

                all_values = ident_scores + handling_scores + consist_scores
                min_val = max(6, min(all_values) - 0.5)
                max_val = min(9, max(all_values) + 0.5)
                if max_val - min_val < 1.5:
                    center = (max_val + min_val) / 2
                    min_val = max(6, center - 0.75)
                    max_val = min(9, center + 0.75)

                y_range = max_val - min_val
                if y_range <= 2:
                    step = 0.5
                else:
                    step = 1.0

                y_ticks = np.arange(np.ceil(min_val * 2) / 2, max_val + 0.01, step)

                plt.ylim(min_val, max_val)
                plt.yticks(y_ticks)

                plt.legend(loc="upper right", prop={'size': 20})
                plt.tight_layout()

                method_name = method.replace(' ', '_')
                filename = f"{dimension}_{method_name}"
                plt.savefig(os.path.join(dim_dir, f'{filename}.pdf'))
                plt.savefig(os.path.join(dim_dir, f'{filename}.png'), dpi=300)
                plt.close()

                print(f"  Chart created: {filename}")

            self._create_dimension_summary_chart(dimension, summary[dimension], dim_dir)

        print(f"Saved to {self.output_dir}")

    def _create_dimension_summary_chart(self, dimension, dimension_data, output_dir):
        """
        Create comprehensive chart for dimension using specified style

        Args:
            dimension: Dimension name
            dimension_data: Dimension data
            output_dir: Output directory
        """
        plt.figure(figsize=(12, 9))

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
        markers = ['.', '+', '^', 's', 'o', 'x', 'D']

        all_methods_data = []
        for method, method_data in dimension_data.items():
            rounds = []
            overall_scores = []

            for round_num in sorted(method_data.keys()):
                if 2 <= round_num <= 5:
                    if 'overall_mean' in method_data[round_num]:
                        rounds.append(round_num)
                        overall_scores.append(method_data[round_num]['overall_mean'])

            # Only include if at least 2 data points
            if len(rounds) >= 2:
                all_methods_data.append({
                    'method': method,
                    'rounds': rounds,
                    'scores': overall_scores
                })

        for i, method_data in enumerate(all_methods_data):
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]
            method = method_data['method']
            method_abbr = self.attack_methods.get(method, "")
            label = f"{method_abbr}" if method_abbr else method

            plt.plot(method_data['rounds'], method_data['scores'],
                     marker=marker, markersize=10,
                     color=color, linewidth=2,
                     label=label)

        plt.grid(True)
        dim_english = self.dim_english.get(dimension, dimension.capitalize())
        plt.title(f"{dim_english}", self.font)
        plt.xlabel("# Turn", self.font)
        plt.ylabel("Score", self.font)
        plt.xticks([2, 3, 4, 5])
        plt.xlim(1.5, 5.5)

        plt.tick_params(labelsize=25)
        plt.ylim(2, 9)
        plt.legend(loc="upper right", prop={'size': 20})
        plt.tight_layout()

        plt.savefig(os.path.join(output_dir, f'{dimension}_summary.pdf'))
        plt.savefig(os.path.join(output_dir, f'{dimension}_summary.png'), dpi=300)
        plt.close()

    def save_results_to_excel(self, data, summary):
        """
        Save results to Excel file

        Args:
            data: DataFrame
            summary: Summary statistics
        """
        data.to_csv(os.path.join(self.output_dir, 'extracted_data.csv'), index=False)
        with pd.ExcelWriter(os.path.join(self.output_dir, 'analysis_results.xlsx')) as writer:
            data.to_excel(writer, sheet_name='Raw_Data', index=False)
            for dimension in summary:
                summary_rows = []

                for method in summary[dimension]:
                    for round_num, round_data in summary[dimension][method].items():
                        row = {
                            'Dimension': dimension,
                            'Dimension_CN': self.dimensions.get(dimension, dimension),
                            'Method': method,
                            'Method_Abbr': self.attack_methods.get(method, ""),
                            'Round': round_num,
                            'Count': round_data['count']
                        }
                        metrics = ['identification_mean', 'handling_mean', 'consistency_mean', 'overall_mean']
                        for metric in metrics:
                            if metric in round_data:
                                col_name = metric.replace('_mean', '').capitalize()
                                row[col_name] = round_data[metric]

                        summary_rows.append(row)

                if summary_rows:
                    summary_df = pd.DataFrame(summary_rows)
                    dimension_name = dimension.capitalize()
                    summary_df.to_excel(writer, sheet_name=f'{dimension_name[:30]}', index=False)

            all_summary_rows = []

            for dimension in summary:
                for method in summary[dimension]:
                    for round_num, round_data in summary[dimension][method].items():
                        row = {
                            'Dimension': dimension,
                            'Dimension_CN': self.dimensions.get(dimension, dimension),
                            'Method': method,
                            'Method_Abbr': self.attack_methods.get(method, ""),
                            'Round': round_num,
                            'Count': round_data['count']
                        }

                        metrics = ['identification_mean', 'handling_mean', 'consistency_mean', 'overall_mean']
                        for metric in metrics:
                            if metric in round_data:
                                col_name = metric.replace('_mean', '').capitalize()
                                row[col_name] = round_data[metric]

                        all_summary_rows.append(row)

            if all_summary_rows:
                all_summary_df = pd.DataFrame(all_summary_rows)
                all_summary_df.to_excel(writer, sheet_name='Summary', index=False)

        print(f"Analysis results saved to {self.output_dir}/analysis_results.xlsx")

    def process_and_analyze(self, excel_file):
        """
        Process Excel file and analyze data

        Args:
            excel_file: Excel file path

        Returns:
            bool: Whether processing was successful
        """
        data = self.process_excel_file(excel_file)

        if data.empty:
            print("No valid data extracted, processing terminated")
            return False
        summary = self.create_summary_statistics(data)
        self.create_visualizations(data, summary)
        self.save_results_to_excel(data, summary)

        print("Data processing and analysis complete!")
        return True

if __name__ == "__main__":
    processor = SecurityExcelProcessor()
    processor.process_and_analyze("data_perturn.xlsx")