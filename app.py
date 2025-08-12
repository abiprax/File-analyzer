import pandas as pd
import numpy as np
import json
import os
from typing import Dict, List, Any
import openai
from langdetect import detect
import warnings
import streamlit as st
warnings.filterwarnings('ignore')

class SimpleFileAnalyzer:
    def __init__(self, openai_api_key: str):
        """
        Simple File Analyzer that works with any dataset
        """
        self.client = openai.OpenAI(api_key=openai_api_key)
        self.datasets = {}  # Store all loaded datasets
        self.analysis_results = {}  # Store analysis results

    def load_files(self, uploaded_files):
        """
        Load uploaded CSV/Excel files from Streamlit
        """
        if not uploaded_files:
            return "No files uploaded."

        results = []

        for uploaded_file in uploaded_files:
            filename = uploaded_file.name

            try:
                if filename.endswith('.csv'):
                    # Load CSV file
                    df = pd.read_csv(uploaded_file)
                    self.datasets[filename] = df
                    results.append(f"Loaded CSV: {filename} ({len(df)} rows, {len(df.columns)} columns)")

                elif filename.endswith(('.xlsx', '.xls')):
                    # Load Excel file with all sheets
                    excel_file = pd.ExcelFile(uploaded_file)
                    for sheet_name in excel_file.sheet_names:
                        df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
                        dataset_name = f"{filename}_{sheet_name}"
                        self.datasets[dataset_name] = df
                        results.append(f"Loaded Excel sheet: {sheet_name} ({len(df)} rows, {len(df.columns)} columns)")

            except Exception as e:
                results.append(f"Error loading {filename}: {str(e)}")

        if self.datasets:
            analysis_result = self.analyze_all_datasets()
            results.append(f"\nTotal datasets loaded: {len(self.datasets)}")
            results.append(analysis_result)

        return "\n".join(results)

    def analyze_column(self, column_name: str, sample_data: List, data_type: str) -> Dict:
        """
        Use OpenAI to analyze a single column and create English description
        """
        # Detect language of column name
        try:
            detected_lang = detect(column_name.replace('_', ' ').replace('-', ' '))
        except:
            detected_lang = 'unknown'

        prompt = f"""
        Analyze this data column and provide a JSON response with English translations:

        Column name: "{column_name}"
        Language detected: {detected_lang}
        Data type: {data_type}
        Sample values: {sample_data[:5]}

        Please provide:
        {{
            "english_name": "Clear English column name (translate if not in English)",
            "description": "Simple English description of what this column contains (max 15 words)",
            "data_category": "numeric/categorical/text/datetime/boolean"
        }}

        Rules:
        - ALWAYS translate column name to English if it's in another language
        - ALWAYS write description in clear, simple English
        - Use simple, everyday language
        - Be specific about what the data represents
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-4.1",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=200
            )

            result = json.loads(response.choices[0].message.content)
            return result

        except Exception as e:
            # Simple fallback with basic translation attempt
            english_name = column_name.replace('_', ' ').replace('-', ' ').title()
            
            return {
                "english_name": english_name,
                "description": f"Data column containing {data_type} values",
                "data_category": "numeric" if data_type in ['int64', 'float64'] else "text"
            }

    def analyze_dataset(self, dataset_name: str) -> Dict:
        """
        Analyze a single dataset completely
        """
        if dataset_name not in self.datasets:
            return {"error": "Dataset not found"}

        df = self.datasets[dataset_name]

        # Basic info
        basic_info = {
            "name": dataset_name,
            "rows": len(df),
            "columns": len(df.columns),
            "size_mb": round(df.memory_usage(deep=True).sum() / (1024*1024), 2)
        }

        # Analyze each column
        column_analysis = {}
        
        for col in df.columns:
            # Get sample data (non-null values) for AI analysis only
            non_null_data = df[col].dropna()
            
            if len(non_null_data) == 0:
                sample_data = []
            elif len(non_null_data) <= 10:
                sample_data = non_null_data.tolist()
            else:
                sample_data = non_null_data.sample(10, random_state=42).tolist()
            
            data_type = str(df[col].dtype)

            # Convert sample data to strings for AI analysis
            sample_str = [str(x)[:50] for x in sample_data]

            # Analyze with OpenAI (get English name and description)
            analysis = self.analyze_column(col, sample_str, data_type)

            # Add statistics (no sample values in final output)
            analysis.update({
                "original_name": col,
                "data_type": data_type,
                "missing_count": int(df[col].isnull().sum()),
                "missing_percent": round(df[col].isnull().sum() / len(df) * 100, 1),
                "unique_values": int(df[col].nunique())
            })

            # Add numeric stats if applicable
            if df[col].dtype in ['int64', 'float64']:
                try:
                    analysis.update({
                        "min": float(df[col].min()),
                        "max": float(df[col].max()),
                        "mean": round(float(df[col].mean()), 2),
                        "median": float(df[col].median())
                    })
                except:
                    pass

            column_analysis[col] = analysis

        # Generate summary
        summary = self.generate_summary(dataset_name, basic_info, column_analysis)

        return {
            "basic_info": basic_info,
            "columns": column_analysis,
            "summary": summary
        }

    def generate_summary(self, dataset_name: str, basic_info: Dict, columns: Dict) -> str:
        """
        Generate a simple English summary of the dataset
        """
        # Create column descriptions using English names and descriptions
        col_descriptions = []
        for col, info in columns.items():
            col_descriptions.append(f"- {info['english_name']}: {info['description']}")

        prompt = f"""
        Write a simple 2-3 sentence summary for this dataset in clear English:

        Dataset: {dataset_name}
        Rows: {basic_info['rows']}
        Columns: {basic_info['columns']}

        Column information:
        {chr(10).join(col_descriptions)}

        Write in simple English explaining what this data is about and what someone could do with it.
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-4.1",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=150
            )
            return response.choices[0].message.content.strip()
        except:
            return f"This dataset contains {basic_info['rows']} rows and {basic_info['columns']} columns of data for analysis."

    def analyze_all_datasets(self):
        """
        Analyze all loaded datasets and return summary
        """
        if not self.datasets:
            return "No datasets to analyze."

        results = []
        results.append("=== DATASET ANALYSIS ===")

        # Analyze each dataset
        for dataset_name in self.datasets.keys():
            analysis = self.analyze_dataset(dataset_name)
            self.analysis_results[dataset_name] = analysis

            if 'error' not in analysis:
                info = analysis['basic_info']
                results.append(f"Dataset: {dataset_name}")
                results.append(f"Size: {info['rows']} rows x {info['columns']} columns")
                results.append(f"Summary: {analysis['summary']}\n")

        # Find relationships if multiple datasets
        if len(self.datasets) > 1:
            relationships = self.find_relationships()
            results.append("=== RELATIONSHIPS ===")

            for rel_name, rel_info in relationships.items():
                if isinstance(rel_info, dict) and 'common_columns' in rel_info:
                    results.append(f"{rel_name}")
                    results.append(f"Common columns: {', '.join(rel_info['common_columns'])}")
                    for join_info in rel_info['potential_joins']:
                        results.append(f"- {join_info['column']}: {join_info['relationship']}")
                    results.append("")

        # Show detailed column analysis (without sample values)
        results.append("=== COLUMN DETAILS ===")
        for dataset_name, analysis in self.analysis_results.items():
            if 'error' in analysis:
                continue

            results.append(f"\n{dataset_name}:")
            results.append("-" * 40)

            for orig_col, col_info in analysis['columns'].items():
                results.append(f"{col_info['english_name']} ({orig_col})")
                results.append(f"  Description: {col_info['description']}")
                results.append(f"  Type: {col_info['data_category']} | Missing: {col_info['missing_percent']}% | Unique: {col_info['unique_values']}")

                if 'mean' in col_info:
                    results.append(f"  Stats: Min={col_info['min']}, Max={col_info['max']}, Mean={col_info['mean']}")

                results.append("")

        return "\n".join(results)

    def find_relationships(self) -> Dict:
        """
        Find relationships between datasets
        """
        if len(self.datasets) < 2:
            return {"message": "Need at least 2 datasets to find relationships"}

        relationships = {}
        dataset_names = list(self.datasets.keys())

        for i in range(len(dataset_names)):
            for j in range(i + 1, len(dataset_names)):
                name1, name2 = dataset_names[i], dataset_names[j]
                df1, df2 = self.datasets[name1], self.datasets[name2]

                # Find common columns
                common_cols = set(df1.columns).intersection(set(df2.columns))

                if common_cols:
                    rel_info = {
                        "common_columns": list(common_cols),
                        "potential_joins": []
                    }

                    # Analyze each common column
                    for col in common_cols:
                        unique1 = df1[col].nunique()
                        unique2 = df2[col].nunique()
                        total1 = len(df1)
                        total2 = len(df2)

                        # Determine relationship type
                        if unique1 == total1 and unique2 == total2:
                            rel_type = "one_to_one"
                        elif unique1 == total1:
                            rel_type = "one_to_many"
                        elif unique2 == total2:
                            rel_type = "many_to_one"
                        else:
                            rel_type = "many_to_many"

                        rel_info["potential_joins"].append({
                            "column": col,
                            "relationship": rel_type,
                            "unique_in_dataset1": unique1,
                            "unique_in_dataset2": unique2
                        })

                    relationships[f"{name1} <-> {name2}"] = rel_info

        return relationships

    def find_relevant_dataset(self, question: str):
        """
        Find the most relevant dataset for the question
        """
        question_lower = question.lower()

        # Look for dataset-specific keywords by checking both original and English column names
        for dataset_name, df in self.datasets.items():
            # Check original column names
            for col in df.columns:
                if col.lower() in question_lower or col.lower().replace(' ', '') in question_lower:
                    return (dataset_name, df)
            
            # Check English column names if analysis is available
            if dataset_name in self.analysis_results and 'columns' in self.analysis_results[dataset_name]:
                for col, analysis in self.analysis_results[dataset_name]['columns'].items():
                    english_name = analysis.get('english_name', '').lower()
                    if english_name and (english_name in question_lower or english_name.replace(' ', '') in question_lower):
                        return (dataset_name, df)

        # If no specific match, return the largest dataset
        return max(self.datasets.items(), key=lambda x: len(x[1]) * len(x[1].columns))

    def generate_query_code(self, question: str, df: pd.DataFrame, dataset_name: str) -> str:
        """
        Generate Python code to answer the question with better column matching
        """
        # Get column info including both original and English names
        columns = list(df.columns)
        dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
        numeric_cols = [col for col, dtype in dtypes.items() if 'int' in dtype or 'float' in dtype]
        text_cols = [col for col, dtype in dtypes.items() if 'object' in dtype]
        date_cols = [col for col, dtype in dtypes.items() if 'datetime' in dtype]

        # Create mapping of English names to original column names
        english_to_original = {}
        column_info = []
        if dataset_name in self.analysis_results and 'columns' in self.analysis_results[dataset_name]:
            for orig_col, analysis in self.analysis_results[dataset_name]['columns'].items():
                english_name = analysis.get('english_name', orig_col)
                english_to_original[english_name.lower()] = orig_col
                column_info.append(f"{english_name} -> '{orig_col}' ({analysis.get('description', 'No description')})")

        prompt = f"""
        Generate Python code to answer this question: "{question}"

        Dataset: {dataset_name}
        Available columns with mappings:
        {chr(10).join(column_info)}

        Original column names: {columns}
        Data types: {dtypes}
        Numeric columns: {numeric_cols}
        Text/categorical columns: {text_cols}
        Date columns: {date_cols}

        English to original column mapping: {english_to_original}

        IMPORTANT RULES:
        1. Use 'df' as the dataframe variable (already loaded)
        2. ALWAYS use the ORIGINAL column names (in quotes) when accessing columns
        3. Store final answer in variable 'result'
        4. Handle missing data appropriately
        5. For text filtering, be case-insensitive when possible
        6. If filtering by specific values, consider partial matches for text columns
        7. Use .dropna() when necessary to handle missing values

        Examples:
        - For "manager of central region": df[df['Region'].str.lower() == 'central']['Manager'].dropna().values
        - For totals: result = df['column_name'].sum()
        - For counts: result = len(df[df['column'] == 'value'])
        - For filtering: filtered_df = df[df['column'].str.contains('value', case=False, na=False)]

        Generate ONLY the Python code that will work:
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-4.1",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=300
            )

            code = response.choices[0].message.content.strip()

            # Clean up the code
            if code.startswith('```python'):
                code = code.replace('```python', '').replace('```', '').strip()
            elif code.startswith('```'):
                code = code.replace('```', '').strip()

            # Remove any import statements
            lines = code.split('\n')
            clean_lines = []
            for line in lines:
                if not line.strip().startswith('import') and not line.strip().startswith('from'):
                    clean_lines.append(line)

            return '\n'.join(clean_lines)

        except Exception as e:
            # More intelligent fallback based on question analysis
            question_lower = question.lower()
            
            # Try to identify the intent and relevant columns
            if any(word in question_lower for word in ['manager', 'who is', 'name']):
                # Looking for a person/manager
                manager_cols = [col for col in columns if 'manager' in col.lower()]
                region_cols = [col for col in columns if any(word in col.lower() for word in ['region', 'area', 'location'])]
                
                if manager_cols and region_cols:
                    manager_col = manager_cols[0]
                    region_col = region_cols[0]
                    
                    return f"""
try:
    # Look for central region manager
    central_matches = df[df['{region_col}'].str.lower().str.contains('central', na=False)]
    if len(central_matches) > 0:
        result = central_matches['{manager_col}'].dropna().iloc[0] if len(central_matches['{manager_col}'].dropna()) > 0 else "No manager found"
    else:
        result = "Central region not found"
except Exception as e:
    result = f"Error: {{str(e)}}"
"""
            
            # Generic fallback
            return f"""
try:
    # Basic analysis based on question
    question_lower = "{question.lower()}"
    
    if 'total' in question_lower or 'sum' in question_lower:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            result = df[numeric_cols[0]].sum()
        else:
            result = f"No numeric columns found. Available columns: {{', '.join(df.columns.tolist())}}"
    
    elif 'count' in question_lower or 'how many' in question_lower:
        result = len(df)
    
    else:
        result = f"Dataset has {{len(df)}} rows and {{len(df.columns)}} columns. Available columns: {{', '.join(df.columns.tolist())}}"

except Exception as e:
    result = f"Error in analysis: {{str(e)}}"
"""

    def generate_insights_and_suggestions(self, question: str, answer: str, df: pd.DataFrame) -> str:
        """
        Generate insights and suggestions based on the answer
        """
        # Get basic dataset info
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

        prompt = f"""
        Based on this data analysis, provide insights and suggestions in English:

        Question: {question}
        Answer: {answer}

        Dataset info:
        - Total rows: {len(df)}
        - Numeric columns: {numeric_cols[:5]}
        - Categorical columns: {categorical_cols[:5]}

        Please provide:
        1. Key insights (2-3 bullet points about what the answer reveals)
        2. Suggestions for further analysis (2-3 specific questions to explore)

        Keep it practical and business-focused. Format as:

        INSIGHTS:
        • [insight 1]
        • [insight 2]

        SUGGESTIONS:
        • [suggestion 1]
        • [suggestion 2]
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-4.1",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=300
            )
            return response.choices[0].message.content.strip()
        except:
            return """
INSIGHTS:
• The analysis provides a specific data point from your dataset
• This information can help with decision making and understanding patterns

SUGGESTIONS:
• Try comparing this result with other categories or time periods
• Look for correlations with other variables in the dataset
"""

    def query_data(self, question: str) -> str:
        """
        Query system that works with any dataset and provides insights
        """
        if not self.datasets:
            return "No datasets loaded. Please upload files first."

        # Find the most relevant dataset for the question
        relevant_dataset = self.find_relevant_dataset(question)
        dataset_name, df = relevant_dataset

        result_parts = []
        result_parts.append(f"Using dataset: {dataset_name}")
        result_parts.append(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")

        # Generate and execute Python code
        code = self.generate_query_code(question, df, dataset_name)

        try:
            # Execute the code safely
            local_vars = {"df": df, "pd": pd, "np": np}
            safe_builtins = {
                "len": len, "str": str, "int": int, "float": float, "max": max, "min": min,
                "sum": sum, "round": round, "abs": abs, "print": print
            }

            exec(code, {"__builtins__": safe_builtins}, local_vars)

            # Get the result
            if 'result' in local_vars:
                answer = str(local_vars['result'])
                result_parts.append(f"\nANSWER: {answer}")

                # Generate insights and suggestions
                insights = self.generate_insights_and_suggestions(question, answer, df)
                result_parts.append(f"\n{insights}")

                return "\n".join(result_parts)
            else:
                return "Code executed but no result found."

        except Exception as e:
            result_parts.append(f"\nError: {str(e)}")
            result_parts.append(f"\nGenerated code: {code}")
            return "\n".join(result_parts)

# Streamlit App
def main():
    st.set_page_config(
        page_title="Simple File Analyzer", 
        page_icon=None,
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("Simple File Analyzer")
    st.markdown("Upload CSV or Excel files and ask questions about your data using AI! All column names and descriptions will be in English.")

    # Get API key from Streamlit secrets
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except:
        st.error("OpenAI API key not found in Streamlit secrets. Please add OPENAI_API_KEY to your secrets.")
        return

    # Initialize session state
    if 'analyzer' not in st.session_state:
        try:
            st.session_state.analyzer = SimpleFileAnalyzer(api_key)
        except Exception as e:
            st.error(f"Error initializing analyzer: {str(e)}")
            return
    
    if 'analysis_done' not in st.session_state:
        st.session_state.analysis_done = False

    # Sidebar for File Upload
    with st.sidebar:
        st.header("Configuration")
        st.success("API Key loaded from secrets")
        
        st.header("File Upload")
        
        # File Upload
        uploaded_files = st.file_uploader(
            "Upload CSV or Excel files",
            type=['csv', 'xlsx', 'xls'],
            accept_multiple_files=True
        )

        if uploaded_files and st.session_state.analyzer:
            if st.button("Process Files", type="primary"):
                with st.spinner("Processing files..."):
                    result = st.session_state.analyzer.load_files(uploaded_files)
                    st.session_state.analysis_done = True
                    st.session_state.analysis_result = result

    # Main Content Area
    if not uploaded_files:
        st.info("Please upload CSV or Excel files in the sidebar.")
        
    else:
        # Two-column layout
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.header("Analysis Results")
            if st.session_state.analysis_done:
                st.text_area(
                    "File Analysis (English Descriptions)",
                    value=st.session_state.analysis_result,
                    height=400,
                    key="analysis_output"
                )
            else:
                st.info("Click 'Process Files' in the sidebar to analyze your data.")

        with col2:
            st.header("Ask Questions")
            
            # Question input
            question = st.text_input(
                "Ask a question about your data:",
                placeholder="e.g., What is the total sales? Which region has maximum profit?",
                key="question_input"
            )
            
            if st.button("Ask Question", type="primary") or (question and st.session_state.get('last_question') != question):
                if question.strip():
                    if st.session_state.analysis_done:
                        with st.spinner("Analyzing your question..."):
                            try:
                                answer = st.session_state.analyzer.query_data(question)
                                st.session_state.last_question = question
                                st.session_state.last_answer = answer
                            except Exception as e:
                                st.error(f"Error: {str(e)}")
                    else:
                        st.warning("Please process your files first!")
                else:
                    st.warning("Please enter a question.")

            # Display answer
            if 'last_answer' in st.session_state:
                st.subheader("Answer & Insights")
                st.text_area(
                    "Results:",
                    value=st.session_state.last_answer,
                    height=300,
                    key="answer_output"
                )

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <p>Built with Streamlit and OpenAI | Upload your data and start exploring!</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
