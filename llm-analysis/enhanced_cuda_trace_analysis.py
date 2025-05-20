#!/usr/bin/env python3
"""
Enhanced CUDA Trace Analysis - Main program integrating all components
"""

import os
import sys
import json
import argparse
import time
from datetime import datetime
import shutil
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Import custom modules
from enhanced_cuda_llm_analyzer import EnhancedCUDATraceLLMAnalyzer
from cuda_prompt_templates import CUDAPromptTemplates
from cuda_llm_analysis_tester import CUDALLMAnalysisTester

def parse_trace_file(trace_file, output_dir):
    """Parse the CUDA trace file using the existing parser"""
    print(f"Parsing trace file: {trace_file}")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Check if the trace file exists
    if not os.path.exists(trace_file):
        print(f"Error: Trace file not found at {trace_file}")
        return None
    
    # Check if cuda_trace_parser.py exists in the current directory
    parser_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cuda_trace_parser.py")
    if not os.path.exists(parser_path):
        print(f"Error: cuda_trace_parser.py not found at {parser_path}")
        return None
    
    # Run the parser
    parsed_output = os.path.join(output_dir, "parsed_trace.json")
    cmd = [sys.executable, parser_path, trace_file, "--output", parsed_output]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"Trace file parsed successfully. Output saved to {parsed_output}")
        return parsed_output
    except subprocess.CalledProcessError as e:
        print(f"Error parsing trace file: {e}")
        return None

def analyze_trace_data(parsed_trace, output_dir):
    """Analyze the parsed trace data using the existing analyzer"""
    print(f"Analyzing trace data from: {parsed_trace}")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Check if cuda_trace_analyzer.py exists in the current directory
    analyzer_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cuda_trace_analyzer.py")
    if not os.path.exists(analyzer_path):
        print(f"Error: cuda_trace_analyzer.py not found at {analyzer_path}")
        return None
    
    # Run the analyzer
    cmd = [sys.executable, analyzer_path, parsed_trace, "--output_dir", output_dir]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"Trace data analyzed successfully. Results saved to {output_dir}")
        return output_dir
    except subprocess.CalledProcessError as e:
        print(f"Error analyzing trace data: {e}")
        return None

def enhance_visualizations(analysis_dir, enhanced_dir):
    """Enhance the visualizations using the existing visualization organizer"""
    print(f"Enhancing visualizations from: {analysis_dir}")
    
    # Create enhanced directory if it doesn't exist
    if not os.path.exists(enhanced_dir):
        os.makedirs(enhanced_dir)
    
    # Check if cuda_visualization_organizer.py exists in the current directory
    organizer_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cuda_visualization_organizer.py")
    if not os.path.exists(organizer_path):
        print(f"Error: cuda_visualization_organizer.py not found at {organizer_path}")
        return None
    
    # Run the visualization organizer
    cmd = [sys.executable, organizer_path, analysis_dir, "--output_dir", enhanced_dir]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"Visualizations enhanced successfully. Results saved to {enhanced_dir}")
        return enhanced_dir
    except subprocess.CalledProcessError as e:
        print(f"Error enhancing visualizations: {e}")
        return None

def perform_llm_analysis(analysis_dir, enhanced_dir, llm_mode, api_key=None, model_endpoint=None):
    """Perform LLM analysis using the enhanced LLM analyzer"""
    print(f"Performing LLM analysis using mode: {llm_mode}")
    
    # Initialize the enhanced LLM analyzer
    analyzer = EnhancedCUDATraceLLMAnalyzer(analysis_dir, enhanced_dir)
    
    # Perform the analysis
    if llm_mode == "openai":
        section_analyses = analyzer.analyze_with_openai(api_key)
    elif llm_mode == "local":
        section_analyses = analyzer.analyze_with_local_llm(model_endpoint)
    else:  # mock mode
        section_analyses = analyzer.generate_mock_analysis()
    
    if not section_analyses:
        print("Error: LLM analysis failed.")
        return None
    
    print(f"LLM analysis completed successfully. Results saved to {os.path.join(analysis_dir, 'llm_analysis')}")
    print(f"HTML report saved to {os.path.join(analysis_dir, 'html_report')}")
    
    return section_analyses

def test_llm_analysis(analysis_dir, enhanced_dir, test_dir, llm_mode, api_key=None, model_endpoint=None):
    """Test the LLM analysis using the testing framework"""
    print(f"Testing LLM analysis using mode: {llm_mode}")
    
    # Initialize the LLM analysis tester
    tester = CUDALLMAnalysisTester(analysis_dir, enhanced_dir, test_dir)
    
    # Perform the tests
    if llm_mode == "openai":
        test_results, quality_metrics = tester.test_openai_analysis(api_key)
    elif llm_mode == "local":
        test_results, quality_metrics = tester.test_local_llm_analysis(model_endpoint)
    else:  # mock mode
        test_results = tester.test_mock_analysis()
        quality_metrics = None
    
    if not test_results:
        print("Error: LLM analysis testing failed.")
        return None
    
    print(f"LLM analysis testing completed successfully. Results saved to {test_dir}")
    
    return test_results, quality_metrics

def create_final_report(analysis_dir, enhanced_dir, llm_analysis_dir, output_file):
    """Create a final report combining all analysis results"""
    print(f"Creating final report: {output_file}")
    
    # Load analysis results
    analysis_results_path = os.path.join(analysis_dir, "analysis_results.json")
    if os.path.exists(analysis_results_path):
        with open(analysis_results_path, 'r') as f:
            analysis_results = json.load(f)
    else:
        print(f"Warning: Analysis results not found at {analysis_results_path}")
        analysis_results = {}
    
    # Load summary
    summary_path = os.path.join(analysis_dir, "analysis_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            summary = json.load(f)
    else:
        print(f"Warning: Analysis summary not found at {summary_path}")
        summary = {}
    
    # Find the combined LLM analysis
    combined_analysis_path = None
    for filename in os.listdir(llm_analysis_dir):
        if filename.startswith("llm_analysis_combined_"):
            combined_analysis_path = os.path.join(llm_analysis_dir, filename)
            break
    
    if not combined_analysis_path:
        print("Warning: Combined LLM analysis not found")
        combined_analysis = "LLM analysis not available."
    else:
        with open(combined_analysis_path, 'r') as f:
            combined_analysis = f.read()
    
    # Find the HTML report
    html_report_dir = os.path.join(analysis_dir, "html_report")
    html_report_path = None
    if os.path.exists(html_report_dir):
        for filename in os.listdir(html_report_dir):
            if filename.startswith("cuda_trace_analysis_report_"):
                html_report_path = os.path.join(html_report_dir, filename)
                break
    
    # Create the final report
    report = f"""# CUDA Trace Analysis Report

## Summary

- Total trace entries: {summary.get('total_trace_entries', 'Unknown')}
- Unique API functions: {summary.get('unique_api_functions', 'Unknown')}
- Unique kernels: {summary.get('unique_kernels', 'Unknown')}
- Trace duration: {summary.get('trace_duration_seconds', 'Unknown')} seconds

## Analysis Results

{combined_analysis}

## Visualizations

The following visualizations are available in the analysis directory:

- API Distribution: {os.path.join(analysis_dir, "api_distribution.png")}
- Kernel Name Distribution: {os.path.join(analysis_dir, "kernel_name_distribution.png")}
- Memory Operations: {os.path.join(analysis_dir, "memory_operations.png")}
- API Call Timeline: {os.path.join(analysis_dir, "api_call_timeline.png")}

Enhanced visualizations are available in the enhanced directory:

- CUDA Trace Dashboard: {os.path.join(enhanced_dir, "cuda_trace_dashboard.png")}

## HTML Report

An interactive HTML report is available at:

{html_report_path if html_report_path else "HTML report not available."}

## Generated on

{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
    
    # Write the report to file
    with open(output_file, 'w') as f:
        f.write(report)
    
    print(f"Final report created successfully: {output_file}")
    
    return output_file

def main():
    parser = argparse.ArgumentParser(description="Enhanced CUDA Trace Analysis")
    parser.add_argument("trace_file", help="Path to the CUDA trace file")
    parser.add_argument("--output_dir", default="./cuda_analysis_results", help="Output directory for analysis results")
    parser.add_argument("--llm_mode", choices=["mock", "openai", "local"], default="mock", help="LLM analysis mode")
    parser.add_argument("--api_key", help="OpenAI API key (for openai mode)")
    parser.add_argument("--model_endpoint", default="http://localhost:8000/v1/chat/completions", help="Local LLM API endpoint (for local mode)")
    parser.add_argument("--skip_parsing", action="store_true", help="Skip trace file parsing (use existing parsed data)")
    parser.add_argument("--skip_analysis", action="store_true", help="Skip trace data analysis (use existing analysis results)")
    parser.add_argument("--skip_visualization", action="store_true", help="Skip visualization enhancement (use existing enhanced visualizations)")
    parser.add_argument("--test_llm", action="store_true", help="Test LLM analysis using the testing framework")
    
    args = parser.parse_args()
    
    # Create output directories
    output_dir = os.path.abspath(args.output_dir)
    analysis_dir = os.path.join(output_dir, "analysis")
    enhanced_dir = os.path.join(output_dir, "enhanced")
    llm_analysis_dir = os.path.join(analysis_dir, "llm_analysis")
    test_dir = os.path.join(output_dir, "test_results")
    
    for directory in [output_dir, analysis_dir, enhanced_dir, llm_analysis_dir, test_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    # Step 1: Parse the trace file
    if not args.skip_parsing:
        parsed_trace = parse_trace_file(args.trace_file, output_dir)
        if not parsed_trace:
            print("Error: Failed to parse trace file. Exiting.")
            return 1
    else:
        parsed_trace = os.path.join(output_dir, "parsed_trace.json")
        if not os.path.exists(parsed_trace):
            print(f"Error: Parsed trace file not found at {parsed_trace}. Please run without --skip_parsing.")
            return 1
        print(f"Using existing parsed trace file: {parsed_trace}")
    
    # Step 2: Analyze the trace data
    if not args.skip_analysis:
        analysis_results = analyze_trace_data(parsed_trace, analysis_dir)
        if not analysis_results:
            print("Error: Failed to analyze trace data. Exiting.")
            return 1
    else:
        if not os.path.exists(os.path.join(analysis_dir, "analysis_results.json")):
            print(f"Error: Analysis results not found in {analysis_dir}. Please run without --skip_analysis.")
            return 1
        print(f"Using existing analysis results in: {analysis_dir}")
    
    # Step 3: Enhance the visualizations
    if not args.skip_visualization:
        enhanced_results = enhance_visualizations(analysis_dir, enhanced_dir)
        if not enhanced_results:
            print("Error: Failed to enhance visualizations. Exiting.")
            return 1
    else:
        if not os.path.exists(os.path.join(enhanced_dir, "cuda_trace_dashboard.png")):
            print(f"Error: Enhanced visualizations not found in {enhanced_dir}. Please run without --skip_visualization.")
            return 1
        print(f"Using existing enhanced visualizations in: {enhanced_dir}")
    
    # Step 4: Perform LLM analysis
    section_analyses = perform_llm_analysis(analysis_dir, enhanced_dir, args.llm_mode, args.api_key, args.model_endpoint)
    if not section_analyses:
        print("Error: Failed to perform LLM analysis. Exiting.")
        return 1
    
    # Step 5: Test LLM analysis (optional)
    if args.test_llm:
        test_results = test_llm_analysis(analysis_dir, enhanced_dir, test_dir, args.llm_mode, args.api_key, args.model_endpoint)
        if not test_results:
            print("Warning: LLM analysis testing failed.")
    
    # Step 6: Create final report
    final_report = create_final_report(analysis_dir, enhanced_dir, llm_analysis_dir, os.path.join(output_dir, "final_report.md"))
    if not final_report:
        print("Error: Failed to create final report. Exiting.")
        return 1
    
    print("\nAnalysis completed successfully!")
    print(f"Results saved to: {output_dir}")
    print(f"Final report: {final_report}")
    print(f"HTML report: {os.path.join(analysis_dir, 'html_report')}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
