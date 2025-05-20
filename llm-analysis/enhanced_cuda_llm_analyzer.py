#!/usr/bin/env python3
"""
Enhanced CUDA Trace LLM Analyzer - Integrates LLM analysis with visualizations and tables
"""

import os
import json
import base64
import requests
from PIL import Image
import io
import argparse
import sys
import time
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from jinja2 import Template, Environment, FileSystemLoader
from openai import OpenAI

class EnhancedCUDATraceLLMAnalyzer:
    """Integrates LLM analysis with visualizations and tables for CUDA trace data"""
    
    def __init__(self, analysis_dir, enhanced_dir=None):
        """Initialize the enhanced LLM analyzer with the path to the analysis results directory"""
        self.analysis_dir = analysis_dir
        self.enhanced_dir = enhanced_dir or os.path.join(analysis_dir, "enhanced")
        self.output_dir = os.path.join(analysis_dir, "llm_analysis")
        self.html_output_dir = os.path.join(analysis_dir, "html_report")
        
        # Create output directories if they don't exist
        for directory in [self.output_dir, self.html_output_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
        
        # Load analysis results
        self.analysis_results = self._load_analysis_results()
        self.summary = self._load_summary()
        
        # Initialize section analyses
        self.section_analyses = {}
    
    def _load_analysis_results(self):
        """Load the analysis results from JSON file"""
        results_path = os.path.join(self.analysis_dir, "analysis_results.json")
        if os.path.exists(results_path):
            with open(results_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _load_summary(self):
        """Load the analysis summary from JSON file"""
        summary_path = os.path.join(self.analysis_dir, "analysis_summary.json")
        if os.path.exists(summary_path):
            with open(summary_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _encode_image(self, image_path):
        """Encode an image to base64 for LLM API"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def _generate_tables(self):
        """Generate HTML and Markdown tables from analysis results"""
        tables = {}
        
        # API Distribution Table
        if 'api_distribution' in self.analysis_results:
            api_data = pd.DataFrame(self.analysis_results['api_distribution'])
            if not api_data.empty:
                # Calculate percentage
                total_calls = api_data['Count'].sum()
                api_data['Percentage'] = (api_data['Count'] / total_calls * 100).round(2)
                
                # Format as HTML and Markdown
                api_html = api_data.head(10).to_html(index=False, classes="data-table")
                api_md = api_data.head(10).to_markdown(index=False)
                
                tables['api_distribution'] = {
                    'html': api_html,
                    'markdown': api_md,
                    'data': api_data.head(10).to_dict('records')
                }
        
        # Kernel Launch Table
        if 'kernel_name_distribution' in self.analysis_results:
            kernel_data = pd.DataFrame(self.analysis_results['kernel_name_distribution'])
            if not kernel_data.empty:
                # Calculate percentage
                total_launches = kernel_data['Count'].sum()
                kernel_data['Percentage'] = (kernel_data['Count'] / total_launches * 100).round(2)
                
                # Format as HTML and Markdown
                kernel_html = kernel_data.head(10).to_html(index=False, classes="data-table")
                kernel_md = kernel_data.head(10).to_markdown(index=False)
                
                tables['kernel_distribution'] = {
                    'html': kernel_html,
                    'markdown': kernel_md,
                    'data': kernel_data.head(10).to_dict('records')
                }
        
        # Memory Operations Table
        if 'memory_operations' in self.analysis_results:
            memory_data = pd.DataFrame(self.analysis_results['memory_operations'])
            if not memory_data.empty:
                # Calculate percentage
                total_ops = memory_data['Count'].sum()
                memory_data['Percentage'] = (memory_data['Count'] / total_ops * 100).round(2)
                
                # Format as HTML and Markdown
                memory_html = memory_data.to_html(index=False, classes="data-table")
                memory_md = memory_data.to_markdown(index=False)
                
                tables['memory_operations'] = {
                    'html': memory_html,
                    'markdown': memory_md,
                    'data': memory_data.to_dict('records')
                }
        
        # Grid/Block Dimensions Table
        if 'grid_block_dimensions' in self.analysis_results:
            grid_block_data = pd.DataFrame(self.analysis_results['grid_block_dimensions'])
            if not grid_block_data.empty:
                # Pivot the data for better presentation
                pivot_data = grid_block_data.pivot_table(
                    index='Dimension', 
                    columns='Value', 
                    values='Count',
                    fill_value=0
                ).reset_index()
                
                # Format as HTML and Markdown
                grid_block_html = pivot_data.to_html(classes="data-table")
                grid_block_md = pivot_data.to_markdown()
                
                tables['grid_block_dimensions'] = {
                    'html': grid_block_html,
                    'markdown': grid_block_md,
                    'data': grid_block_data.to_dict('records')
                }
        
        # Performance Bottlenecks Table (derived from analysis)
        bottlenecks = []
        
        # Check for memory transfer bottlenecks
        if 'memory_operations' in self.analysis_results:
            memory_ops = pd.DataFrame(self.analysis_results['memory_operations'])
            memcpy_ops = memory_ops[memory_ops['Memory Operation'].str.contains('cudaMemcpy', na=False)]
            if not memcpy_ops.empty:
                memcpy_count = memcpy_ops['Count'].sum()
                
                # Check if memcpy operations are a significant portion of all operations
                if 'api_distribution' in self.analysis_results:
                    api_data = pd.DataFrame(self.analysis_results['api_distribution'])
                    total_ops = api_data['Count'].sum()
                    memcpy_percentage = (memcpy_count / total_ops) * 100
                    
                    if memcpy_percentage > 20:  # Threshold for bottleneck
                        bottlenecks.append({
                            'Bottleneck': 'Memory Transfer Overhead',
                            'Metric': f'{memcpy_percentage:.2f}% of operations',
                            'Severity': 'High' if memcpy_percentage > 30 else 'Medium',
                            'Recommendation': 'Use pinned memory, batch transfers, or keep data on GPU longer'
                        })
        
        # Check for synchronization bottlenecks
        sync_ops = ['cudaStreamSynchronize', 'cudaDeviceSynchronize', 'cudaEventSynchronize']
        sync_count = 0
        
        if 'api_distribution' in self.analysis_results:
            api_data = pd.DataFrame(self.analysis_results['api_distribution'])
            for op in sync_ops:
                sync_op = api_data[api_data['API Function'] == op]
                if not sync_op.empty:
                    sync_count += sync_op.iloc[0]['Count']
            
            if sync_count > 0:
                total_ops = api_data['Count'].sum()
                sync_percentage = (sync_count / total_ops) * 100
                
                if sync_percentage > 5:  # Threshold for bottleneck
                    bottlenecks.append({
                        'Bottleneck': 'Excessive Synchronization',
                        'Metric': f'{sync_count} sync operations ({sync_percentage:.2f}%)',
                        'Severity': 'High' if sync_percentage > 10 else 'Medium',
                        'Recommendation': 'Reduce synchronization points, use multiple streams for parallelism'
                    })
        
        # Check for small kernel launches
        if 'grid_block_dimensions' in self.analysis_results:
            grid_block_data = pd.DataFrame(self.analysis_results['grid_block_dimensions'])
            block_x = grid_block_data[(grid_block_data['Dimension'] == 'block_x') & (grid_block_data['Value'] != 0)]
            
            if not block_x.empty and isinstance(block_x.iloc[0]['Value'], (int, float)):
                block_size = block_x.iloc[0]['Value']
                if block_size < 128:
                    bottlenecks.append({
                        'Bottleneck': 'Low GPU Occupancy',
                        'Metric': f'Block size: {block_size} threads',
                        'Severity': 'Medium',
                        'Recommendation': 'Increase threads per block (ideal: 128-256)'
                    })
        
        # Create bottlenecks table
        if bottlenecks:
            bottlenecks_df = pd.DataFrame(bottlenecks)
            bottlenecks_html = bottlenecks_df.to_html(index=False, classes="data-table")
            bottlenecks_md = bottlenecks_df.to_markdown(index=False)
            
            tables['performance_bottlenecks'] = {
                'html': bottlenecks_html,
                'markdown': bottlenecks_md,
                'data': bottlenecks
            }
        
        return tables
    
    def _prepare_section_prompts(self):
        """Prepare prompts for each section of the analysis"""
        tables = self._generate_tables()
        prompts = {}
        
        # Overview Section Prompt
        overview_prompt = """
        # CUDA Trace Analysis - Overview

        I need a comprehensive overview of the CUDA trace data provided. The data comes from a BPF program that traces various CUDA API routines.
        
        ## Analysis Task
        
        Please provide a high-level executive summary of the CUDA trace data that includes:
        
        1. The main characteristics of the application based on its CUDA API usage
        2. The most significant patterns observed in the trace data
        3. The key performance considerations identified
        4. A brief assessment of the application's CUDA implementation quality
        
        ## Available Data
        
        """
        
        # Add summary information
        if self.summary:
            overview_prompt += "### Summary Statistics\n\n"
            overview_prompt += f"- Total trace entries: {self.summary.get('total_trace_entries', 'Unknown')}\n"
            overview_prompt += f"- Unique API functions: {self.summary.get('unique_api_functions', 'Unknown')}\n"
            overview_prompt += f"- Unique kernels: {self.summary.get('unique_kernels', 'Unknown')}\n"
            overview_prompt += f"- Trace duration: {self.summary.get('trace_duration_seconds', 'Unknown')} seconds\n\n"
        
        prompts['overview'] = overview_prompt
        
        # API Distribution Section Prompt
        api_prompt = """
        # CUDA Trace Analysis - API Distribution

        I need a detailed analysis of the CUDA API distribution in the trace data.
        
        ## Analysis Task
        
        Please analyze the API distribution data and provide:
        
        1. Insights into the most frequently used CUDA API functions and what they indicate about the application
        2. Analysis of the balance between different types of operations (compute, memory, synchronization)
        3. Identification of any unusual or inefficient API usage patterns
        4. Recommendations for optimizing the API usage
        
        ## Available Data
        
        """
        
        # Add API distribution information
        if 'api_distribution' in tables:
            api_prompt += "### API Distribution Table\n\n"
            api_prompt += tables['api_distribution']['markdown'] + "\n\n"
        
        # Reference to visualization
        api_prompt += "### API Distribution Visualization\n\n"
        api_prompt += "The API distribution visualization (Figure 1) shows the frequency of different CUDA API calls.\n\n"
        
        prompts['api_distribution'] = api_prompt
        
        # Memory Operations Section Prompt
        memory_prompt = """
        # CUDA Trace Analysis - Memory Operations

        I need a detailed analysis of the memory operations in the CUDA trace data.
        
        ## Analysis Task
        
        Please analyze the memory operations data and provide:
        
        1. Assessment of the memory transfer patterns and their efficiency
        2. Analysis of the balance between different types of memory operations
        3. Identification of potential memory-related bottlenecks
        4. Recommendations for optimizing memory usage and transfers
        
        ## Available Data
        
        """
        
        # Add memory operations information
        if 'memory_operations' in tables:
            memory_prompt += "### Memory Operations Table\n\n"
            memory_prompt += tables['memory_operations']['markdown'] + "\n\n"
        
        # Reference to visualization
        memory_prompt += "### Memory Operations Visualization\n\n"
        memory_prompt += "The memory operations visualization (Figure 2) shows the distribution of memory-related operations.\n\n"
        
        prompts['memory_operations'] = memory_prompt
        
        # Kernel Launch Section Prompt
        kernel_prompt = """
        # CUDA Trace Analysis - Kernel Launches

        I need a detailed analysis of the kernel launch patterns in the CUDA trace data.
        
        ## Analysis Task
        
        Please analyze the kernel launch data and provide:
        
        1. Assessment of the kernel launch patterns and their implications for performance
        2. Analysis of the grid and block dimensions used for kernel launches
        3. Evaluation of kernel occupancy and efficiency based on the launch parameters
        4. Recommendations for optimizing kernel launch configurations
        
        ## Available Data
        
        """
        
        # Add kernel distribution information
        if 'kernel_distribution' in tables:
            kernel_prompt += "### Kernel Distribution Table\n\n"
            kernel_prompt += tables['kernel_distribution']['markdown'] + "\n\n"
        
        # Add grid/block dimensions information
        if 'grid_block_dimensions' in tables:
            kernel_prompt += "### Grid/Block Dimensions Table\n\n"
            kernel_prompt += tables['grid_block_dimensions']['markdown'] + "\n\n"
        
        # Reference to visualization
        kernel_prompt += "### Kernel Launch Visualization\n\n"
        kernel_prompt += "The kernel launch visualization (Figure 3) shows the distribution of kernel launches.\n\n"
        
        prompts['kernel_launches'] = kernel_prompt
        
        # Performance Bottlenecks Section Prompt
        bottlenecks_prompt = """
        # CUDA Trace Analysis - Performance Bottlenecks

        I need a detailed analysis of the performance bottlenecks identified in the CUDA trace data.
        
        ## Analysis Task
        
        Please analyze the performance bottlenecks data and provide:
        
        1. Detailed explanation of each identified bottleneck and its impact on performance
        2. Root cause analysis for each bottleneck
        3. Prioritized recommendations for addressing each bottleneck
        4. Potential performance gains from implementing the recommendations
        
        ## Available Data
        
        """
        
        # Add bottlenecks information
        if 'performance_bottlenecks' in tables:
            bottlenecks_prompt += "### Performance Bottlenecks Table\n\n"
            bottlenecks_prompt += tables['performance_bottlenecks']['markdown'] + "\n\n"
        
        # Reference to visualization
        bottlenecks_prompt += "### Performance Timeline Visualization\n\n"
        bottlenecks_prompt += "The performance timeline visualization (Figure 4) shows the API calls over time, which can help identify bottlenecks.\n\n"
        
        prompts['performance_bottlenecks'] = bottlenecks_prompt
        
        # Optimization Recommendations Section Prompt
        optimization_prompt = """
        # CUDA Trace Analysis - Optimization Recommendations

        I need detailed optimization recommendations based on the CUDA trace data analysis.
        
        ## Analysis Task
        
        Please provide comprehensive optimization recommendations that include:
        
        1. Specific code-level optimizations with examples where possible
        2. Architectural changes to improve performance
        3. Alternative approaches or CUDA features that could be leveraged
        4. Prioritization of recommendations based on expected impact
        
        ## Available Data
        
        """
        
        # Add summary of findings from other sections
        optimization_prompt += "### Summary of Findings\n\n"
        optimization_prompt += "Based on the analysis of the trace data, please provide optimization recommendations that address the identified issues.\n\n"
        
        # Reference to dashboard
        optimization_prompt += "### CUDA Trace Dashboard\n\n"
        optimization_prompt += "The CUDA trace dashboard (Figure 5) provides an overview of all aspects of the trace data.\n\n"
        
        prompts['optimization_recommendations'] = optimization_prompt
        
        return prompts, tables
    
    def analyze_with_openai(self, api_key=None):
        """Analyze the trace data using OpenAI's API with enhanced visualization integration"""
        if not api_key:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                print("Error: OpenAI API key not provided. Please set the OPENAI_API_KEY environment variable or provide it as an argument.")
                return None
        
        print("Analyzing trace data with OpenAI's API (enhanced visualization integration)...")
        
        # Initialize the OpenAI client
        client = OpenAI(api_key=api_key)
        
        # Prepare section prompts and tables
        section_prompts, tables = self._prepare_section_prompts()
        
        # Get list of visualization images
        dashboard_path = os.path.join(self.enhanced_dir, "cuda_trace_dashboard.png")
        api_dist_path = os.path.join(self.analysis_dir, "api_distribution.png")
        kernel_path = os.path.join(self.analysis_dir, "kernel_name_distribution.png")
        memory_path = os.path.join(self.analysis_dir, "memory_operations.png")
        timeline_path = os.path.join(self.analysis_dir, "api_call_timeline.png")
        
        # Encode images
        encoded_images = {}
        for img_id, img_path in [
            ("dashboard", dashboard_path),
            ("api_distribution", api_dist_path),
            ("kernel_distribution", kernel_path),
            ("memory_operations", memory_path),
            ("timeline", timeline_path)
        ]:
            if os.path.exists(img_path):
                try:
                    encoded_images[img_id] = self._encode_image(img_path)
                except Exception as e:
                    print(f"Error encoding image {img_path}: {e}")
        
        # Analyze each section
        for section, prompt in section_prompts.items():
            print(f"Analyzing section: {section}")
            
            # Prepare the message content
            content = [
                {
                    "type": "text",
                    "text": prompt
                }
            ]
            
            # Add relevant images for this section
            if section == 'overview' and 'dashboard' in encoded_images:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{encoded_images['dashboard']}"
                    }
                })
            elif section == 'api_distribution' and 'api_distribution' in encoded_images:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{encoded_images['api_distribution']}"
                    }
                })
            elif section == 'memory_operations' and 'memory_operations' in encoded_images:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{encoded_images['memory_operations']}"
                    }
                })
            elif section == 'kernel_launches' and 'kernel_distribution' in encoded_images:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{encoded_images['kernel_distribution']}"
                    }
                })
            
            try:
                # Make the API request using the OpenAI client
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "user",
                            "content": content
                        }
                    ],
                    max_tokens=4000
                )
                
                # Parse the response
                if response.choices and len(response.choices) > 0:
                    analysis = response.choices[0].message.content
                    
                    # Save the section analysis
                    self.section_analyses[section] = analysis
                    
                    # Save the analysis to a file
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_path = os.path.join(self.output_dir, f"llm_analysis_{section}_{timestamp}.md")
                    with open(output_path, "w") as f:
                        f.write(analysis)
                    
                    print(f"Analysis for section '{section}' saved to {output_path}")
                else:
                    print(f"Error: Unexpected response format from OpenAI API for section '{section}'")
            
            except Exception as e:
                print(f"Error calling OpenAI API for section '{section}': {e}")
        
        # Generate the combined analysis
        self._generate_combined_analysis()
        
        # Generate the HTML report
        self._generate_html_report(tables, encoded_images)
        
        return self.section_analyses
    
    def analyze_with_local_llm(self, model_endpoint="http://localhost:8000/v1/chat/completions"):
        """Analyze the trace data using a local LLM API endpoint with enhanced visualization integration"""
        print(f"Analyzing trace data with local LLM at {model_endpoint} (enhanced visualization integration)...")
        
        # Prepare section prompts and tables
        section_prompts, tables = self._prepare_section_prompts()
        
        # Analyze each section
        for section, prompt in section_prompts.items():
            print(f"Analyzing section: {section}")
            
            # Prepare the API request
            headers = {
                "Content-Type": "application/json"
            }
            
            # Prepare the API request payload
            payload = {
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": 4000
            }
            
            try:
                # Make the API request
                response = requests.post(
                    model_endpoint,
                    headers=headers,
                    json=payload
                )
                
                # Check for errors
                response.raise_for_status()
                
                # Parse the response
                result = response.json()
                
                if "choices" in result and len(result["choices"]) > 0:
                    analysis = result["choices"][0]["message"]["content"]
                    
                    # Save the section analysis
                    self.section_analyses[section] = analysis
                    
                    # Save the analysis to a file
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_path = os.path.join(self.output_dir, f"llm_analysis_{section}_{timestamp}.md")
                    with open(output_path, "w") as f:
                        f.write(analysis)
                    
                    print(f"Analysis for section '{section}' saved to {output_path}")
                else:
                    print(f"Error: Unexpected response format from local LLM API for section '{section}'")
            
            except Exception as e:
                print(f"Error calling local LLM API for section '{section}': {e}")
        
        # Generate the combined analysis
        self._generate_combined_analysis()
        
        # Generate the HTML report
        self._generate_html_report(tables, {})
        
        return self.section_analyses
    
    def generate_mock_analysis(self):
        """Generate a mock analysis for testing purposes"""
        print("Generating mock analysis...")
        
        # Prepare section prompts and tables
        section_prompts, tables = self._prepare_section_prompts()
        
        # Generate mock analysis for each section
        for section in section_prompts.keys():
            print(f"Generating mock analysis for section: {section}")
            
            # Generate a mock analysis based on the section
            if section == 'overview':
                analysis = self._generate_mock_overview()
            elif section == 'api_distribution':
                analysis = self._generate_mock_api_distribution()
            elif section == 'memory_operations':
                analysis = self._generate_mock_memory_operations()
            elif section == 'kernel_launches':
                analysis = self._generate_mock_kernel_launches()
            elif section == 'performance_bottlenecks':
                analysis = self._generate_mock_performance_bottlenecks()
            elif section == 'optimization_recommendations':
                analysis = self._generate_mock_optimization_recommendations()
            else:
                analysis = f"# Mock Analysis for {section}\n\nThis is a placeholder for the {section} analysis."
            
            # Save the section analysis
            self.section_analyses[section] = analysis
            
            # Save the analysis to a file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.output_dir, f"llm_analysis_{section}_{timestamp}.md")
            with open(output_path, "w") as f:
                f.write(analysis)
            
            print(f"Mock analysis for section '{section}' saved to {output_path}")
        
        # Generate the combined analysis
        self._generate_combined_analysis()
        
        # Generate the HTML report
        self._generate_html_report(tables, {})
        
        return self.section_analyses
    
    def _generate_mock_overview(self):
        """Generate a mock overview analysis"""
        return """# CUDA Trace Analysis Overview

## Executive Summary

Based on the CUDA trace data provided, this application appears to be a **compute-intensive application with significant memory transfer operations**. The trace reveals a pattern of kernel launches interspersed with memory operations, suggesting a batch processing workflow where data is transferred to the GPU, processed, and then results are transferred back to the host.

## Key Characteristics

1. **Balanced API Usage**: The application shows a relatively balanced distribution of CUDA API calls across memory management, kernel execution, and synchronization operations.

2. **Regular Kernel Launch Pattern**: The kernel launch pattern suggests a structured, iterative computation approach, likely processing data in batches or chunks.

3. **Memory Transfer Overhead**: A significant portion of the trace is dedicated to memory transfer operations, indicating potential for optimization in this area.

4. **Synchronization Points**: The trace shows regular synchronization points, which may be limiting the potential for overlapping operations.

## Performance Considerations

- **Memory Transfer Bottleneck**: The high frequency of memory transfer operations suggests this could be a performance bottleneck.
- **Synchronization Overhead**: Regular synchronization points may be limiting parallelism and overall performance.
- **Kernel Efficiency**: Some kernels show suboptimal grid and block dimensions, potentially underutilizing the GPU.

## Implementation Quality Assessment

The CUDA implementation appears to be **functionally correct but with room for optimization**. The structured approach suggests a methodical implementation, but the frequency of memory transfers and synchronization points indicates that the application may not be fully leveraging CUDA's parallel processing capabilities.

Overall, this appears to be a **moderate-quality CUDA implementation** that could benefit from targeted optimizations to reduce memory transfer overhead and increase parallelism.
"""
    
    def _generate_mock_api_distribution(self):
        """Generate a mock API distribution analysis"""
        return """# CUDA API Distribution Analysis

## Key Insights

The API distribution reveals a **compute-focused application with significant memory management overhead**. The most frequently used API functions provide valuable insights into the application's behavior and potential optimization opportunities.

## Most Frequently Used API Functions

1. **cudaLaunchKernel (32.5%)**: The high frequency of kernel launches indicates a compute-intensive application. This is expected for a CUDA application, as the primary purpose is to offload computation to the GPU.

2. **cudaMemcpy (18.7%)**: The significant proportion of memory copy operations suggests substantial data transfer between host and device. This could be a performance bottleneck, especially if these transfers are not overlapped with computation.

3. **cudaMalloc (12.3%)**: The frequency of memory allocation calls indicates that the application may be allocating memory frequently during execution rather than pre-allocating at initialization.

4. **cudaStreamSynchronize (8.5%)**: The presence of stream synchronization calls suggests the application is using CUDA streams, but the relatively high frequency may indicate excessive synchronization.

5. **cudaFree (7.9%)**: The proportion of memory deallocation calls matches closely with allocation calls, suggesting good memory management practices.

## Operation Type Balance

- **Compute Operations**: ~35% (primarily cudaLaunchKernel)
- **Memory Operations**: ~40% (cudaMemcpy, cudaMalloc, cudaFree)
- **Synchronization Operations**: ~15% (cudaStreamSynchronize, cudaDeviceSynchronize)
- **Other Operations**: ~10% (configuration, device management, etc.)

This distribution shows a relatively balanced application with a slight emphasis on memory operations, which is common in many CUDA applications but may indicate optimization opportunities.

## Inefficient API Usage Patterns

1. **Frequent Memory Allocation/Deallocation**: The high frequency of cudaMalloc and cudaFree calls suggests memory is being allocated and deallocated frequently, which is inefficient. Consider pooling or pre-allocating memory.

2. **Excessive Synchronization**: The proportion of synchronization calls is relatively high, potentially limiting parallelism and performance.

3. **High Memory Transfer Volume**: The significant proportion of cudaMemcpy calls indicates substantial data movement, which is often a performance bottleneck in CUDA applications.

## Optimization Recommendations

1. **Implement Memory Pooling**: Replace frequent cudaMalloc/cudaFree calls with a memory pool to reuse allocated memory.

2. **Reduce Synchronization Points**: Minimize cudaStreamSynchronize calls by restructuring the application to allow more asynchronous operation.

3. **Batch Memory Transfers**: Combine smaller memory transfers into larger batches to reduce overhead.

4. **Use Pinned Memory**: For frequent host-device transfers, use pinned memory to improve transfer speeds.

5. **Implement Asynchronous Memory Copies**: Use cudaMemcpyAsync with streams to overlap memory transfers with computation.

By addressing these API usage patterns, the application could potentially achieve significant performance improvements, particularly in reducing memory management overhead and increasing parallelism.
"""
    
    def _generate_mock_memory_operations(self):
        """Generate a mock memory operations analysis"""
        return """# Memory Operations Analysis

## Memory Transfer Patterns

The trace data reveals several distinct memory transfer patterns that significantly impact the application's performance:

1. **Bulk Data Transfers**: Large, infrequent transfers (typically at the beginning and end of major processing phases)
   - These account for approximately 45% of total memory transfer volume
   - Average transfer size: ~128MB

2. **Regular Small Transfers**: Frequent, small transfers throughout execution
   - These account for approximately 35% of total memory transfer volume
   - Average transfer size: ~256KB
   - Occur at regular intervals, suggesting iterative processing

3. **Sporadic Medium Transfers**: Occasional medium-sized transfers
   - These account for approximately 20% of total memory transfer volume
   - Average transfer size: ~4MB
   - Irregular pattern, possibly related to dynamic data requirements

## Memory Operation Efficiency

The memory operation efficiency shows several areas of concern:

1. **Direction Imbalance**: Host-to-device transfers (cudaMemcpyHostToDevice) account for 65% of all transfers, while device-to-host transfers (cudaMemcpyDeviceToHost) account for 35%. This imbalance suggests that the application may be transferring more data to the GPU than necessary.

2. **Transfer Size Distribution**: The prevalence of small transfers (< 1MB) indicates potential inefficiency, as each transfer incurs overhead regardless of size.

3. **Temporal Clustering**: Memory operations show clustering in time, with periods of intense memory activity followed by computation. This sequential pattern suggests limited overlap between memory transfers and computation.

## Memory-Related Bottlenecks

Based on the analysis, the following memory-related bottlenecks have been identified:

1. **Small Transfer Overhead**: The frequent small transfers incur significant overhead, potentially limiting overall throughput.
   - Impact: Estimated 15-20% performance penalty
   - Root cause: Granular data handling instead of batched processing

2. **Synchronous Memory Operations**: Most memory operations appear to be synchronous, blocking the CPU and preventing overlap with computation.
   - Impact: Estimated 10-15% performance penalty
   - Root cause: Not utilizing asynchronous memory operations and CUDA streams effectively

3. **Repeated Transfers**: Some data appears to be transferred multiple times between host and device.
   - Impact: Estimated 5-10% redundant transfer volume
   - Root cause: Possibly poor data locality or caching strategy

## Optimization Recommendations

To improve memory operation efficiency, consider the following recommendations:

1. **Batch Small Transfers**: Combine small, frequent transfers into larger batches to reduce overhead.
   - Implementation: Buffer data on the host side and transfer in larger chunks
   - Expected impact: 10-15% reduction in memory transfer time

2. **Use Asynchronous Memory Operations**: Implement cudaMemcpyAsync with CUDA streams to overlap memory transfers with computation.
   - Implementation: Restructure code to use multiple streams and asynchronous transfers
   - Expected impact: 15-20% overall performance improvement

3. **Implement Pinned Memory**: Use pinned (page-locked) host memory for frequent transfers to improve bandwidth.
   - Implementation: Replace standard malloc with cudaHostAlloc for transfer buffers
   - Expected impact: 20-30% improvement in memory transfer speed

4. **Reduce Host-Device Transfers**: Keep data on the GPU for longer periods to minimize transfers.
   - Implementation: Restructure algorithms to maximize data reuse on the GPU
   - Expected impact: 10-15% reduction in total transfer volume

5. **Consider Unified Memory**: For appropriate workloads, evaluate using CUDA Unified Memory to simplify memory management.
   - Implementation: Replace explicit memory management with unified memory allocations
   - Expected impact: Simplified code and potentially improved performance for certain access patterns

By implementing these recommendations, the application could significantly reduce memory-related bottlenecks and improve overall performance.
"""
    
    def _generate_mock_kernel_launches(self):
        """Generate a mock kernel launches analysis"""
        return """# Kernel Launch Analysis

## Kernel Launch Patterns

The trace data reveals several distinct kernel launch patterns that characterize the application's execution profile:

1. **Primary Computation Kernels**: A set of 3-4 frequently launched kernels that appear to form the core computational workflow
   - These account for approximately 65% of all kernel launches
   - Launched in a consistent sequence, suggesting a pipeline or iterative algorithm
   - Example kernels: `matrixMultiply`, `vectorAdd`, `dataTransform`

2. **Preprocessing/Postprocessing Kernels**: Less frequent kernels that appear to handle data preparation and result processing
   - These account for approximately 20% of all kernel launches
   - Typically launched before and after the main computation sequence
   - Example kernels: `dataPreprocess`, `resultNormalize`

3. **Utility Kernels**: Infrequently launched kernels for specialized operations
   - These account for approximately 15% of all kernel launches
   - Irregular launch pattern, suggesting on-demand usage
   - Example kernels: `errorCheck`, `memoryInitialize`

## Grid and Block Dimensions Analysis

The grid and block dimensions used for kernel launches show several patterns:

1. **Block Size Distribution**:
   - Most common block size: 256 threads (128 × 1 × 2)
   - Range: 64 to 512 threads per block
   - Observation: Block sizes are generally power-of-two values, which is good practice

2. **Grid Size Distribution**:
   - Highly variable grid sizes, ranging from small (10-20 blocks) to very large (10,000+ blocks)
   - Primary computation kernels tend to use larger grid sizes
   - Utility kernels typically use smaller grid sizes

3. **Dimension Utilization**:
   - Primarily 2D grid configurations (x and y dimensions)
   - Block configurations predominantly 1D or 2D
   - Limited use of 3D configurations, suggesting the application is not processing volumetric data

## Kernel Occupancy and Efficiency

Based on the launch configurations, the following observations about kernel occupancy and efficiency can be made:

1. **Occupancy Concerns**:
   - Some kernels use block sizes of 64 or 128 threads, which may lead to suboptimal occupancy on modern GPUs
   - The primary computation kernels generally use appropriate block sizes (256 threads) for good occupancy

2. **Execution Efficiency**:
   - The consistent use of power-of-two block sizes suggests awareness of warp-based execution
   - Some grid configurations may lead to imbalanced workloads across SMs (Streaming Multiprocessors)

3. **Resource Utilization**:
   - Without detailed kernel code, it's difficult to assess register and shared memory usage
   - The launch patterns suggest compute-bound rather than memory-bound kernels

## Optimization Recommendations

To improve kernel launch efficiency, consider the following recommendations:

1. **Optimize Block Sizes**:
   - Increase block sizes for kernels currently using 64 or 128 threads to 256 threads where possible
   - Implementation: Adjust kernel launch parameters
   - Expected impact: 10-15% improvement in occupancy for affected kernels

2. **Dynamic Grid Sizing**:
   - Implement dynamic grid sizing based on problem dimensions and available GPU resources
   - Implementation: Calculate optimal grid dimensions at runtime based on device properties
   - Expected impact: Better load balancing and potentially 5-10% performance improvement

3. **Kernel Fusion**:
   - Consider combining some of the sequential kernels in the main computation pipeline
   - Implementation: Merge compatible kernels to reduce launch overhead and improve data locality
   - Expected impact: Reduced kernel launch overhead and potentially 10-20% performance improvement for the affected sequence

4. **Persistent Threads**:
   - For iterative algorithms, evaluate using persistent threads to reduce kernel launch overhead
   - Implementation: Restructure kernels to process multiple iterations within a single launch
   - Expected impact: Reduced launch overhead, potentially 5-10% improvement for iterative sections

5. **Explore 3D Block Configurations**:
   - For appropriate algorithms, consider 3D block configurations to better match data access patterns
   - Implementation: Restructure thread indexing to utilize 3D block dimensions
   - Expected impact: Improved memory access patterns, potentially 5-15% performance improvement for spatial algorithms

By implementing these recommendations, the application could achieve better GPU utilization and improved overall performance through more efficient kernel execution.
"""
    
    def _generate_mock_performance_bottlenecks(self):
        """Generate a mock performance bottlenecks analysis"""
        return """# Performance Bottlenecks Analysis

## Identified Bottlenecks

Based on the trace data analysis, the following performance bottlenecks have been identified, listed in order of severity:

### 1. Memory Transfer Overhead (High Severity)

**Description**: Excessive time spent transferring data between host and device memory.

**Impact**: Approximately 35% of total execution time is spent on memory transfers, making this the most significant bottleneck.

**Root Cause**: The application performs frequent, small memory transfers instead of batching them. Additionally, most transfers appear to be synchronous, blocking further execution until completion.

**Recommendations**:
- Batch small transfers into larger chunks
- Use asynchronous memory transfers (cudaMemcpyAsync)
- Implement pinned memory for faster transfer speeds
- Keep data on the GPU for longer periods to reduce transfer frequency

**Expected Performance Gain**: 15-25% reduction in overall execution time.

### 2. Excessive Synchronization (Medium Severity)

**Description**: Frequent synchronization points that limit parallelism and prevent overlapping operations.

**Impact**: Approximately 12% of total execution time is spent waiting at synchronization points.

**Root Cause**: The application uses cudaStreamSynchronize and cudaDeviceSynchronize calls frequently, often after each kernel launch or memory operation.

**Recommendations**:
- Reduce synchronization frequency by grouping operations
- Use multiple CUDA streams to enable operation overlap
- Implement asynchronous execution patterns
- Only synchronize when results are actually needed

**Expected Performance Gain**: 8-12% reduction in overall execution time.

### 3. Suboptimal Kernel Launch Configuration (Medium Severity)

**Description**: Some kernels are launched with grid and block dimensions that do not maximize GPU utilization.

**Impact**: Affected kernels show approximately 30-40% lower throughput than optimal.

**Root Cause**: Block sizes are often too small (64 or 128 threads) for modern GPUs, and grid dimensions don't always balance work evenly across streaming multiprocessors.

**Recommendations**:
- Increase block sizes to 256-512 threads where possible
- Implement dynamic grid sizing based on problem size and GPU capabilities
- Consider kernel fusion for frequently launched sequential kernels

**Expected Performance Gain**: 5-10% reduction in overall execution time.

### 4. Frequent Memory Allocation/Deallocation (Medium Severity)

**Description**: Excessive time spent on cudaMalloc and cudaFree operations during execution.

**Impact**: Approximately 8% of total execution time is spent on memory management.

**Root Cause**: The application allocates and deallocates device memory frequently instead of reusing previously allocated buffers.

**Recommendations**:
- Implement a memory pool to reuse allocated memory
- Pre-allocate memory at initialization when possible
- Use persistent allocations for iterative processing

**Expected Performance Gain**: 3-8% reduction in overall execution time.

### 5. Uncoalesced Memory Access Patterns (Low Severity)

**Description**: Some kernels appear to have suboptimal memory access patterns, reducing memory throughput.

**Impact**: Affected kernels show approximately 20-30% lower memory throughput than optimal.

**Root Cause**: Based on the kernel launch patterns and memory operation distribution, it appears that some kernels may not be accessing memory in a coalesced manner.

**Recommendations**:
- Restructure data layout to improve memory coalescing
- Adjust thread indexing to match memory access patterns
- Consider using shared memory for frequently accessed data

**Expected Performance Gain**: 2-5% reduction in overall execution time.

## Cumulative Impact

If all bottlenecks were addressed, the application could potentially see a **30-45% reduction in overall execution time**. The most significant gains would come from addressing the memory transfer overhead and excessive synchronization issues.

## Implementation Priority

Based on the potential performance gains and implementation complexity, the recommended implementation priority is:

1. Memory Transfer Optimization (highest impact, moderate complexity)
2. Synchronization Reduction (high impact, low complexity)
3. Kernel Launch Configuration Optimization (medium impact, low complexity)
4. Memory Pool Implementation (medium impact, medium complexity)
5. Memory Access Pattern Optimization (lower impact, high complexity)

This prioritization maximizes the performance improvement while considering the implementation effort required.
"""
    
    def _generate_mock_optimization_recommendations(self):
        """Generate a mock optimization recommendations analysis"""
        return """# Optimization Recommendations

## Executive Summary

Based on the comprehensive analysis of the CUDA trace data, this application would benefit significantly from optimizations focused on memory management, kernel execution efficiency, and increased parallelism. The recommendations below are prioritized by expected impact and implementation feasibility.

## High-Priority Optimizations

### 1. Implement Asynchronous Memory Operations with CUDA Streams

**Description**: Replace synchronous memory operations with asynchronous ones and use multiple CUDA streams to overlap memory transfers with computation.

**Implementation**:
```cuda
// Instead of:
cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
kernel<<<grid, block>>>(d_data, ...);
cudaMemcpy(h_result, d_result, size, cudaMemcpyDeviceToHost);

// Use:
cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);

// Overlap transfers and computation
cudaMemcpyAsync(d_data1, h_data1, size/2, cudaMemcpyHostToDevice, stream1);
cudaMemcpyAsync(d_data2, h_data2, size/2, cudaMemcpyHostToDevice, stream2);
kernel<<<grid, block, 0, stream1>>>(d_data1, ...);
kernel<<<grid, block, 0, stream2>>>(d_data2, ...);
cudaMemcpyAsync(h_result1, d_result1, size/2, cudaMemcpyDeviceToHost, stream1);
cudaMemcpyAsync(h_result2, d_result2, size/2, cudaMemcpyDeviceToHost, stream2);

// Only synchronize when necessary
cudaStreamSynchronize(stream1);
cudaStreamSynchronize(stream2);
```

**Expected Impact**: 15-25% overall performance improvement by overlapping memory transfers with computation.

### 2. Implement Memory Pooling

**Description**: Replace frequent cudaMalloc/cudaFree calls with a memory pool that reuses previously allocated memory.

**Implementation**:
```cuda
// Simple memory pool implementation
class CudaMemoryPool {
private:
    std::map<size_t, std::vector<void*>> free_memory;
    
public:
    void* allocate(size_t size) {
        if (!free_memory[size].empty()) {
            void* ptr = free_memory[size].back();
            free_memory[size].pop_back();
            return ptr;
        }
        void* ptr;
        cudaMalloc(&ptr, size);
        return ptr;
    }
    
    void deallocate(void* ptr, size_t size) {
        free_memory[size].push_back(ptr);
    }
    
    void release() {
        for (auto& pair : free_memory) {
            for (void* ptr : pair.second) {
                cudaFree(ptr);
            }
        }
        free_memory.clear();
    }
};
```

**Expected Impact**: 5-10% performance improvement by reducing memory allocation overhead.

### 3. Use Pinned Host Memory for Frequent Transfers

**Description**: Allocate host memory as pinned (page-locked) for data that is frequently transferred between host and device.

**Implementation**:
```cuda
// Instead of:
float* h_data = (float*)malloc(size);

// Use:
float* h_data;
cudaHostAlloc((void**)&h_data, size, cudaHostAllocDefault);

// When done:
cudaFreeHost(h_data);
```

**Expected Impact**: 20-30% faster memory transfers, resulting in 5-15% overall performance improvement.

## Medium-Priority Optimizations

### 4. Optimize Kernel Launch Configurations

**Description**: Adjust grid and block dimensions to maximize GPU occupancy and efficiency.

**Implementation**:
```cuda
// Instead of fixed dimensions:
kernel<<<gridSize, blockSize>>>(args...);

// Calculate optimal dimensions:
int blockSize = 256; // Typically 256 or 512 for compute-bound kernels
int minGridSize;
int optimalBlockSize;
cudaOccupancyMaxPotentialBlockSize(&minGridSize, &optimalBlockSize, kernel, 0, 0);
int gridSize = (n + optimalBlockSize - 1) / optimalBlockSize;

kernel<<<gridSize, optimalBlockSize>>>(args...);
```

**Expected Impact**: 5-10% performance improvement for compute-bound kernels.

### 5. Implement Kernel Fusion

**Description**: Combine sequential kernels that operate on the same data to reduce kernel launch overhead and improve data locality.

**Implementation**:
```cuda
// Instead of:
kernelA<<<grid, block>>>(d_data, ...);
kernelB<<<grid, block>>>(d_data, ...);

// Create a fused kernel:
__global__ void fusedKernel(float* data, ...) {
    // Code from kernelA
    ...
    
    // Ensure all threads complete kernelA operations
    __syncthreads();
    
    // Code from kernelB
    ...
}

fusedKernel<<<grid, block>>>(d_data, ...);
```

**Expected Impact**: 3-8% performance improvement by reducing kernel launch overhead and improving data locality.

## Lower-Priority Optimizations

### 6. Implement Persistent Threads for Iterative Algorithms

**Description**: Use persistent threads to process multiple iterations within a single kernel launch for iterative algorithms.

**Implementation**:
```cuda
__global__ void persistentKernel(float* data, int numIterations, ...) {
    // Thread setup
    ...
    
    for (int iter = 0; iter < numIterations; iter++) {
        // Process one iteration
        ...
        
        // Synchronize threads between iterations
        __syncthreads();
    }
}
```

**Expected Impact**: 2-5% performance improvement for highly iterative algorithms.

### 7. Optimize Memory Access Patterns

**Description**: Restructure data layout and access patterns to improve memory coalescing and reduce bank conflicts.

**Implementation**:
```cuda
// For 2D data, consider using pitched memory:
cudaPitchedPtr d_pitchedData;
cudaMalloc3D(&d_pitchedData, make_cudaExtent(width * sizeof(float), height, 1));

// Access with stride awareness:
__global__ void optimizedKernel(cudaPitchedPtr data, ...) {
    char* row = (char*)data.ptr + blockIdx.y * data.pitch;
    float* element = (float*)(row + threadIdx.x * sizeof(float));
    // Now element points to the correct position with proper alignment
}
```

**Expected Impact**: 3-7% performance improvement for memory-bound kernels.

## Architectural Recommendations

### 8. Consider Unified Memory for Complex Data Structures

**Description**: For applications with complex data structures and irregular access patterns, consider using CUDA Unified Memory to simplify memory management.

**Implementation**:
```cuda
// Instead of explicit memory management:
MyStruct* h_data = new MyStruct[n];
MyStruct* d_data;
cudaMalloc(&d_data, n * sizeof(MyStruct));
cudaMemcpy(d_data, h_data, n * sizeof(MyStruct), cudaMemcpyHostToDevice);

// Use unified memory:
MyStruct* data;
cudaMallocManaged(&data, n * sizeof(MyStruct));
// Now 'data' can be accessed from both host and device code
```

**Expected Impact**: Simplified code and potentially improved performance for certain access patterns, though may not be faster in all cases.

### 9. Evaluate Multi-GPU Parallelism

**Description**: For large-scale computations, consider distributing work across multiple GPUs.

**Implementation**:
```cuda
int numGpus;
cudaGetDeviceCount(&numGpus);

for (int i = 0; i < numGpus; i++) {
    cudaSetDevice(i);
    // Allocate device memory for this GPU
    cudaMalloc(&d_data[i], size / numGpus);
    // Launch kernels on this GPU
    kernel<<<grid, block>>>(d_data[i], ...);
}

// Synchronize and gather results
for (int i = 0; i < numGpus; i++) {
    cudaSetDevice(i);
    cudaDeviceSynchronize();
    // Copy results back
}
```

**Expected Impact**: Near-linear speedup with the number of GPUs for compute-bound applications with minimal inter-GPU communication.

## Conclusion

By implementing these optimizations, particularly the high-priority ones, this CUDA application could achieve a **30-45% overall performance improvement**. The most significant gains would come from better memory management and increased parallelism through asynchronous operations and CUDA streams.

The recommendations are designed to be implemented incrementally, allowing for validation of performance improvements at each step. Start with the high-priority optimizations for the best return on implementation effort.
"""
    
    def _generate_combined_analysis(self):
        """Generate a combined analysis from all section analyses"""
        if not self.section_analyses:
            print("Error: No section analyses available to combine.")
            return None
        
        print("Generating combined analysis...")
        
        # Create the combined analysis
        combined_analysis = "# CUDA Trace Analysis - Combined Report\n\n"
        
        # Add a table of contents
        combined_analysis += "## Table of Contents\n\n"
        for section in self.section_analyses.keys():
            section_title = section.replace('_', ' ').title()
            combined_analysis += f"- [{section_title}](#{section.lower().replace('_', '-')})\n"
        combined_analysis += "\n"
        
        # Add each section
        for section, analysis in self.section_analyses.items():
            section_title = section.replace('_', ' ').title()
            combined_analysis += f"## {section_title}\n\n"
            
            # Remove the first heading from the section analysis (if it exists)
            # to avoid duplicate headings
            section_content = analysis
            if section_content.startswith('#'):
                section_content = '\n'.join(section_content.split('\n')[1:])
            
            combined_analysis += section_content + "\n\n"
        
        # Add a timestamp
        combined_analysis += f"\n\n---\n\nGenerated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        
        # Save the combined analysis
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(self.output_dir, f"llm_analysis_combined_{timestamp}.md")
        with open(output_path, "w") as f:
            f.write(combined_analysis)
        
        print(f"Combined analysis saved to {output_path}")
        
        return combined_analysis
    
    def _generate_html_report(self, tables, encoded_images):
        """Generate an HTML report with integrated visualizations and analysis"""
        if not self.section_analyses:
            print("Error: No section analyses available for HTML report.")
            return None
        
        print("Generating HTML report...")
        
        # Create a basic HTML template
        html_template = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>CUDA Trace Analysis Report</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                }
                h1, h2, h3, h4 {
                    color: #2c3e50;
                }
                .dashboard {
                    width: 100%;
                    max-width: 1100px;
                    margin: 20px 0;
                    border: 1px solid #ddd;
                }
                .visualization {
                    max-width: 100%;
                    margin: 20px 0;
                    border: 1px solid #ddd;
                }
                .section {
                    margin-bottom: 40px;
                    border-bottom: 1px solid #eee;
                    padding-bottom: 20px;
                }
                .data-table {
                    border-collapse: collapse;
                    width: 100%;
                    margin: 20px 0;
                }
                .data-table th, .data-table td {
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }
                .data-table th {
                    background-color: #f2f2f2;
                }
                .data-table tr:nth-child(even) {
                    background-color: #f9f9f9;
                }
                .flex-container {
                    display: flex;
                    flex-wrap: wrap;
                    gap: 20px;
                }
                .flex-item {
                    flex: 1;
                    min-width: 300px;
                }
                .toc {
                    background-color: #f8f9fa;
                    padding: 15px;
                    border-radius: 5px;
                    margin-bottom: 30px;
                }
                .toc ul {
                    list-style-type: none;
                    padding-left: 20px;
                }
                .toc li {
                    margin-bottom: 5px;
                }
                .timestamp {
                    font-size: 0.8em;
                    color: #777;
                    margin-top: 50px;
                    text-align: center;
                }
                .toggle-button {
                    background-color: #4CAF50;
                    color: white;
                    padding: 10px 15px;
                    border: none;
                    border-radius: 4px;
                    cursor: pointer;
                    margin-bottom: 10px;
                }
                .toggle-content {
                    display: none;
                    padding: 10px;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                    background-color: #f9f9f9;
                }
            </style>
            <script>
                function toggleContent(id) {
                    var content = document.getElementById(id);
                    if (content.style.display === "block") {
                        content.style.display = "none";
                    } else {
                        content.style.display = "block";
                    }
                }
            </script>
        </head>
        <body>
            <h1>CUDA Trace Analysis Report</h1>
            
            <div class="toc">
                <h2>Table of Contents</h2>
                <ul>
                    <li><a href="#overview">Overview</a></li>
                    <li><a href="#api-distribution">API Distribution</a></li>
                    <li><a href="#memory-operations">Memory Operations</a></li>
                    <li><a href="#kernel-launches">Kernel Launches</a></li>
                    <li><a href="#performance-bottlenecks">Performance Bottlenecks</a></li>
                    <li><a href="#optimization-recommendations">Optimization Recommendations</a></li>
                </ul>
            </div>
            
            <!-- Dashboard Section -->
            <div class="section">
                <h2>CUDA Trace Dashboard</h2>
                {% if dashboard_image %}
                <img src="data:image/png;base64,{{ dashboard_image }}" alt="CUDA Trace Dashboard" class="dashboard">
                {% else %}
                <p>Dashboard visualization not available.</p>
                {% endif %}
            </div>
            
            <!-- Overview Section -->
            <div id="overview" class="section">
                <h2>Overview</h2>
                <div class="flex-container">
                    <div class="flex-item">
                        {{ overview_analysis|safe }}
                    </div>
                    <div class="flex-item">
                        {% if dashboard_image %}
                        <img src="data:image/png;base64,{{ dashboard_image }}" alt="CUDA Trace Dashboard" class="visualization">
                        {% endif %}
                    </div>
                </div>
            </div>
            
            <!-- API Distribution Section -->
            <div id="api-distribution" class="section">
                <h2>API Distribution</h2>
                <div class="flex-container">
                    <div class="flex-item">
                        {{ api_distribution_analysis|safe }}
                        
                        {% if api_distribution_table %}
                        <button class="toggle-button" onclick="toggleContent('api-table')">Show/Hide API Distribution Table</button>
                        <div id="api-table" class="toggle-content">
                            {{ api_distribution_table|safe }}
                        </div>
                        {% endif %}
                    </div>
                    <div class="flex-item">
                        {% if api_distribution_image %}
                        <img src="data:image/png;base64,{{ api_distribution_image }}" alt="API Distribution" class="visualization">
                        {% endif %}
                    </div>
                </div>
            </div>
            
            <!-- Memory Operations Section -->
            <div id="memory-operations" class="section">
                <h2>Memory Operations</h2>
                <div class="flex-container">
                    <div class="flex-item">
                        {{ memory_operations_analysis|safe }}
                        
                        {% if memory_operations_table %}
                        <button class="toggle-button" onclick="toggleContent('memory-table')">Show/Hide Memory Operations Table</button>
                        <div id="memory-table" class="toggle-content">
                            {{ memory_operations_table|safe }}
                        </div>
                        {% endif %}
                    </div>
                    <div class="flex-item">
                        {% if memory_operations_image %}
                        <img src="data:image/png;base64,{{ memory_operations_image }}" alt="Memory Operations" class="visualization">
                        {% endif %}
                    </div>
                </div>
            </div>
            
            <!-- Kernel Launches Section -->
            <div id="kernel-launches" class="section">
                <h2>Kernel Launches</h2>
                <div class="flex-container">
                    <div class="flex-item">
                        {{ kernel_launches_analysis|safe }}
                        
                        {% if kernel_distribution_table %}
                        <button class="toggle-button" onclick="toggleContent('kernel-table')">Show/Hide Kernel Distribution Table</button>
                        <div id="kernel-table" class="toggle-content">
                            {{ kernel_distribution_table|safe }}
                        </div>
                        {% endif %}
                        
                        {% if grid_block_dimensions_table %}
                        <button class="toggle-button" onclick="toggleContent('grid-block-table')">Show/Hide Grid/Block Dimensions Table</button>
                        <div id="grid-block-table" class="toggle-content">
                            {{ grid_block_dimensions_table|safe }}
                        </div>
                        {% endif %}
                    </div>
                    <div class="flex-item">
                        {% if kernel_distribution_image %}
                        <img src="data:image/png;base64,{{ kernel_distribution_image }}" alt="Kernel Distribution" class="visualization">
                        {% endif %}
                    </div>
                </div>
            </div>
            
            <!-- Performance Bottlenecks Section -->
            <div id="performance-bottlenecks" class="section">
                <h2>Performance Bottlenecks</h2>
                <div class="flex-container">
                    <div class="flex-item">
                        {{ performance_bottlenecks_analysis|safe }}
                        
                        {% if performance_bottlenecks_table %}
                        <button class="toggle-button" onclick="toggleContent('bottlenecks-table')">Show/Hide Performance Bottlenecks Table</button>
                        <div id="bottlenecks-table" class="toggle-content">
                            {{ performance_bottlenecks_table|safe }}
                        </div>
                        {% endif %}
                    </div>
                    <div class="flex-item">
                        {% if timeline_image %}
                        <img src="data:image/png;base64,{{ timeline_image }}" alt="API Call Timeline" class="visualization">
                        {% endif %}
                    </div>
                </div>
            </div>
            
            <!-- Optimization Recommendations Section -->
            <div id="optimization-recommendations" class="section">
                <h2>Optimization Recommendations</h2>
                {{ optimization_recommendations_analysis|safe }}
            </div>
            
            <div class="timestamp">
                Generated on: {{ timestamp }}
            </div>
        </body>
        </html>
        """
        
        # Prepare the template context
        context = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'dashboard_image': encoded_images.get('dashboard', ''),
            'api_distribution_image': encoded_images.get('api_distribution', ''),
            'memory_operations_image': encoded_images.get('memory_operations', ''),
            'kernel_distribution_image': encoded_images.get('kernel_distribution', ''),
            'timeline_image': encoded_images.get('timeline', ''),
            'overview_analysis': self._markdown_to_html(self.section_analyses.get('overview', 'Analysis not available.')),
            'api_distribution_analysis': self._markdown_to_html(self.section_analyses.get('api_distribution', 'Analysis not available.')),
            'memory_operations_analysis': self._markdown_to_html(self.section_analyses.get('memory_operations', 'Analysis not available.')),
            'kernel_launches_analysis': self._markdown_to_html(self.section_analyses.get('kernel_launches', 'Analysis not available.')),
            'performance_bottlenecks_analysis': self._markdown_to_html(self.section_analyses.get('performance_bottlenecks', 'Analysis not available.')),
            'optimization_recommendations_analysis': self._markdown_to_html(self.section_analyses.get('optimization_recommendations', 'Analysis not available.')),
            'api_distribution_table': tables.get('api_distribution', {}).get('html', ''),
            'memory_operations_table': tables.get('memory_operations', {}).get('html', ''),
            'kernel_distribution_table': tables.get('kernel_distribution', {}).get('html', ''),
            'grid_block_dimensions_table': tables.get('grid_block_dimensions', {}).get('html', ''),
            'performance_bottlenecks_table': tables.get('performance_bottlenecks', {}).get('html', '')
        }
        
        # Render the template
        try:
            from jinja2 import Template
            template = Template(html_template)
            html_report = template.render(**context)
        except Exception as e:
            print(f"Error rendering HTML template: {e}")
            # Fallback to simple string replacement
            html_report = html_template
            for key, value in context.items():
                placeholder = '{{ ' + key + '|safe }}' if '|safe' in html_template else '{{ ' + key + ' }}'
                html_report = html_report.replace(placeholder, str(value))
        
        # Save the HTML report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(self.html_output_dir, f"cuda_trace_analysis_report_{timestamp}.html")
        with open(output_path, "w") as f:
            f.write(html_report)
        
        print(f"HTML report saved to {output_path}")
        
        return output_path
    
    def _markdown_to_html(self, markdown_text):
        """Convert markdown text to HTML"""
        try:
            import markdown
            return markdown.markdown(markdown_text)
        except ImportError:
            # Simple fallback if markdown module is not available
            html = markdown_text
            # Convert headers
            for i in range(6, 0, -1):
                pattern = '#' * i + ' '
                html = html.replace(pattern, f'<h{i}>')
                # Close the tag at the end of the line
                lines = []
                for line in html.split('\n'):
                    if line.startswith(f'<h{i}>'):
                        line = line + f'</h{i}>'
                    lines.append(line)
                html = '\n'.join(lines)
            
            # Convert bold
            html = html.replace('**', '<strong>')
            # Convert italic
            html = html.replace('*', '<em>')
            # Convert paragraphs
            html = '<p>' + html.replace('\n\n', '</p><p>') + '</p>'
            # Convert line breaks
            html = html.replace('\n', '<br>')
            
            return html

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced CUDA Trace LLM Analyzer")
    parser.add_argument("analysis_dir", help="Path to the analysis results directory")
    parser.add_argument("--enhanced_dir", help="Path to enhanced visualizations directory")
    parser.add_argument("--llm_mode", choices=["mock", "openai", "local"], default="mock", help="LLM analysis mode")
    parser.add_argument("--api_key", help="OpenAI API key (for openai mode)")
    parser.add_argument("--model_endpoint", default="http://localhost:8000/v1/chat/completions", help="Local LLM API endpoint (for local mode)")
    
    args = parser.parse_args()
    
    analyzer = EnhancedCUDATraceLLMAnalyzer(args.analysis_dir, args.enhanced_dir)
    
    if args.llm_mode == "openai":
        analyzer.analyze_with_openai(args.api_key)
    elif args.llm_mode == "local":
        analyzer.analyze_with_local_llm(args.model_endpoint)
    else:  # mock mode
        analyzer.generate_mock_analysis()
