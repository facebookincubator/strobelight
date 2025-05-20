#!/usr/bin/env python3
"""
CUDA Trace Analyzer - Analyzes CUDA trace data and generates insights
"""

import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import numpy as np
import os
import argparse
import sys
from datetime import datetime
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
import re

class CUDATraceAnalyzer:
    """Analyzer for CUDA trace data"""
    
    def __init__(self, data_path, output_dir=None):
        """Initialize the analyzer with the path to the parsed data"""
        self.data_path = data_path
        self.df = None
        self.analysis_results = {}
        
        # Use provided output directory or default to current directory
        if output_dir:
            self.output_dir = output_dir
        else:
            self.output_dir = os.path.join(os.path.dirname(os.path.abspath(data_path)), "analysis_results")
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        print(f"Analysis results will be saved to: {self.output_dir}")
    
    def load_data(self):
        """Load the parsed trace data"""
        print(f"Loading data from {self.data_path}")
        
        # Check if the data file exists
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Parsed trace data file not found: {self.data_path}")
        
        with open(self.data_path, 'r') as f:
            data = json.load(f)
        
        self.df = pd.DataFrame(data)
        
        # Convert timestamp to datetime
        if 'timestamp' in self.df.columns:
            self.df['datetime'] = pd.to_datetime(self.df['timestamp'])
            
            # Calculate relative time from the first timestamp
            first_time = self.df['datetime'].min()
            self.df['relative_time'] = (self.df['datetime'] - first_time).dt.total_seconds()
        
        # Clean up grid and block dimensions
        for col in ['grid_x', 'grid_y', 'grid_z', 'block_x', 'block_y', 'block_z']:
            if col in self.df.columns:
                # Convert string values to numeric where possible
                self.df[col] = pd.to_numeric(self.df[col], errors='ignore')
        
        # Extract kernel name from args if available
        if 'args' in self.df.columns:
            self.df['kernel_name'] = self.df['args'].apply(
                lambda x: re.match(r'^(\w+)', str(x)).group(1) if isinstance(x, str) and re.match(r'^(\w+)', str(x)) else None
            )
        
        print(f"Loaded {len(self.df)} trace entries")
        return self.df
    
    def analyze(self):
        """Perform comprehensive analysis on the trace data"""
        if self.df is None:
            self.load_data()
        
        print("Performing analysis on trace data...")
        
        # Basic statistics
        self.analyze_api_distribution()
        self.analyze_kernel_launches()
        self.analyze_temporal_patterns()
        self.analyze_memory_operations()
        self.analyze_stack_traces()
        
        print("Analysis complete")
        return self.analysis_results
    
    def analyze_api_distribution(self):
        """Analyze the distribution of CUDA API calls"""
        print("Analyzing CUDA API call distribution...")
        
        # Count occurrences of each CUDA API function
        api_counts = self.df['cuda_api_function'].value_counts().reset_index()
        api_counts.columns = ['API Function', 'Count']
        
        # Save to analysis results
        self.analysis_results['api_distribution'] = api_counts.to_dict('records')
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Count', y='API Function', data=api_counts.head(15), palette='viridis')
        plt.title('Top 15 CUDA API Functions by Frequency')
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/api_distribution.png", dpi=300)
        plt.close()
        
        return api_counts
    
    def analyze_kernel_launches(self):
        """Analyze kernel launch patterns"""
        print("Analyzing kernel launch patterns...")
        
        # Filter for kernel launches
        kernel_launches = self.df[self.df['cuda_api_function'] == 'cudaLaunchKernel'].copy()
        
        if len(kernel_launches) == 0:
            print("No kernel launches found in the trace data")
            return None
        
        # Analyze kernel names
        if 'kernel_name' in kernel_launches.columns:
            kernel_name_counts = kernel_launches['kernel_name'].value_counts().reset_index()
            kernel_name_counts.columns = ['Kernel Name', 'Count']
            self.analysis_results['kernel_name_distribution'] = kernel_name_counts.to_dict('records')
            
            # Create visualization
            plt.figure(figsize=(12, 8))
            sns.barplot(x='Count', y='Kernel Name', data=kernel_name_counts.head(15), palette='magma')
            plt.title('Kernel Launch Frequency by Kernel Name')
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/kernel_name_distribution.png", dpi=300)
            plt.close()
        
        # Analyze grid and block dimensions
        grid_block_dims = []
        for col in ['grid_x', 'grid_y', 'grid_z', 'block_x', 'block_y', 'block_z']:
            if col in kernel_launches.columns:
                # Get the most common values
                top_values = kernel_launches[col].value_counts().head(5).reset_index()
                top_values.columns = ['Value', 'Count']
                top_values['Dimension'] = col
                grid_block_dims.append(top_values)
        
        if grid_block_dims:
            grid_block_df = pd.concat(grid_block_dims)
            self.analysis_results['grid_block_dimensions'] = grid_block_df.to_dict('records')
            
            # Create visualization for grid dimensions
            plt.figure(figsize=(15, 10))
            grid_dims = grid_block_df[grid_block_df['Dimension'].str.startswith('grid_')]
            if not grid_dims.empty:
                sns.barplot(x='Value', y='Count', hue='Dimension', data=grid_dims, palette='cool')
                plt.title('Distribution of Grid Dimensions in Kernel Launches')
                plt.tight_layout()
                plt.savefig(f"{self.output_dir}/grid_dimensions.png", dpi=300)
                plt.close()
            
            # Create visualization for block dimensions
            plt.figure(figsize=(15, 10))
            block_dims = grid_block_df[grid_block_df['Dimension'].str.startswith('block_')]
            if not block_dims.empty:
                sns.barplot(x='Value', y='Count', hue='Dimension', data=block_dims, palette='plasma')
                plt.title('Distribution of Block Dimensions in Kernel Launches')
                plt.tight_layout()
                plt.savefig(f"{self.output_dir}/block_dimensions.png", dpi=300)
                plt.close()
        
        return kernel_launches
    
    def analyze_temporal_patterns(self):
        """Analyze temporal patterns in the trace data"""
        print("Analyzing temporal patterns...")
        
        if 'relative_time' not in self.df.columns:
            print("No temporal data available for analysis")
            return None
        
        # Create a timeline of API calls
        timeline_data = self.df.copy()
        
        # Group by timestamp and count API calls
        timeline = timeline_data.groupby('relative_time')['cuda_api_function'].count().reset_index()
        timeline.columns = ['Relative Time (s)', 'API Call Count']
        
        self.analysis_results['temporal_distribution'] = timeline.to_dict('records')
        
        # Create visualization
        plt.figure(figsize=(15, 8))
        plt.plot(timeline['Relative Time (s)'], timeline['API Call Count'], marker='o', linestyle='-', color='blue')
        plt.title('CUDA API Call Frequency Over Time')
        plt.xlabel('Relative Time (seconds)')
        plt.ylabel('Number of API Calls')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/api_call_timeline.png", dpi=300)
        plt.close()
        
        # Analyze API call distribution over time
        if len(timeline_data) > 0:
            # Get top 5 API functions
            top_apis = timeline_data['cuda_api_function'].value_counts().head(5).index.tolist()
            
            # Filter for top APIs
            top_api_data = timeline_data[timeline_data['cuda_api_function'].isin(top_apis)]
            
            # Create a pivot table for the heatmap
            api_time_pivot = pd.pivot_table(
                top_api_data,
                values='pid',  # Just need a column to count
                index='cuda_api_function',
                columns=pd.cut(top_api_data['relative_time'], bins=10),
                aggfunc='count',
                fill_value=0
            )
            
            # Create visualization
            plt.figure(figsize=(15, 8))
            sns.heatmap(api_time_pivot, cmap='YlGnBu', annot=True, fmt='g')
            plt.title('Distribution of Top 5 CUDA API Calls Over Time')
            plt.xlabel('Relative Time Bins')
            plt.ylabel('CUDA API Function')
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/api_time_heatmap.png", dpi=300)
            plt.close()
        
        return timeline
    
    def analyze_memory_operations(self):
        """Analyze memory-related operations"""
        print("Analyzing memory operations...")
        
        # Filter for memory-related API calls
        memory_ops = self.df[
            self.df['cuda_api_function'].str.contains('cudaMalloc|cudaFree|cudaMemcpy|cudaMemcpyAsync', na=False)
        ].copy()
        
        if len(memory_ops) == 0:
            print("No memory operations found in the trace data")
            return None
        
        # Count by operation type
        memory_op_counts = memory_ops['cuda_api_function'].value_counts().reset_index()
        memory_op_counts.columns = ['Memory Operation', 'Count']
        
        self.analysis_results['memory_operations'] = memory_op_counts.to_dict('records')
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Count', y='Memory Operation', data=memory_op_counts, palette='Oranges_r')
        plt.title('Distribution of CUDA Memory Operations')
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/memory_operations.png", dpi=300)
        plt.close()
        
        # Analyze memory operations over time if temporal data is available
        if 'relative_time' in memory_ops.columns:
            plt.figure(figsize=(15, 8))
            
            # Group by time and operation type
            memory_timeline = memory_ops.groupby(['relative_time', 'cuda_api_function']).size().reset_index()
            memory_timeline.columns = ['Relative Time (s)', 'Memory Operation', 'Count']
            
            # Plot
            for op in memory_op_counts['Memory Operation'].unique():
                op_data = memory_timeline[memory_timeline['Memory Operation'] == op]
                plt.plot(op_data['Relative Time (s)'], op_data['Count'], marker='o', linestyle='-', label=op)
            
            plt.title('Memory Operations Over Time')
            plt.xlabel('Relative Time (seconds)')
            plt.ylabel('Operation Count')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/memory_operations_timeline.png", dpi=300)
            plt.close()
        
        return memory_op_counts
    
    def analyze_stack_traces(self):
        """Analyze stack trace patterns"""
        print("Analyzing stack trace patterns...")
        
        # Extract call sites from stack traces
        if 'stack_trace' not in self.df.columns:
            print("No stack trace data available for analysis")
            return None
        
        # Get the second frame in each stack trace (the caller of the CUDA API)
        call_sites = []
        for stack in self.df['stack_trace']:
            if isinstance(stack, list) and len(stack) > 1:
                call_sites.append(stack[1])
            else:
                call_sites.append('Unknown')
        
        self.df['call_site'] = call_sites
        
        # Count occurrences of each call site
        call_site_counts = self.df['call_site'].value_counts().reset_index()
        call_site_counts.columns = ['Call Site', 'Count']
        
        self.analysis_results['call_site_distribution'] = call_site_counts.to_dict('records')
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Count', y='Call Site', data=call_site_counts.head(15), palette='rocket')
        plt.title('Top 15 Call Sites for CUDA API Functions')
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/call_site_distribution.png", dpi=300)
        plt.close()
        
        # Analyze relationship between call sites and API functions
        call_site_api = self.df.groupby(['call_site', 'cuda_api_function']).size().reset_index()
        call_site_api.columns = ['Call Site', 'CUDA API Function', 'Count']
        call_site_api = call_site_api.sort_values('Count', ascending=False).head(20)
        
        self.analysis_results['call_site_api_relationship'] = call_site_api.to_dict('records')
        
        # Create visualization
        plt.figure(figsize=(15, 10))
        pivot_table = call_site_api.pivot_table(
            values='Count', 
            index='Call Site', 
            columns='CUDA API Function', 
            fill_value=0
        )
        sns.heatmap(pivot_table, cmap='viridis', annot=True, fmt='g')
        plt.title('Relationship Between Call Sites and CUDA API Functions')
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/call_site_api_heatmap.png", dpi=300)
        plt.close()
        
        return call_site_counts
    
    def generate_summary(self):
        """Generate a summary of the analysis results"""
        print("Generating analysis summary...")
        
        summary = {
            "total_trace_entries": len(self.df),
            "unique_api_functions": self.df['cuda_api_function'].nunique(),
            "top_api_functions": self.df['cuda_api_function'].value_counts().head(5).to_dict(),
        }
        
        if 'kernel_name' in self.df.columns:
            kernel_names = self.df['kernel_name'].dropna().unique()
            summary["unique_kernels"] = len(kernel_names)
            summary["kernel_names"] = kernel_names.tolist()
        
        if 'relative_time' in self.df.columns:
            summary["trace_duration_seconds"] = self.df['relative_time'].max()
        
        self.analysis_results['summary'] = summary
        
        # Save summary to file
        with open(f"{self.output_dir}/analysis_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary
    
    def save_results(self):
        """Save all analysis results to a JSON file"""
        print("Saving analysis results...")
        
        # Generate summary if not already done
        if 'summary' not in self.analysis_results:
            self.generate_summary()
        
        # Save to file
        with open(f"{self.output_dir}/analysis_results.json", 'w') as f:
            json.dump(self.analysis_results, f, indent=2, default=str)
        
        print(f"Analysis results saved to {self.output_dir}/analysis_results.json")
        
        # Create a list of all generated files
        generated_files = [
            f"{self.output_dir}/analysis_results.json",
            f"{self.output_dir}/analysis_summary.json"
        ]
        
        for file in os.listdir(self.output_dir):
            if file.endswith('.png'):
                generated_files.append(f"{self.output_dir}/{file}")
        
        return generated_files

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="CUDA Trace Analyzer")
    parser.add_argument("data_path", help="Path to the parsed trace data JSON file")
    parser.add_argument("--output_dir", help="Output directory for analysis results")
    
    args = parser.parse_args()
    
    # Initialize and run the analyzer
    analyzer = CUDATraceAnalyzer(args.data_path, args.output_dir)
    analyzer.load_data()
    analyzer.analyze()
    analyzer.generate_summary()
    analyzer.save_results()
    
    print("Analysis complete. Results saved to output directory.")
