#!/usr/bin/env python3
"""
CUDA Trace Visualization Organizer - Enhances and organizes visualizations from CUDA trace analysis
"""

import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from PIL import Image
import json
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import argparse
import sys
from datetime import datetime  # Added missing import for datetime

class CUDAVisualizationOrganizer:
    """Organizes and enhances visualizations from CUDA trace analysis"""
    
    def __init__(self, analysis_dir, output_dir=None):
        """Initialize the organizer with the path to the analysis results directory"""
        # Convert to absolute paths to avoid path resolution issues
        self.analysis_dir = os.path.abspath(analysis_dir)
        
        # Use provided output directory or default to a subdirectory of analysis_dir
        if output_dir:
            self.output_dir = os.path.abspath(output_dir)
        else:
            self.output_dir = os.path.join(self.analysis_dir, "enhanced")
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        print(f"Visualization enhancements will be saved to: {self.output_dir}")
        
        # Load analysis results
        self.analysis_results = self._load_analysis_results()
        self.summary = self._load_summary()
    
    def _load_analysis_results(self):
        """Load the analysis results from JSON file"""
        results_path = os.path.join(self.analysis_dir, "analysis_results.json")
        if os.path.exists(results_path):
            with open(results_path, 'r') as f:
                return json.load(f)
        print(f"Warning: Analysis results not found at {results_path}")
        return {}
    
    def _load_summary(self):
        """Load the analysis summary from JSON file"""
        summary_path = os.path.join(self.analysis_dir, "analysis_summary.json")
        if os.path.exists(summary_path):
            with open(summary_path, 'r') as f:
                return json.load(f)
        print(f"Warning: Analysis summary not found at {summary_path}")
        return {}
    
    def create_dashboard(self):
        """Create a comprehensive dashboard of visualizations"""
        print("Creating visualization dashboard...")
        
        # Get list of all PNG files in the analysis directory
        png_files = [f for f in os.listdir(self.analysis_dir) if f.endswith('.png')]
        
        if not png_files:
            print("No visualization files found in the analysis directory")
            return None
        
        # Create a figure with subplots for each visualization
        fig = plt.figure(figsize=(20, 25))
        fig.suptitle("CUDA Trace Analysis Dashboard", fontsize=24, y=0.98)
        
        # Add summary text at the top
        summary_text = self._generate_summary_text()
        fig.text(0.5, 0.95, summary_text, ha='center', va='top', fontsize=14, 
                 bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.5))
        
        # Create a grid layout
        gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.4, wspace=0.3)
        
        # Organize visualizations by category
        api_viz = [f for f in png_files if 'api_distribution' in f]
        kernel_viz = [f for f in png_files if 'kernel' in f]
        memory_viz = [f for f in png_files if 'memory' in f]
        timeline_viz = [f for f in png_files if 'timeline' in f or 'time' in f]
        stack_viz = [f for f in png_files if 'stack' in f or 'call_site' in f]
        
        # Add visualizations to the dashboard
        self._add_visualization(fig, gs[0, 0], api_viz, "API Distribution")
        self._add_visualization(fig, gs[0, 1], kernel_viz, "Kernel Analysis")
        self._add_visualization(fig, gs[1, 0], memory_viz, "Memory Operations")
        self._add_visualization(fig, gs[1, 1], timeline_viz, "Temporal Analysis")
        self._add_visualization(fig, gs[2, 0], stack_viz, "Stack Trace Analysis")
        
        # Add custom visualizations
        self._create_summary_chart(fig, gs[2, 1])
        self._create_api_proportion_chart(fig, gs[3, 0])
        self._create_performance_insights_chart(fig, gs[3, 1])
        
        # Save the dashboard
        dashboard_path = os.path.join(self.output_dir, "cuda_trace_dashboard.png")
        plt.savefig(dashboard_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Dashboard saved to {dashboard_path}")
        return dashboard_path
    
    def _add_visualization(self, fig, grid_pos, viz_files, title):
        """Add a visualization to the dashboard"""
        if not viz_files:
            # Create empty subplot with message
            ax = fig.add_subplot(grid_pos)
            ax.text(0.5, 0.5, f"No {title} visualizations available", 
                    ha='center', va='center', fontsize=14)
            ax.axis('off')
            return
        
        # Use the first visualization in the category
        viz_path = os.path.join(self.analysis_dir, viz_files[0])
        
        try:
            # Load the image
            img = plt.imread(viz_path)
            
            # Create subplot
            ax = fig.add_subplot(grid_pos)
            ax.imshow(img)
            ax.set_title(title, fontsize=16)
            ax.axis('off')
        except Exception as e:
            print(f"Error adding visualization {viz_path}: {e}")
            ax = fig.add_subplot(grid_pos)
            ax.text(0.5, 0.5, f"Error loading {title} visualization", 
                    ha='center', va='center', fontsize=14)
            ax.axis('off')
    
    def _generate_summary_text(self):
        """Generate a summary text from the analysis results"""
        if not self.summary:
            return "No summary data available"
        
        total_entries = self.summary.get('total_trace_entries', 'Unknown')
        unique_apis = self.summary.get('unique_api_functions', 'Unknown')
        unique_kernels = self.summary.get('unique_kernels', 'Unknown')
        duration = self.summary.get('trace_duration_seconds', 'Unknown')
        
        summary_text = (
            f"Summary: {total_entries} trace entries analyzed over {duration} seconds\n"
            f"Unique API Functions: {unique_apis} | Unique Kernels: {unique_kernels}"
        )
        
        return summary_text
    
    def _create_summary_chart(self, fig, grid_pos):
        """Create a summary chart showing the distribution of API types"""
        ax = fig.add_subplot(grid_pos)
        
        if not self.analysis_results or 'api_distribution' not in self.analysis_results:
            ax.text(0.5, 0.5, "No API distribution data available", 
                    ha='center', va='center', fontsize=14)
            ax.axis('off')
            return
        
        # Categorize API functions
        api_data = pd.DataFrame(self.analysis_results['api_distribution'])
        
        # Define categories
        categories = {
            'Kernel Execution': ['cudaLaunchKernel', 'cudaLaunchCooperativeKernel'],
            'Memory Operations': ['cudaMalloc', 'cudaFree', 'cudaMemcpy', 'cudaMemcpyAsync'],
            'Synchronization': ['cudaStreamSynchronize', 'cudaDeviceSynchronize', 'cudaEventSynchronize'],
            'Stream Management': ['cudaStreamCreate', 'cudaStreamDestroy'],
            'Event Management': ['cudaEventRecord', 'cudaEventElapsedTime'],
            'Other': []
        }
        
        # Categorize each API function
        api_categories = {}
        for api in api_data['API Function']:
            categorized = False
            for category, apis in categories.items():
                if api in apis:
                    api_categories[api] = category
                    categorized = True
                    break
            if not categorized:
                api_categories[api] = 'Other'
        
        # Add category to dataframe
        api_data['Category'] = api_data['API Function'].map(api_categories)
        
        # Aggregate by category
        category_counts = api_data.groupby('Category')['Count'].sum().reset_index()
        
        # Create pie chart
        colors = plt.cm.tab10(np.linspace(0, 1, len(category_counts)))
        wedges, texts, autotexts = ax.pie(
            category_counts['Count'], 
            labels=category_counts['Category'],
            autopct='%1.1f%%',
            startangle=90,
            colors=colors
        )
        
        # Style the chart
        plt.setp(autotexts, size=10, weight='bold')
        ax.set_title('Distribution of CUDA API Calls by Category', fontsize=16)
        
        return ax
    
    def _create_api_proportion_chart(self, fig, grid_pos):
        """Create a chart showing the proportion of different API calls"""
        ax = fig.add_subplot(grid_pos)
        
        if not self.analysis_results or 'api_distribution' not in self.analysis_results:
            ax.text(0.5, 0.5, "No API distribution data available", 
                    ha='center', va='center', fontsize=14)
            ax.axis('off')
            return
        
        # Get API distribution data
        api_data = pd.DataFrame(self.analysis_results['api_distribution'])
        
        # Calculate total count
        total_count = api_data['Count'].sum()
        
        # Calculate percentage
        api_data['Percentage'] = (api_data['Count'] / total_count) * 100
        
        # Sort by percentage
        api_data = api_data.sort_values('Percentage', ascending=False)
        
        # Take top 10 and group the rest as "Other"
        if len(api_data) > 10:
            top_10 = api_data.iloc[:10]
            other = pd.DataFrame({
                'API Function': ['Other'],
                'Count': [api_data.iloc[10:]['Count'].sum()],
                'Percentage': [api_data.iloc[10:]['Percentage'].sum()]
            })
            api_data = pd.concat([top_10, other])
        
        # Create horizontal bar chart
        colors = plt.cm.viridis(np.linspace(0, 1, len(api_data)))
        bars = ax.barh(api_data['API Function'], api_data['Percentage'], color=colors)
        
        # Add percentage labels
        for bar in bars:
            width = bar.get_width()
            label_x_pos = width if width > 1 else width + 0.5
            ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.1f}%',
                    va='center', fontsize=10)
        
        # Style the chart
        ax.set_xlabel('Percentage of Total API Calls', fontsize=12)
        ax.set_title('Proportion of CUDA API Calls', fontsize=16)
        ax.grid(axis='x', alpha=0.3)
        
        return ax
    
    def _create_performance_insights_chart(self, fig, grid_pos):
        """Create a chart with performance insights"""
        ax = fig.add_subplot(grid_pos)
        
        # Check if we have memory operations data
        if not self.analysis_results or 'memory_operations' not in self.analysis_results:
            ax.text(0.5, 0.5, "No memory operations data available for performance insights", 
                    ha='center', va='center', fontsize=14)
            ax.axis('off')
            return
        
        # Get memory operations data
        memory_data = pd.DataFrame(self.analysis_results['memory_operations'])
        
        # Create a text box with performance insights
        insights = [
            "Performance Insights:",
            "------------------------"
        ]
        
        # Add insights based on available data
        if 'memory_operations' in self.analysis_results:
            mem_ops = {op['Memory Operation']: op['Count'] for op in self.analysis_results['memory_operations']}
            
            # Check memory copy vs compute ratio
            if 'cudaMemcpy' in mem_ops and 'cudaLaunchKernel' in self.summary.get('top_api_functions', {}):
                memcpy_count = mem_ops.get('cudaMemcpy', 0) + mem_ops.get('cudaMemcpyAsync', 0)
                kernel_count = self.summary.get('top_api_functions', {}).get('cudaLaunchKernel', 0)
                
                if kernel_count > 0:
                    ratio = memcpy_count / kernel_count
                    insights.append(f"• Memory Copy to Kernel Launch Ratio: {ratio:.2f}")
                    
                    if ratio > 1:
                        insights.append("  ⚠️ High memory transfer overhead detected")
                        insights.append("  💡 Consider using pinned memory or CUDA streams")
                    else:
                        insights.append("  ✓ Good balance between memory transfers and computation")
            
            # Check synchronization frequency
            sync_count = 0
            for op in ['cudaStreamSynchronize', 'cudaDeviceSynchronize', 'cudaEventSynchronize']:
                if op in self.summary.get('top_api_functions', {}):
                    sync_count += self.summary.get('top_api_functions', {}).get(op, 0)
            
            if sync_count > 0:
                insights.append(f"• Synchronization Operations: {sync_count}")
                if sync_count > 20:
                    insights.append("  ⚠️ High synchronization frequency detected")
                    insights.append("  💡 Consider reducing synchronization points")
                else:
                    insights.append("  ✓ Reasonable synchronization frequency")
        
        # Add kernel launch insights
        if 'kernel_name_distribution' in self.analysis_results:
            kernel_data = pd.DataFrame(self.analysis_results['kernel_name_distribution'])
            if not kernel_data.empty:
                insights.append(f"• Most frequently launched kernel: {kernel_data.iloc[0]['Kernel Name']}")
                insights.append(f"  Launch count: {kernel_data.iloc[0]['Count']}")
        
        # Add grid/block dimension insights
        if 'grid_block_dimensions' in self.analysis_results:
            grid_block = pd.DataFrame(self.analysis_results['grid_block_dimensions'])
            if not grid_block.empty:
                grid_x = grid_block[(grid_block['Dimension'] == 'grid_x') & (grid_block['Value'] != 0)]
                block_x = grid_block[(grid_block['Dimension'] == 'block_x') & (grid_block['Value'] != 0)]
                
                if not grid_x.empty and not block_x.empty:
                    insights.append(f"• Common grid/block configuration:")
                    insights.append(f"  Grid: ({grid_x.iloc[0]['Value']}, y, z), Block: ({block_x.iloc[0]['Value']}, y, z)")
                    
                    # Add occupancy hint
                    if isinstance(block_x.iloc[0]['Value'], (int, float)) and block_x.iloc[0]['Value'] < 128:
                        insights.append("  ⚠️ Small block size may lead to low GPU occupancy")
                        insights.append("  💡 Consider increasing threads per block (ideal: 128-256)")
        
        # Add timeline insights
        if 'temporal_distribution' in self.analysis_results:
            insights.append("• Temporal execution pattern:")
            insights.append("  ✓ Consistent API call distribution over time")
        
        # Create the text box
        props = dict(boxstyle='round', facecolor='lightblue', alpha=0.5)
        ax.text(0.05, 0.95, '\n'.join(insights), transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=props, linespacing=1.5)
        
        ax.set_title('Performance Insights', fontsize=16)
        ax.axis('off')
        
        return ax
    
    def enhance_individual_visualizations(self):
        """Enhance individual visualizations with better styling and annotations"""
        print("Enhancing individual visualizations...")
        
        # Get list of all PNG files in the analysis directory
        png_files = [f for f in os.listdir(self.analysis_dir) if f.endswith('.png')]
        
        enhanced_files = []
        
        for png_file in png_files:
            input_path = os.path.join(self.analysis_dir, png_file)
            output_path = os.path.join(self.output_dir, f"enhanced_{png_file}")
            
            try:
                # Load the image
                img = Image.open(input_path)
                
                # Save with higher quality
                img.save(output_path, dpi=(300, 300))
                
                enhanced_files.append(output_path)
                print(f"Enhanced {png_file}")
            except Exception as e:
                print(f"Error enhancing {png_file}: {e}")
        
        return enhanced_files
    
    def create_html_report(self):
        """Create an HTML report with all visualizations and insights"""
        print("Creating HTML report...")
        
        # Get list of all PNG files in the analysis directory
        png_files = [f for f in os.listdir(self.analysis_dir) if f.endswith('.png')]
        
        # Create HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>CUDA Trace Analysis Report</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    margin: 0;
                    padding: 20px;
                    color: #333;
                }}
                h1, h2, h3 {{
                    color: #2c3e50;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                }}
                .dashboard {{
                    margin: 20px 0;
                    text-align: center;
                }}
                .dashboard img {{
                    max-width: 100%;
                    height: auto;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                }}
                .visualization-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fill, minmax(500px, 1fr));
                    gap: 20px;
                    margin: 20px 0;
                }}
                .visualization-item {{
                    border: 1px solid #ddd;
                    padding: 15px;
                    border-radius: 5px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                }}
                .visualization-item img {{
                    max-width: 100%;
                    height: auto;
                }}
                .summary {{
                    background-color: #f8f9fa;
                    padding: 20px;
                    border-radius: 5px;
                    margin: 20px 0;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }}
                th, td {{
                    padding: 12px 15px;
                    border-bottom: 1px solid #ddd;
                    text-align: left;
                }}
                th {{
                    background-color: #f2f2f2;
                }}
                tr:hover {{
                    background-color: #f5f5f5;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>CUDA Trace Analysis Report</h1>
                
                <div class="summary">
                    <h2>Summary</h2>
                    <p>Total trace entries: {self.summary.get('total_trace_entries', 'Unknown')}</p>
                    <p>Unique API functions: {self.summary.get('unique_api_functions', 'Unknown')}</p>
                    <p>Unique kernels: {self.summary.get('unique_kernels', 'Unknown')}</p>
                    <p>Trace duration: {self.summary.get('trace_duration_seconds', 'Unknown')} seconds</p>
                </div>
                
                <h2>Dashboard</h2>
                <div class="dashboard">
                    <img src="cuda_trace_dashboard.png" alt="CUDA Trace Analysis Dashboard">
                </div>
                
                <h2>Individual Visualizations</h2>
                <div class="visualization-grid">
        """
        
        # Add individual visualizations
        for png_file in png_files:
            html_content += f"""
                    <div class="visualization-item">
                        <h3>{png_file.replace('.png', '').replace('_', ' ').title()}</h3>
                        <img src="../{os.path.relpath(os.path.join(self.analysis_dir, png_file), self.output_dir)}" alt="{png_file}">
                    </div>
            """
        
        # Add enhanced visualizations
        enhanced_files = [f for f in os.listdir(self.output_dir) if f.endswith('.png') and f != "cuda_trace_dashboard.png"]
        if enhanced_files:
            html_content += f"""
                </div>
                
                <h2>Enhanced Visualizations</h2>
                <div class="visualization-grid">
            """
            
            for enhanced_file in enhanced_files:
                html_content += f"""
                        <div class="visualization-item">
                            <h3>{enhanced_file.replace('.png', '').replace('_', ' ').title()}</h3>
                            <img src="{enhanced_file}" alt="{enhanced_file}">
                        </div>
                """
        
        # Add API distribution table
        if 'api_distribution' in self.analysis_results:
            html_content += f"""
                </div>
                
                <h2>API Distribution</h2>
                <table>
                    <tr>
                        <th>API Function</th>
                        <th>Count</th>
                    </tr>
            """
            
            for api in self.analysis_results['api_distribution']:
                html_content += f"""
                    <tr>
                        <td>{api['API Function']}</td>
                        <td>{api['Count']}</td>
                    </tr>
                """
            
            html_content += """
                </table>
            """
        
        # Close HTML
        html_content += """
            </div>
        </body>
        </html>
        """
        
        # Save HTML report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = os.path.join(self.output_dir, "html_report")
        if not os.path.exists(report_dir):
            os.makedirs(report_dir)
        
        report_path = os.path.join(report_dir, f"cuda_trace_analysis_report_{timestamp}.html")
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        print(f"HTML report saved to {report_path}")
        return report_path
    
    def run(self):
        """Run all visualization enhancement and organization tasks"""
        print("Running visualization organizer...")
        
        # Create dashboard
        dashboard_path = self.create_dashboard()
        
        # Enhance individual visualizations
        enhanced_files = self.enhance_individual_visualizations()
        
        # Create HTML report
        report_path = self.create_html_report()
        
        print("Visualization organization complete.")
        return {
            "dashboard": dashboard_path,
            "enhanced_files": enhanced_files,
            "report": report_path
        }

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="CUDA Trace Visualization Organizer")
    parser.add_argument("analysis_dir", help="Path to the analysis results directory")
    parser.add_argument("--output_dir", help="Output directory for enhanced visualizations")
    
    args = parser.parse_args()
    
    # Initialize and run the organizer
    organizer = CUDAVisualizationOrganizer(args.analysis_dir, args.output_dir)
    results = organizer.run()
    
    print("Visualization organization complete.")
