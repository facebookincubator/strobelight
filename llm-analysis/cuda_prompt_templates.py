#!/usr/bin/env python3
"""
CUDA Trace LLM Prompt Templates - Enhanced prompt templates for LLM analysis of CUDA trace data
"""

import json
import os

class CUDAPromptTemplates:
    """Provides enhanced prompt templates for LLM analysis of CUDA trace data"""
    
    def __init__(self):
        """Initialize the prompt templates"""
        self.templates = self._load_default_templates()
    
    def _load_default_templates(self):
        """Load the default prompt templates"""
        return {
            "overview": self._get_overview_template(),
            "api_distribution": self._get_api_distribution_template(),
            "memory_operations": self._get_memory_operations_template(),
            "kernel_launches": self._get_kernel_launches_template(),
            "performance_bottlenecks": self._get_performance_bottlenecks_template(),
            "optimization_recommendations": self._get_optimization_recommendations_template()
        }
    
    def _get_overview_template(self):
        """Get the template for the overview section"""
        return """
        # CUDA Trace Analysis - Overview

        I need a comprehensive overview of the CUDA trace data provided. The data comes from a BPF program that traces various CUDA API routines.
        
        ## Analysis Task
        
        Please provide a high-level executive summary of the CUDA trace data that includes:
        
        1. The main characteristics of the application based on its CUDA API usage
        2. The most significant patterns observed in the trace data
        3. The key performance considerations identified
        4. A brief assessment of the application's CUDA implementation quality
        
        ## Available Data
        
        ### Summary Statistics
        
        - Total trace entries: {total_trace_entries}
        - Unique API functions: {unique_api_functions}
        - Unique kernels: {unique_kernels}
        - Trace duration: {trace_duration_seconds} seconds
        
        ### Dashboard Visualization
        
        The dashboard visualization (attached as an image) provides an overview of all aspects of the CUDA trace analysis, including API distribution, kernel launches, memory operations, and performance insights.
        
        ## Output Format
        
        Please structure your response as follows:
        
        1. **Application Characteristics**: A paragraph describing the main characteristics of the application based on its CUDA API usage.
        2. **Key Patterns Observed**: A list of the most significant patterns observed in the trace data.
        3. **Performance Considerations**: A list of key performance considerations identified from the trace data.
        4. **Implementation Quality Assessment**: A paragraph assessing the quality of the CUDA implementation.
        
        Focus on providing a concise but comprehensive overview that highlights the most important aspects of the trace data. This overview will serve as an executive summary for the detailed analysis that follows.
        """
    
    def _get_api_distribution_template(self):
        """Get the template for the API distribution section"""
        return """
        # CUDA Trace Analysis - API Distribution

        I need a detailed analysis of the CUDA API distribution in the trace data.
        
        ## Analysis Task
        
        Please analyze the API distribution data and provide:
        
        1. Insights into the most frequently used CUDA API functions and what they indicate about the application
        2. Analysis of the balance between different types of operations (compute, memory, synchronization)
        3. Identification of any unusual or inefficient API usage patterns
        4. Recommendations for optimizing the API usage
        
        ## Available Data
        
        ### API Distribution Table
        
        {api_distribution_table}
        
        ### API Distribution Visualization
        
        The API distribution visualization (attached as an image) shows the frequency of different CUDA API calls. Pay special attention to:
        
        - The relative proportions of different API types
        - The dominance of specific API calls
        - Any unusual patterns in the distribution
        
        ### API Categories
        
        For reference, CUDA API functions can be categorized as follows:
        
        - **Compute Operations**: cudaLaunchKernel, cudaLaunchCooperativeKernel
        - **Memory Operations**: cudaMalloc, cudaFree, cudaMemcpy, cudaMemcpyAsync, cudaMemset
        - **Synchronization**: cudaStreamSynchronize, cudaDeviceSynchronize, cudaEventSynchronize
        - **Stream Management**: cudaStreamCreate, cudaStreamDestroy, cudaStreamWaitEvent
        - **Event Management**: cudaEventCreate, cudaEventRecord, cudaEventElapsedTime
        
        ## Output Format
        
        Please structure your response as follows:
        
        1. **Most Frequently Used API Functions**: Analysis of the top API functions and what they indicate about the application.
        2. **Balance Between Operation Types**: Analysis of the distribution of different operation types.
        3. **Inefficient API Usage Patterns**: Identification of any unusual or inefficient patterns.
        4. **Recommendations for Optimizing API Usage**: Specific recommendations for improving API usage.
        
        Be specific in your analysis, referencing the actual data from the table and visualization. Provide concrete recommendations that could improve the application's performance.
        """
    
    def _get_memory_operations_template(self):
        """Get the template for the memory operations section"""
        return """
        # CUDA Trace Analysis - Memory Operations

        I need a detailed analysis of the memory operations in the CUDA trace data.
        
        ## Analysis Task
        
        Please analyze the memory operations data and provide:
        
        1. Assessment of the memory transfer patterns and their efficiency
        2. Analysis of the balance between different types of memory operations
        3. Identification of potential memory-related bottlenecks
        4. Recommendations for optimizing memory usage and transfers
        
        ## Available Data
        
        ### Memory Operations Table
        
        {memory_operations_table}
        
        ### Memory Operations Visualization
        
        The memory operations visualization (attached as an image) shows the distribution of memory-related operations. Pay special attention to:
        
        - The relative proportions of different memory operation types
        - The balance between synchronous and asynchronous operations
        - The frequency of allocation and deallocation operations
        
        ### Memory Operation Types
        
        For reference, CUDA memory operations include:
        
        - **Data Transfer**: cudaMemcpy (synchronous), cudaMemcpyAsync (asynchronous)
        - **Memory Allocation**: cudaMalloc, cudaMallocHost, cudaMallocPitch
        - **Memory Deallocation**: cudaFree, cudaFreeHost
        - **Memory Setting**: cudaMemset, cudaMemsetAsync
        
        ## Memory Transfer Efficiency Considerations
        
        - **Synchronous vs. Asynchronous**: Asynchronous transfers (cudaMemcpyAsync) allow overlapping with computation
        - **Transfer Size**: Larger transfers are generally more efficient than many small transfers
        - **Transfer Frequency**: Frequent transfers can indicate inefficient data management
        - **Pinned Memory**: Transfers using pinned host memory are more efficient
        
        ## Output Format
        
        Please structure your response as follows:
        
        1. **Memory Transfer Patterns and Efficiency**: Analysis of how data is being transferred between host and device.
        2. **Balance Between Memory Operation Types**: Analysis of the distribution of different memory operation types.
        3. **Potential Memory-Related Bottlenecks**: Identification of inefficiencies in memory usage.
        4. **Recommendations for Optimizing Memory Usage**: Specific recommendations for improving memory operations.
        
        Be specific in your analysis, referencing the actual data from the table and visualization. Provide concrete recommendations that could improve the application's memory usage efficiency.
        """
    
    def _get_kernel_launches_template(self):
        """Get the template for the kernel launches section"""
        return """
        # CUDA Trace Analysis - Kernel Launches

        I need a detailed analysis of the kernel launch patterns in the CUDA trace data.
        
        ## Analysis Task
        
        Please analyze the kernel launch data and provide:
        
        1. Assessment of the kernel launch patterns and their implications for performance
        2. Analysis of the grid and block dimensions used for kernel launches
        3. Evaluation of kernel occupancy and efficiency based on the launch parameters
        4. Recommendations for optimizing kernel launch configurations
        
        ## Available Data
        
        ### Kernel Distribution Table
        
        {kernel_distribution_table}
        
        ### Grid/Block Dimensions Table
        
        {grid_block_dimensions_table}
        
        ### Kernel Launch Visualization
        
        The kernel launch visualization (attached as an image) shows the distribution of kernel launches. Pay special attention to:
        
        - The relative frequency of different kernels
        - The patterns in grid and block dimensions
        - Any unusual or suboptimal launch configurations
        
        ### GPU Occupancy Considerations
        
        For reference, kernel launch efficiency depends on:
        
        - **Block Size**: Typically, block sizes between 128 and 256 threads provide good occupancy
        - **Grid Size**: Should be large enough to fully utilize the GPU
        - **Dimensions**: 1D, 2D, or 3D configurations should match the data structure
        - **Resource Usage**: Registers and shared memory per thread affect occupancy
        
        ## Output Format
        
        Please structure your response as follows:
        
        1. **Kernel Launch Patterns**: Analysis of how kernels are being launched and the implications for performance.
        2. **Grid and Block Dimensions Analysis**: Evaluation of the grid and block dimensions used.
        3. **Kernel Occupancy and Efficiency**: Assessment of how well the kernels are likely to utilize the GPU.
        4. **Recommendations for Optimizing Kernel Launches**: Specific recommendations for improving kernel launch configurations.
        
        Be specific in your analysis, referencing the actual data from the tables and visualization. Consider the implications of the launch patterns for GPU utilization and overall performance. Provide concrete recommendations that could improve kernel execution efficiency.
        """
    
    def _get_performance_bottlenecks_template(self):
        """Get the template for the performance bottlenecks section"""
        return """
        # CUDA Trace Analysis - Performance Bottlenecks

        I need a detailed analysis of the performance bottlenecks identified in the CUDA trace data.
        
        ## Analysis Task
        
        Please analyze the performance bottlenecks data and provide:
        
        1. Detailed explanation of each identified bottleneck and its impact on performance
        2. Root cause analysis for each bottleneck
        3. Prioritized recommendations for addressing each bottleneck
        4. Potential performance gains from implementing the recommendations
        
        ## Available Data
        
        ### Performance Bottlenecks Table
        
        {performance_bottlenecks_table}
        
        ### Bottleneck Categories
        
        For reference, common CUDA performance bottlenecks include:
        
        - **Memory Transfer Overhead**: Excessive time spent transferring data between host and device
        - **Synchronization Points**: Excessive synchronization limiting parallelism
        - **Low GPU Occupancy**: Suboptimal kernel launch parameters leading to underutilization
        - **Memory Access Patterns**: Uncoalesced memory access reducing memory throughput
        - **Divergent Execution**: Warp divergence reducing computational efficiency
        - **Resource Contention**: Excessive register or shared memory usage limiting occupancy
        
        ## Output Format
        
        Please structure your response as follows:
        
        For each bottleneck identified in the table:
        
        1. **Detailed Explanation**: A thorough explanation of the bottleneck and its impact on performance.
        2. **Root Cause Analysis**: An analysis of the underlying causes of the bottleneck.
        3. **Prioritized Recommendations**: Specific, actionable recommendations for addressing the bottleneck.
        4. **Potential Performance Gains**: An estimate of the performance improvement that could be achieved.
        
        Conclude with a summary of all bottlenecks in order of priority, with an overall assessment of the potential performance improvement if all recommendations are implemented.
        
        Be specific in your analysis, referencing the actual data from the table. Provide concrete, actionable recommendations that could improve the application's performance.
        """
    
    def _get_optimization_recommendations_template(self):
        """Get the template for the optimization recommendations section"""
        return """
        # CUDA Trace Analysis - Optimization Recommendations

        I need comprehensive optimization recommendations based on the CUDA trace data analysis.
        
        ## Analysis Task
        
        Please provide:
        
        1. A prioritized list of optimization recommendations
        2. Detailed explanation of each recommendation and its expected impact
        3. Implementation guidance for each recommendation, including code examples where applicable
        4. Estimation of potential performance improvements for each recommendation
        
        ## Available Data
        
        The recommendations should be based on all the analyses performed on the trace data, including:
        - API distribution analysis
        - Memory operations analysis
        - Kernel launch analysis
        - Performance bottlenecks analysis
        
        ## CUDA Optimization Best Practices
        
        For reference, consider these CUDA optimization best practices:
        
        - **Memory Transfers**: Minimize host-device transfers, use pinned memory, batch transfers
        - **Asynchronous Execution**: Use streams for overlapping operations, minimize synchronization
        - **Memory Access Patterns**: Ensure coalesced memory access, use shared memory for data reuse
        - **Kernel Launch Configuration**: Optimize block size for occupancy, match grid dimensions to data
        - **Resource Usage**: Manage register and shared memory usage to maximize occupancy
        - **Algorithm Design**: Restructure algorithms to maximize parallelism and minimize dependencies
        
        ## Output Format
        
        Please structure your response as follows:
        
        For each recommendation (in order of priority):
        
        1. **Recommendation**: A clear statement of the recommended optimization.
        2. **Detailed Explanation**: Why this optimization is important and how it addresses issues in the trace data.
        3. **Implementation Guidance**: Specific steps to implement the recommendation, including code examples.
        4. **Expected Impact**: An estimate of the performance improvement that could be achieved.
        
        Conclude with a summary of the expected overall performance improvement if all recommendations are implemented.
        
        Be specific and actionable in your recommendations. Provide concrete code examples that demonstrate how to implement each optimization. Focus on recommendations that are likely to have the most significant impact on performance.
        """
    
    def get_template(self, section, data=None):
        """Get a template for a specific section with data filled in"""
        if section not in self.templates:
            raise ValueError(f"Unknown section: {section}")
        
        template = self.templates[section]
        
        if data:
            # Fill in the template with the provided data
            template = template.format(**data)
        
        return template
    
    def save_templates(self, output_dir):
        """Save all templates to files in the specified directory"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        for section, template in self.templates.items():
            output_path = os.path.join(output_dir, f"{section}_template.txt")
            with open(output_path, "w") as f:
                f.write(template)
        
        print(f"Templates saved to {output_dir}")
    
    def load_templates(self, input_dir):
        """Load templates from files in the specified directory"""
        templates = {}
        
        for section in self.templates.keys():
            input_path = os.path.join(input_dir, f"{section}_template.txt")
            if os.path.exists(input_path):
                with open(input_path, "r") as f:
                    templates[section] = f.read()
        
        if templates:
            self.templates.update(templates)
            print(f"Templates loaded from {input_dir}")
        else:
            print(f"No templates found in {input_dir}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Manage CUDA trace LLM prompt templates")
    parser.add_argument("--save", help="Directory to save templates to")
    parser.add_argument("--load", help="Directory to load templates from")
    
    args = parser.parse_args()
    
    templates = CUDAPromptTemplates()
    
    if args.save:
        templates.save_templates(args.save)
    
    if args.load:
        templates.load_templates(args.load)
