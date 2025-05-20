#!/usr/bin/env python3
"""
CUDA Trace LLM Analysis Testing Framework - Test and evaluate LLM analysis quality
"""

import os
import json
import argparse
import time
import re
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import io
import base64
import requests

from enhanced_cuda_llm_analyzer import EnhancedCUDATraceLLMAnalyzer
from cuda_prompt_templates import CUDAPromptTemplates

class CUDALLMAnalysisTester:
    """Test and evaluate LLM analysis quality for CUDA trace data"""
    
    def __init__(self, analysis_dir, enhanced_dir=None, output_dir=None):
        """Initialize the tester with the path to the analysis results directory"""
        self.analysis_dir = analysis_dir
        self.enhanced_dir = enhanced_dir or os.path.join(analysis_dir, "enhanced")
        self.output_dir = output_dir or os.path.join(analysis_dir, "test_results")
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # Initialize the analyzer and prompt templates
        self.analyzer = EnhancedCUDATraceLLMAnalyzer(analysis_dir, enhanced_dir)
        self.prompt_templates = CUDAPromptTemplates()
        
        # Load analysis results
        self.analysis_results = self._load_analysis_results()
        self.summary = self._load_summary()
    
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
    
    def test_mock_analysis(self):
        """Test the mock analysis functionality"""
        print("Testing mock analysis...")
        
        start_time = time.time()
        section_analyses = self.analyzer.generate_mock_analysis()
        end_time = time.time()
        
        test_results = {
            "test_name": "mock_analysis",
            "execution_time": end_time - start_time,
            "sections_analyzed": list(section_analyses.keys()),
            "section_word_counts": {section: len(analysis.split()) for section, analysis in section_analyses.items()},
            "total_word_count": sum(len(analysis.split()) for analysis in section_analyses.values()),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Save test results
        output_path = os.path.join(self.output_dir, "mock_analysis_test_results.json")
        with open(output_path, 'w') as f:
            json.dump(test_results, f, indent=2)
        
        print(f"Mock analysis test completed in {test_results['execution_time']:.2f} seconds")
        print(f"Total word count: {test_results['total_word_count']}")
        print(f"Test results saved to {output_path}")
        
        return test_results
    
    def test_openai_analysis(self, api_key=None):
        """Test the OpenAI analysis functionality"""
        if not api_key:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                print("Error: OpenAI API key not provided. Please set the OPENAI_API_KEY environment variable or provide it as an argument.")
                return None
        
        print("Testing OpenAI analysis...")
        
        start_time = time.time()
        section_analyses = self.analyzer.analyze_with_openai(api_key)
        end_time = time.time()
        
        if not section_analyses:
            print("Error: OpenAI analysis failed.")
            return None
        
        test_results = {
            "test_name": "openai_analysis",
            "execution_time": end_time - start_time,
            "sections_analyzed": list(section_analyses.keys()),
            "section_word_counts": {section: len(analysis.split()) for section, analysis in section_analyses.items()},
            "total_word_count": sum(len(analysis.split()) for analysis in section_analyses.values()),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Save test results
        output_path = os.path.join(self.output_dir, "openai_analysis_test_results.json")
        with open(output_path, 'w') as f:
            json.dump(test_results, f, indent=2)
        
        print(f"OpenAI analysis test completed in {test_results['execution_time']:.2f} seconds")
        print(f"Total word count: {test_results['total_word_count']}")
        print(f"Test results saved to {output_path}")
        
        # Evaluate the quality of the analysis
        quality_metrics = self.evaluate_analysis_quality(section_analyses)
        
        # Save quality metrics
        quality_path = os.path.join(self.output_dir, "openai_analysis_quality_metrics.json")
        with open(quality_path, 'w') as f:
            json.dump(quality_metrics, f, indent=2)
        
        print(f"Analysis quality metrics saved to {quality_path}")
        
        return test_results, quality_metrics
    
    def test_local_llm_analysis(self, model_endpoint="http://localhost:8000/v1/chat/completions"):
        """Test the local LLM analysis functionality"""
        print(f"Testing local LLM analysis with endpoint {model_endpoint}...")
        
        start_time = time.time()
        section_analyses = self.analyzer.analyze_with_local_llm(model_endpoint)
        end_time = time.time()
        
        if not section_analyses:
            print("Error: Local LLM analysis failed.")
            return None
        
        test_results = {
            "test_name": "local_llm_analysis",
            "execution_time": end_time - start_time,
            "sections_analyzed": list(section_analyses.keys()),
            "section_word_counts": {section: len(analysis.split()) for section, analysis in section_analyses.items()},
            "total_word_count": sum(len(analysis.split()) for analysis in section_analyses.values()),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Save test results
        output_path = os.path.join(self.output_dir, "local_llm_analysis_test_results.json")
        with open(output_path, 'w') as f:
            json.dump(test_results, f, indent=2)
        
        print(f"Local LLM analysis test completed in {test_results['execution_time']:.2f} seconds")
        print(f"Total word count: {test_results['total_word_count']}")
        print(f"Test results saved to {output_path}")
        
        # Evaluate the quality of the analysis
        quality_metrics = self.evaluate_analysis_quality(section_analyses)
        
        # Save quality metrics
        quality_path = os.path.join(self.output_dir, "local_llm_analysis_quality_metrics.json")
        with open(quality_path, 'w') as f:
            json.dump(quality_metrics, f, indent=2)
        
        print(f"Analysis quality metrics saved to {quality_path}")
        
        return test_results, quality_metrics
    
    def evaluate_analysis_quality(self, section_analyses):
        """Evaluate the quality of the LLM analysis"""
        print("Evaluating analysis quality...")
        
        quality_metrics = {
            "section_metrics": {},
            "overall_metrics": {}
        }
        
        # Define expected content patterns for each section
        expected_patterns = {
            "overview": [
                r"application characteristics",
                r"key patterns",
                r"performance considerations",
                r"implementation quality"
            ],
            "api_distribution": [
                r"most frequent(ly)?\s+used API",
                r"balance between",
                r"operation types",
                r"inefficient",
                r"recommendations"
            ],
            "memory_operations": [
                r"memory transfer patterns",
                r"efficiency",
                r"balance between",
                r"bottlenecks",
                r"recommendations"
            ],
            "kernel_launches": [
                r"kernel launch patterns",
                r"grid and block dimensions",
                r"occupancy",
                r"efficiency",
                r"recommendations"
            ],
            "performance_bottlenecks": [
                r"bottleneck",
                r"impact",
                r"root cause",
                r"recommendations",
                r"performance gains"
            ],
            "optimization_recommendations": [
                r"recommendation",
                r"implementation",
                r"code example",
                r"expected impact",
                r"performance improvement"
            ]
        }
        
        # Check for code examples in optimization recommendations
        code_pattern = r"```(cuda|c\+\+)?\s.*?```"
        
        # Evaluate each section
        for section, analysis in section_analyses.items():
            if section not in expected_patterns:
                continue
            
            section_metrics = {
                "word_count": len(analysis.split()),
                "pattern_matches": {}
            }
            
            # Check for expected patterns
            for pattern in expected_patterns[section]:
                matches = re.findall(pattern, analysis.lower())
                section_metrics["pattern_matches"][pattern] = len(matches)
            
            # Calculate pattern coverage
            total_patterns = len(expected_patterns[section])
            matched_patterns = sum(1 for count in section_metrics["pattern_matches"].values() if count > 0)
            section_metrics["pattern_coverage"] = matched_patterns / total_patterns if total_patterns > 0 else 0
            
            # Check for code examples in optimization recommendations
            if section == "optimization_recommendations":
                code_matches = re.findall(code_pattern, analysis, re.DOTALL)
                section_metrics["code_examples"] = len(code_matches)
            
            quality_metrics["section_metrics"][section] = section_metrics
        
        # Calculate overall metrics
        total_word_count = sum(metrics["word_count"] for metrics in quality_metrics["section_metrics"].values())
        avg_pattern_coverage = sum(metrics["pattern_coverage"] for metrics in quality_metrics["section_metrics"].values()) / len(quality_metrics["section_metrics"])
        
        quality_metrics["overall_metrics"] = {
            "total_word_count": total_word_count,
            "average_pattern_coverage": avg_pattern_coverage,
            "sections_analyzed": len(quality_metrics["section_metrics"]),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return quality_metrics
    
    def compare_analysis_modes(self, test_results_list):
        """Compare different analysis modes based on test results"""
        print("Comparing analysis modes...")
        
        if len(test_results_list) < 2:
            print("Error: At least two test results are required for comparison.")
            return None
        
        # Extract data for comparison
        modes = [results["test_name"] for results in test_results_list]
        execution_times = [results["execution_time"] for results in test_results_list]
        word_counts = [results["total_word_count"] for results in test_results_list]
        
        # Create comparison metrics
        comparison = {
            "modes": modes,
            "execution_times": execution_times,
            "word_counts": word_counts,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Generate comparison charts
        self._generate_comparison_charts(comparison)
        
        # Save comparison results
        output_path = os.path.join(self.output_dir, "analysis_modes_comparison.json")
        with open(output_path, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        print(f"Analysis modes comparison saved to {output_path}")
        
        return comparison
    
    def _generate_comparison_charts(self, comparison):
        """Generate charts comparing different analysis modes"""
        modes = comparison["modes"]
        execution_times = comparison["execution_times"]
        word_counts = comparison["word_counts"]
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Execution time comparison
        ax1.bar(modes, execution_times, color='skyblue')
        ax1.set_title('Execution Time Comparison')
        ax1.set_ylabel('Time (seconds)')
        ax1.set_xlabel('Analysis Mode')
        
        # Word count comparison
        ax2.bar(modes, word_counts, color='lightgreen')
        ax2.set_title('Word Count Comparison')
        ax2.set_ylabel('Total Words')
        ax2.set_xlabel('Analysis Mode')
        
        plt.tight_layout()
        
        # Save the comparison chart
        output_path = os.path.join(self.output_dir, "analysis_modes_comparison.png")
        plt.savefig(output_path)
        plt.close()
        
        print(f"Comparison charts saved to {output_path}")
    
    def test_prompt_variations(self, api_key=None, section="overview"):
        """Test variations of prompts for a specific section to optimize prompt engineering"""
        if not api_key:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                print("Error: OpenAI API key not provided. Please set the OPENAI_API_KEY environment variable or provide it as an argument.")
                return None
        
        print(f"Testing prompt variations for section: {section}...")
        
        # Define prompt variations
        variations = {
            "standard": self.prompt_templates.get_template(section),
            "concise": self._create_concise_variation(section),
            "detailed": self._create_detailed_variation(section),
            "structured": self._create_structured_variation(section)
        }
        
        results = {}
        
        # Test each variation
        for variation_name, prompt in variations.items():
            print(f"Testing {variation_name} variation...")
            
            # Prepare the API request
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            
            # Prepare the API request payload
            payload = {
                "model": "gpt-4",
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": 2000
            }
            
            try:
                # Make the API request
                start_time = time.time()
                response = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json=payload
                )
                end_time = time.time()
                
                # Check for errors
                response.raise_for_status()
                
                # Parse the response
                result = response.json()
                
                if "choices" in result and len(result["choices"]) > 0:
                    analysis = result["choices"][0]["message"]["content"]
                    
                    # Save the analysis
                    output_path = os.path.join(self.output_dir, f"{section}_{variation_name}_variation.md")
                    with open(output_path, "w") as f:
                        f.write(analysis)
                    
                    # Calculate metrics
                    word_count = len(analysis.split())
                    execution_time = end_time - start_time
                    
                    # Evaluate quality
                    quality_metrics = self._evaluate_single_analysis_quality(analysis, section)
                    
                    results[variation_name] = {
                        "word_count": word_count,
                        "execution_time": execution_time,
                        "quality_metrics": quality_metrics
                    }
                    
                    print(f"{variation_name} variation test completed in {execution_time:.2f} seconds")
                    print(f"Word count: {word_count}")
                else:
                    print(f"Error: Unexpected response format from OpenAI API for {variation_name} variation")
            
            except Exception as e:
                print(f"Error calling OpenAI API for {variation_name} variation: {e}")
        
        # Compare variations
        self._compare_prompt_variations(results, section)
        
        # Save results
        output_path = os.path.join(self.output_dir, f"{section}_prompt_variations_results.json")
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Prompt variation test results saved to {output_path}")
        
        return results
    
    def _create_concise_variation(self, section):
        """Create a more concise variation of the prompt for the specified section"""
        template = self.prompt_templates.get_template(section)
        
        # Simplify the prompt by removing detailed explanations and examples
        concise = re.sub(r"For reference.*?(?=\n\n)", "", template, flags=re.DOTALL)
        concise = re.sub(r"## Output Format.*?(?=\n\n)", "## Output Format\n\nProvide a concise analysis focusing on the most important aspects.", concise, flags=re.DOTALL)
        
        return concise
    
    def _create_detailed_variation(self, section):
        """Create a more detailed variation of the prompt for the specified section"""
        template = self.prompt_templates.get_template(section)
        
        # Add more detailed instructions and examples
        detailed = template + "\n\n## Additional Guidance\n\nPlease provide a very detailed analysis with specific examples from the data. Include quantitative assessments where possible and make sure to thoroughly explain all recommendations. Consider edge cases and potential trade-offs in your analysis."
        
        return detailed
    
    def _create_structured_variation(self, section):
        """Create a more structured variation of the prompt for the specified section"""
        template = self.prompt_templates.get_template(section)
        
        # Add more structured output format requirements
        structured = re.sub(r"## Output Format.*?(?=\n\n)", """## Output Format

Please structure your response using the following exact format:

```
# [Section Title]

## Key Findings
1. [First key finding]
2. [Second key finding]
3. [Third key finding]

## Detailed Analysis
[Detailed analysis with subsections]

## Recommendations
1. [First recommendation]
   - Implementation: [How to implement]
   - Impact: [Expected impact]
2. [Second recommendation]
   - Implementation: [How to implement]
   - Impact: [Expected impact]
3. [Third recommendation]
   - Implementation: [How to implement]
   - Impact: [Expected impact]
```

Follow this structure exactly, filling in the appropriate content for each section.
""", template, flags=re.DOTALL)
        
        return structured
    
    def _evaluate_single_analysis_quality(self, analysis, section):
        """Evaluate the quality of a single analysis"""
        # Define expected content patterns for the section
        expected_patterns = {
            "overview": [
                r"application characteristics",
                r"key patterns",
                r"performance considerations",
                r"implementation quality"
            ],
            "api_distribution": [
                r"most frequent(ly)?\s+used API",
                r"balance between",
                r"operation types",
                r"inefficient",
                r"recommendations"
            ],
            "memory_operations": [
                r"memory transfer patterns",
                r"efficiency",
                r"balance between",
                r"bottlenecks",
                r"recommendations"
            ],
            "kernel_launches": [
                r"kernel launch patterns",
                r"grid and block dimensions",
                r"occupancy",
                r"efficiency",
                r"recommendations"
            ],
            "performance_bottlenecks": [
                r"bottleneck",
                r"impact",
                r"root cause",
                r"recommendations",
                r"performance gains"
            ],
            "optimization_recommendations": [
                r"recommendation",
                r"implementation",
                r"code example",
                r"expected impact",
                r"performance improvement"
            ]
        }
        
        if section not in expected_patterns:
            return {}
        
        metrics = {
            "pattern_matches": {}
        }
        
        # Check for expected patterns
        for pattern in expected_patterns[section]:
            matches = re.findall(pattern, analysis.lower())
            metrics["pattern_matches"][pattern] = len(matches)
        
        # Calculate pattern coverage
        total_patterns = len(expected_patterns[section])
        matched_patterns = sum(1 for count in metrics["pattern_matches"].values() if count > 0)
        metrics["pattern_coverage"] = matched_patterns / total_patterns if total_patterns > 0 else 0
        
        # Check for code examples in optimization recommendations
        if section == "optimization_recommendations":
            code_pattern = r"```(cuda|c\+\+)?\s.*?```"
            code_matches = re.findall(code_pattern, analysis, re.DOTALL)
            metrics["code_examples"] = len(code_matches)
        
        # Check for structure (headings, lists)
        heading_pattern = r"#+\s+.+"
        heading_matches = re.findall(heading_pattern, analysis)
        metrics["headings"] = len(heading_matches)
        
        list_pattern = r"^\s*\d+\.\s+.+|^\s*-\s+.+"
        list_matches = re.findall(list_pattern, analysis, re.MULTILINE)
        metrics["list_items"] = len(list_matches)
        
        return metrics
    
    def _compare_prompt_variations(self, results, section):
        """Compare different prompt variations based on test results"""
        if len(results) < 2:
            print("Error: At least two variation results are required for comparison.")
            return
        
        # Extract data for comparison
        variations = list(results.keys())
        execution_times = [results[var]["execution_time"] for var in variations]
        word_counts = [results[var]["word_count"] for var in variations]
        pattern_coverages = [results[var]["quality_metrics"]["pattern_coverage"] for var in variations]
        
        # Create figure with three subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Execution time comparison
        ax1.bar(variations, execution_times, color='skyblue')
        ax1.set_title('Execution Time Comparison')
        ax1.set_ylabel('Time (seconds)')
        ax1.set_xlabel('Prompt Variation')
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
        
        # Word count comparison
        ax2.bar(variations, word_counts, color='lightgreen')
        ax2.set_title('Word Count Comparison')
        ax2.set_ylabel('Total Words')
        ax2.set_xlabel('Prompt Variation')
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        
        # Pattern coverage comparison
        ax3.bar(variations, pattern_coverages, color='salmon')
        ax3.set_title('Pattern Coverage Comparison')
        ax3.set_ylabel('Coverage (0-1)')
        ax3.set_xlabel('Prompt Variation')
        plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save the comparison chart
        output_path = os.path.join(self.output_dir, f"{section}_prompt_variations_comparison.png")
        plt.savefig(output_path)
        plt.close()
        
        print(f"Prompt variations comparison chart saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test and evaluate LLM analysis quality for CUDA trace data")
    parser.add_argument("--analysis_dir", required=True, help="Path to analysis results directory")
    parser.add_argument("--enhanced_dir", default=None, help="Path to enhanced visualizations directory")
    parser.add_argument("--output_dir", default=None, help="Path to output directory for test results")
    parser.add_argument("--test_mock", action="store_true", help="Test mock analysis")
    parser.add_argument("--test_openai", action="store_true", help="Test OpenAI analysis")
    parser.add_argument("--test_local", action="store_true", help="Test local LLM analysis")
    parser.add_argument("--api_key", help="OpenAI API key (for OpenAI tests)")
    parser.add_argument("--model_endpoint", default="http://localhost:8000/v1/chat/completions", help="Local LLM API endpoint (for local tests)")
    parser.add_argument("--test_prompt_variations", action="store_true", help="Test prompt variations")
    parser.add_argument("--section", default="overview", help="Section to test prompt variations for")
    
    args = parser.parse_args()
    
    tester = CUDALLMAnalysisTester(args.analysis_dir, args.enhanced_dir, args.output_dir)
    
    test_results = []
    
    if args.test_mock:
        mock_results = tester.test_mock_analysis()
        test_results.append(mock_results)
    
    if args.test_openai:
        openai_results = tester.test_openai_analysis(args.api_key)
        if openai_results:
            test_results.append(openai_results[0])
    
    if args.test_local:
        local_results = tester.test_local_llm_analysis(args.model_endpoint)
        if local_results:
            test_results.append(local_results[0])
    
    if len(test_results) >= 2:
        tester.compare_analysis_modes(test_results)
    
    if args.test_prompt_variations:
        tester.test_prompt_variations(args.api_key, args.section)
