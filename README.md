# SemanMark: Semantic Watermarking Framework

A framework for code watermarking using Adaptive Direction Generators (ADG) for robust semantic watermark embedding in source code.

## Features

- **Strategy 6**: Adaptive Direction Generator (ADG) driven watermark embedding
- **Multi-language Support**: Java and JavaScript code watermarking
- **Robustness Testing**: Multiple attack scenarios and evaluation
- **Complete Pipeline**: End-to-end watermark embedding and extraction
- **Semantic Preservation**: Maintains code functionality while embedding watermarks

## Quick Start

### System Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-compatible GPU (recommended)
- 16GB+ RAM

### Installation

```bash
pip install -r requirements.txt
```

### Running Tests

#### Method 1: Direct Python Execution

**Java Watermarking (Default)**:
```bash
cd dimension_strategy_comparison
python test_strategy6_pipeline.py --concurrency 5 --resume
```

**JavaScript Watermarking**:
```bash
cd dimension_strategy_comparison
python test_strategy6_pipeline.py --config configs/base_config_js.json --concurrency 10 --resume
```

#### Method 2: Batch Files (Windows)

**Java Watermarking**:
```bash
cd dimension_strategy_comparison
run_background.bat
```

**JavaScript Watermarking**:
```bash
cd dimension_strategy_comparison
run_background_js.bat
```

## Project Structure

```
SemanMark/
|-- dimension_strategy_comparison/           # Main framework
|   |-- test_strategy6_pipeline.py           # Main test orchestrator
|   |-- run_background.bat                   # Java batch execution
|   |-- run_background_js.bat                # JavaScript batch execution
|   |-- scripts/                             # Step implementations
|   |   |-- step1c_train_adaptive_generator.py  # Train ADG
|   |   |-- step2_select_dimensions.py          # Dimension selection
|   |   |-- step3_embed_watermarks.py           # Watermark embedding
|   |   |-- step4_extract_with_attacks.py       # Attack extraction
|   |   |-- step5_analyze_results.py            # Result analysis
|   |   |-- dataset_loader.py                   # Data loading utility
|   |-- configs/                             # Configuration files
|   |   |-- base_config.json                 # Default Java config
|   |   |-- base_config_js.json              # JavaScript config
|   |   |-- strategy_6_adaptive.json         # Strategy 6 config
|   |-- models/                              # Model definitions
|   |   |-- adaptive_direction_generator.py  # ADG implementation
|   |-- data/                                # Data directory (empty)
|   |-- results/                             # Results directory (empty)
|
|-- Watermark4code/                          # Core watermarking library
|   |-- encoder/                             # Model encoding
|   |-- injection/                           # Watermark injection
|   |-- keys/                                # Key generation
|   |-- utils/                               # Utility functions
|   |-- experiments/Attack/                  # Attack implementations
|
|-- SrcMarker-main/                          # Code transformation toolkit
|   |-- mutable_tree/                        # AST manipulation
|   |-- contrastive_learning/                # Code augmentation
|   |-- metrics/                             # Evaluation metrics
|   |-- code_transform_provider.py           # Transformation provider
|
|-- requirements.txt                         # Python dependencies
|-- README.md                                # This file
```

## Configuration Files

### base_config.json
Default configuration for Java watermarking:
- Dataset: CSN-Java
- Embedding: 4-bit watermarks
- Model: CodeT5-base
- Training: 200 code samples
- Testing: 30 code samples

### base_config_js.json
Configuration for JavaScript watermarking:
- Dataset: JavaScript subset
- Same embedding parameters as Java
- Adapted preprocessing for JS syntax

### strategy_6_adaptive.json
Strategy-specific parameters:
- Adaptive Direction Generator settings
- Embedding strength parameters
- Extraction thresholds

## Pipeline Steps

1. **Step 1**: Train Adaptive Direction Generator
   - Learns optimal embedding directions for current dataset
   - Uses contrastive learning for robustness
   - Output: trained_generator.pth

2. **Step 2**: Select Embedding Dimensions
   - Choose most robust dimensions for watermarking
   - Generates orthogonal embedding directions
   - Output: selected_dimensions.json

3. **Step 3**: Embed Watermarks
   - Inject watermarks into code samples
   - Preserves semantic meaning and functionality
   - Output: embedded_watermarks.json

4. **Step 4**: Extract with Attacks
   - Tests watermark robustness against transformations
   - Applies attacks: variable renaming, code formatting, etc.
   - Output: extraction_results.json

5. **Step 5**: Analyze Results
   - Computes extraction accuracy and robustness metrics
   - Generates performance analysis
   - Output: test_report.json

## Command Line Options

### test_strategy6_pipeline.py

```bash
python test_strategy6_pipeline.py [options]

Options:
  --config CONFIG           Configuration file path
                            Default: configs/base_config.json
  --concurrency N           Number of parallel processes
                            Default: 5
  --resume                  Resume from previous run
                            Skips completed steps
  --skip-steps STEPS        Skip specific steps (comma-separated)
                            Example: --skip-steps step1,step2
  --verbose                 Enable verbose output
  --test-mode               Quick test mode with fewer samples
```

### Batch Files

- **run_background.bat**: Java watermarking with default settings
- **run_background_js.bat**: JavaScript watermarking

## Output Files

Results are saved in `dimension_strategy_comparison/results/strategy_6_adaptive*/`:

- **trained_generator.pth**: Trained ADG model weights
- **selected_dimensions.json**: Chosen embedding dimensions and directions
- **embedded_watermarks.json**: Watermark embedding results for each sample
- **extraction_results.json**: Attack robustness evaluation results
- **test_report.json**: Complete evaluation report with metrics

## Troubleshooting

### Common Issues

1. **CUDA out of memory**
   - Solution: Reduce --concurrency parameter
   - Example: python test_strategy6_pipeline.py --concurrency 2

2. **Missing dependencies**
   - Solution: Install all requirements
   - Command: pip install -r requirements.txt

3. **Path issues**
   - Solution: Run from project root directory
   - Ensure data files exist in expected locations

4. **Model download timeout**
   - Solution: Ensure stable internet connection for first run
   - Pre-trained models will be downloaded automatically

### Performance Tuning

- **GPU Memory**: Adjust batch sizes in config files
- **CPU Cores**: Set --concurrency based on your system (2-16)
- **Dataset Size**: Use --test-mode for quick validation

## License

MIT License
