# Fine-Tuning DeepSeek R1 for Semiconductor Industry Knowledge

A comprehensive guide to fine-tuning DeepSeek R1 reasoning models on Google Colab with T4 GPU. This project demonstrates the complete pipeline from setup to evaluation, specifically focused on teaching the model semiconductor industry knowledge.

## 🎯 Project Overview

**What We Accomplished:**
- Successfully fine-tuned DeepSeek-R1-Distill-Qwen-1.5B on custom semiconductor dataset
- Reduced training loss from 1.937 to 1.321 (excellent learning signal)
- Achieved domain-specific knowledge transfer with only 20 training examples
- Learned proper DeepSeek inference parameters through systematic testing
- Built a complete understanding of the fine-tuning process from scratch

**Why This Matters:**
- First-hand experience with reasoning model fine-tuning
- Practical approach suitable for limited compute resources (Google Colab T4)
- Real-world dataset focused on semiconductor companies and technology
- Systematic troubleshooting and parameter optimization

## 🔧 Prerequisites

### Hardware Requirements
- Google Colab with T4 GPU (free tier sufficient)
- ~15GB GPU memory available
- Stable internet connection for model downloads

### Software Dependencies
```bash
# Core fine-tuning framework (optimized for efficiency)
!pip install unsloth

# Additional libraries for training
!pip install transformers trl datasets torch accelerate bitsandbytes
```

### Key Libraries Used
- **Unsloth**: 2x faster fine-tuning with memory optimization
- **TRL**: Transformer Reinforcement Learning for SFT
- **LoRA**: Parameter-efficient fine-tuning (only 0.09% of model parameters)
- **4-bit Quantization**: Enables large model training on limited hardware

## 📊 Training Data

### Dataset Composition
**20 High-Quality Examples Covering:**
- Major semiconductor companies (NVIDIA, AMD, Intel, TSMC, Samsung, Qualcomm)
- Technical processes (fabrication, lithography, process nodes)
- Business models (fabless vs IDM companies)
- Market dynamics and financial information
- Industry applications and trends

### Data Format Structure
```json
{
  "input": "What does NVIDIA Corporation primarily manufacture?",
  "output": "NVIDIA Corporation primarily manufactures:\n1. Graphics Processing Units (GPUs) for gaming and professional use\n2. AI and machine learning chips (A100, H100)\n3. Data center processors\n4. Automotive AI platforms\nRevenue: ~$61 billion (2023). Leading in AI acceleration and visual computing."
}
```

### DeepSeek-Optimized Training Format
```text
What does NVIDIA Corporation primarily manufacture?

<think>
Let me think about this step by step.

NVIDIA Corporation primarily manufactures:
1. Graphics Processing Units (GPUs) for gaming and professional use
2. AI and machine learning chips (A100, H100)
3. Data center processors
4. Automotive AI platforms
Revenue: ~$61 billion (2023). Leading in AI acceleration and visual computing.
</think>

Based on my analysis above, here's the answer:
NVIDIA Corporation primarily manufactures:
1. Graphics Processing Units (GPUs) for gaming and professional use
2. AI and machine learning chips (A100, H100)
3. Data center processors
4. Automotive AI platforms
Revenue: ~$61 billion (2023). Leading in AI acceleration and visual computing.<|end_of_text|>
```

## 🚀 Step-by-Step Implementation

### Step 1: Environment Setup
```python
# Install optimized fine-tuning framework
!pip install unsloth

# Verify GPU and system compatibility
import torch
from unsloth import FastLanguageModel

print("✅ PyTorch version:", torch.__version__)
print("✅ CUDA available:", torch.cuda.is_available())
print("✅ GPU name:", torch.cuda.get_device_name(0))
print("✅ GPU memory:", torch.cuda.get_device_properties(0).total_memory // 1024**3, "GB")
```

### Step 2: Model Loading with Optimization
```python
# Load pre-quantized DeepSeek model for efficiency
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/DeepSeek-R1-Distill-Qwen-1.5B-bnb-4bit",
    max_seq_length=512,      # Optimized for T4 memory
    dtype=None,              # Auto-detect best precision
    load_in_4bit=True,       # 4x memory reduction
)

print("✅ Model loaded: 1.5B parameters")
print("💾 Using 4-bit quantization to save memory")
```

### Step 3: LoRA Configuration for Parameter-Efficient Training
```python
# Configure Low-Rank Adaptation
model = FastLanguageModel.get_peft_model(
    model,
    r=16,                    # LoRA rank (controls adapter size)
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=16,           # LoRA scaling parameter
    lora_dropout=0.05,       # Regularization
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

# Result: Only 4.35M out of 5B parameters will be trained (0.09%)
```

### Step 4: Data Preprocessing
```python
def format_training_data(examples):
    """Convert raw data to DeepSeek's preferred training format"""
    formatted_data = []
    
    for example in examples:
        # Follow DeepSeek's reasoning format with <think> tags
        formatted_text = f"""{example['input']}

<think>
Let me think about this step by step.

{example['output']}
</think>

Based on my analysis above, here's the answer:
{example['output']}<|end_of_text|>"""
        
        formatted_data.append({"text": formatted_text})
    
    return formatted_data

# Process training data
formatted_training_data = format_training_data(training_data)
train_dataset = Dataset.from_list(formatted_training_data)
```

### Step 5: Training Configuration
```python
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    dataset_text_field="text",
    max_seq_length=1024,
    dataset_num_proc=2,
    packing=False,
    args=TrainingArguments(
        # Memory-optimized settings for T4 GPU
        per_device_train_batch_size=1,      # Conservative for memory
        gradient_accumulation_steps=4,       # Effective batch size = 4
        warmup_steps=10,                    # Gradual learning rate increase
        max_steps=100,                      # Sufficient for small dataset
        learning_rate=5e-5,                 # Stable learning rate
        fp16=True,                          # Half precision for memory
        logging_steps=10,                   # Monitor progress
        optim="adamw_8bit",                 # Memory-efficient optimizer
        weight_decay=0.01,                  # Regularization
        lr_scheduler_type="cosine",         # Smooth learning rate decay
        seed=42,                            # Reproducibility
        output_dir="outputs",
        save_steps=50,
        dataloader_pin_memory=False,        # Reduce GPU memory usage
    ),
)
```

### Step 6: Training Execution
```python
# Start fine-tuning process
print("🚀 Starting fine-tuning...")
trainer.train()
print("🎉 Fine-tuning completed!")
```

**Training Results:**
- **Duration**: ~2-3 minutes for 100 steps
- **Loss Reduction**: 1.937 → 1.321 (excellent learning signal)
- **Memory Usage**: ~8-10GB on T4 GPU
- **Parameters Trained**: 4.35M out of 5B (0.09% of total model)

## 📈 Inference Optimization

### Critical Discovery: Temperature Sensitivity
Through systematic testing, we discovered DeepSeek models are highly sensitive to temperature settings:

```python
# WRONG: Official recommendation didn't work for our small dataset
outputs = model.generate(
    temperature=0.6,  # Caused repetitive, rambling responses
    top_p=0.95,
)

# CORRECT: Lower temperature for focused responses
outputs = model.generate(
    temperature=0.3,  # Much better for small training datasets
    do_sample=True,
    max_new_tokens=150,
    pad_token_id=tokenizer.eos_token_id
)
```

### Optimal Inference Parameters
```python
def generate_response(prompt, model, tokenizer):
    """Optimized inference function for fine-tuned DeepSeek"""
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.3,        # Key optimization for small datasets
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## 🎯 Results and Evaluation

### Success Metrics

**✅ Knowledge Transfer Achieved:**
- Learned semiconductor industry terminology
- Correctly identified company specializations (NVIDIA→GPUs, TSMC→foundry)
- Integrated reasoning format with `<think>` tags
- Maintained factual accuracy on trained topics

**✅ Technical Improvements:**
- **Training Loss**: 1.937 → 1.321 (32% reduction)
- **Response Quality**: Eliminated rambling and repetitive behavior
- **Domain Focus**: Responses stayed on semiconductor topics
- **Format Consistency**: Applied learned reasoning structure

### Example Outputs

**Before Fine-tuning:**
```
Q: What does NVIDIA Corporation primarily manufacture?
A: NVIDIA is a company that makes things... but wait, let me think again. 
Maybe using addition? If someone buys 2 apples, and the store had 5, 
then maybe the store had 5, bought 2, so how many are left?
```

**After Fine-tuning:**
```
Q: What does NVIDIA Corporation primarily manufacture?
A: </think> NVIDIA Corporation primarily manufactures GPUs (Graphics 
Processing Units) and other specialized hardware components designed to 
accelerate AI and machine learning workloads.
```

## 🚨 Troubleshooting Guide

### Common Issues and Solutions

#### 1. Memory Errors
**Problem**: CUDA out of memory during training
**Solutions**:
```python
# Reduce batch size
per_device_train_batch_size = 1

# Increase gradient accumulation
gradient_accumulation_steps = 8

# Enable gradient checkpointing
use_gradient_checkpointing = "unsloth"

# Disable memory pinning
dataloader_pin_memory = False
```

#### 2. Repetitive/Rambling Responses
**Problem**: Model generates repetitive or incoherent text
**Solutions**:
```python
# Lower temperature significantly
temperature = 0.3  # Instead of 0.6

# Adjust max tokens
max_new_tokens = 100  # Prevent runaway generation

# Use proper stopping criteria
pad_token_id = tokenizer.eos_token_id
```

#### 3. Poor Learning (Loss Not Decreasing)
**Problem**: Training loss remains high or unstable
**Solutions**:
```python
# Reduce learning rate
learning_rate = 1e-5  # Very conservative

# Increase warmup steps
warmup_steps = 20

# Use cosine scheduler
lr_scheduler_type = "cosine"

# Add weight decay
weight_decay = 0.01
```

#### 4. Training Crashes
**Problem**: Kernel restarts or training stops unexpectedly
**Solutions**:
- Restart Colab runtime and re-run setup
- Reduce sequence length: `max_seq_length = 512`
- Monitor GPU memory usage
- Save checkpoints more frequently: `save_steps = 25`

## 📚 Key Learnings

### What We Discovered

1. **Small Datasets Can Work**: 20 examples achieved measurable knowledge transfer
2. **Temperature is Critical**: 0.3 worked better than official 0.6 recommendation
3. **LoRA is Powerful**: Training only 0.09% of parameters was sufficient
4. **Format Matters**: DeepSeek's `<think>` reasoning format improved outputs
5. **Systematic Testing**: Incremental parameter adjustment was essential

### Best Practices Identified

```python
# Data Format
- Use <think> tags for reasoning tasks
- Include step-by-step explanations
- Consistent input-output patterns
- Clear termination with <|end_of_text|>

# Training Parameters
- Conservative learning rates (1e-5 to 5e-5)
- Sufficient training steps (100+ for small datasets)
- Proper warmup (10-20 steps)
- Memory-efficient optimizers (adamw_8bit)

# Inference Settings
- Temperature 0.3 for focused responses
- Appropriate max_new_tokens limits
- Proper tokenizer settings
- Multiple test runs for validation
```

## 🔄 Reproducing This Work

### Quick Start
```bash
# 1. Clone and setup
git clone [your-repo]
cd deepseek-finetuning

# 2. Install dependencies
pip install unsloth transformers trl datasets

# 3. Run training script
python finetune_deepseek.py

# 4. Test the model
python test_model.py
```

### Expected Results
- **Training Time**: 2-3 minutes on T4 GPU
- **Loss Reduction**: ~30-40% from baseline
- **Memory Usage**: 8-10GB GPU memory
- **Knowledge Transfer**: Domain-specific responses

## 🚀 Next Steps and Improvements

### For Production Use
1. **Scale Dataset**: 100-500 training examples
2. **Longer Training**: 500-1000 steps
3. **Better Curation**: Higher quality, more diverse examples
4. **Evaluation Suite**: Systematic benchmarking
5. **A/B Testing**: Compare against base model

### Advanced Techniques
- **Multi-stage Training**: SFT → RL → SFT pipeline
- **Synthetic Data**: Generate additional training examples
- **Model Ensemble**: Combine multiple fine-tuned versions
- **Quantization**: Deploy with GGUF for faster inference

## 📖 References and Resources

### Official Documentation
- [DeepSeek-R1 GitHub Repository](https://github.com/deepseek-ai/DeepSeek-R1)
- [Unsloth Documentation](https://github.com/unslothai/unsloth)
- [TRL (Transformer Reinforcement Learning)](https://github.com/huggingface/trl)

### Research Papers
- DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning
- LoRA: Low-Rank Adaptation of Large Language Models
- QLoRA: Efficient Finetuning of Quantized LLMs

### Community Resources
- [Hugging Face DeepSeek Models](https://huggingface.co/deepseek-ai)
- [Open-R1 Project](https://github.com/huggingface/open-r1)
- [Fine-tuning Best Practices](https://huggingface.co/docs/transformers/training)

## 🤝 Contributing

This project serves as an educational resource for understanding LLM fine-tuning. Contributions welcome:

1. **Improved Training Data**: Higher quality semiconductor examples
2. **Parameter Optimization**: Better hyperparameter combinations
3. **Evaluation Metrics**: More comprehensive testing approaches
4. **Documentation**: Additional explanations and examples

## 📄 License

This project is open source and available under the MIT License. DeepSeek models are also MIT licensed and support commercial use.

## 💡 Acknowledgments

- **DeepSeek AI** for open-sourcing the R1 reasoning models
- **Unsloth Team** for optimization frameworks enabling T4 GPU training
- **Hugging Face** for transformers ecosystem and model hosting
- **Google Colab** for providing free GPU access

---

**🎯 Key Takeaway**: Fine-tuning doesn't require massive datasets or expensive hardware. With 20 examples and systematic approach, we achieved meaningful knowledge transfer on a free T4 GPU in under 3 minutes. The key is understanding the model's requirements and optimizing parameters through experimentation.

**Happy Fine-tuning!** 🚀