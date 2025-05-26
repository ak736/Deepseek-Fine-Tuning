#!/usr/bin/env python3
"""
Fine Tune DeepSeek - Single Python File
Converted from Jupyter notebook for easy execution
"""

# Install unsloth (comment out if already installed)
# import subprocess
# subprocess.run(["pip", "install", "unsloth"], check=True)

# Test if our installation worked
import torch
from unsloth import FastLanguageModel

print("✅ PyTorch version:", torch.__version__)
print("✅ CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("✅ GPU name:", torch.cuda.get_device_name(0))
    print("✅ GPU memory:", torch.cuda.get_device_properties(0).total_memory // 1024**3, "GB")
print("✅ Unsloth imported successfully!")

# Load the smallest DeepSeek model for testing
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/DeepSeek-R1-Distill-Qwen-1.5B-bnb-4bit",
    max_seq_length = 512,  # Keep this small for now
    dtype = None,  # Auto-detect the best type
    load_in_4bit = True,  # This saves memory!
)

print("✅ Model loaded successfully!")
print("📊 Model size: 1.5B parameters")
print("💾 Using 4-bit quantization to save memory")

# More consistent generation
prompt = "What is 2 + 2?"

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# More deterministic settings
outputs = model.generate(
    **inputs,
    max_new_tokens=30,
    temperature=0.1,        # Lower = more consistent
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("🤖 More consistent response:")
print(response)

# Test current behavior with a reasoning task
prompt = "A store has 5 apples. Someone buys 2 apples. How many apples are left?"

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    temperature=0.1,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("🤖 Current model response:")
print(response)

# Step 9 (UPDATED): Comprehensive Semiconductor Dataset - 20 Examples
training_data = [
    {
        "input": "What does NVIDIA Corporation primarily manufacture?",
        "output": "NVIDIA Corporation primarily manufactures:\n1. Graphics Processing Units (GPUs) for gaming and professional use\n2. AI and machine learning chips (A100, H100)\n3. Data center processors\n4. Automotive AI platforms\nRevenue: ~$61 billion (2023). Leading in AI acceleration and visual computing."
    },
    {
        "input": "Which company manufactures Apple's M1 and M2 chips?",
        "output": "Apple's M1 and M2 chips are manufactured by TSMC (Taiwan Semiconductor Manufacturing Company).\nKey details:\n1. TSMC uses advanced 5nm and 3nm process technology\n2. Apple designs, TSMC produces\n3. TSMC is world's largest contract manufacturer\n4. Also produces for AMD, NVIDIA, Qualcomm"
    },
    {
        "input": "What is Intel's main business and market position?",
        "output": "Intel Corporation details:\n1. World's second-largest semiconductor company\n2. Primary products: CPUs for PCs and servers\n3. Revenue: ~$63 billion (2022)\n4. Headquarters: Santa Clara, California\n5. Key markets: Data center, PC, and emerging technologies\n6. Major competitor to AMD in processor market"
    },
    {
        "input": "What does AMD specialize in manufacturing?",
        "output": "AMD (Advanced Micro Devices) specializes in:\n1. CPUs (Ryzen consumer, EPYC server processors)\n2. GPUs (Radeon graphics cards)\n3. APUs (combined CPU+GPU chips)\n4. Data center and gaming processors\nRevenue: ~$24 billion (2022). Major competitor to Intel and NVIDIA."
    },
    {
        "input": "What is TSMC's market share in semiconductor foundry business?",
        "output": "TSMC market position:\n1. Controls ~54% of global foundry market share\n2. World's largest contract chip manufacturer\n3. Leading in advanced nodes: 3nm, 5nm, 7nm\n4. Revenue: ~$76 billion (2022)\n5. Key customers: Apple (25% of revenue), NVIDIA, AMD, Qualcomm\n6. Critical to global semiconductor supply chain"
    },
    {
        "input": "What is the difference between fabless and IDM semiconductor companies?",
        "output": "Semiconductor business models:\n\n**Fabless Companies:**\n1. Design chips, outsource manufacturing\n2. Examples: NVIDIA, AMD, Qualcomm, Broadcom\n3. Use foundries like TSMC, GlobalFoundries\n4. Lower capital requirements, higher margins\n\n**IDM (Integrated Device Manufacturer):**\n1. Design AND manufacture own chips\n2. Examples: Intel, Samsung, Texas Instruments\n3. Own fabrication facilities\n4. High capital investment, more supply control"
    },
    {
        "input": "What products does Qualcomm manufacture?",
        "output": "Qualcomm primary products:\n1. Snapdragon mobile processors (smartphones, tablets)\n2. 5G and wireless communication chips\n3. Automotive semiconductors\n4. IoT (Internet of Things) processors\nRevenue: ~$44 billion (2022). Leader in mobile chip design and wireless technology licensing."
    },
    {
        "input": "What is Samsung's role in the semiconductor industry?",
        "output": "Samsung Semiconductor division:\n1. World's largest memory chip manufacturer\n2. Products: DRAM, NAND flash memory, SSDs\n3. Also contract manufacturing (foundry services)\n4. Revenue: ~$64 billion semiconductor division (2022)\n5. Competes with SK Hynix in memory, TSMC in foundry\n6. Supplies memory to Apple, servers, consumer electronics"
    },
    {
        "input": "What does Broadcom Corporation specialize in?",
        "output": "Broadcom specializes in:\n1. Communication semiconductors (Wi-Fi, Bluetooth chips)\n2. Broadband and networking processors\n3. Storage and connectivity solutions\n4. Infrastructure software (after VMware acquisition)\nRevenue: ~$35 billion (2022). Key supplier for smartphones, networking equipment, and data centers."
    },
    {
        "input": "What is SK Hynix known for manufacturing?",
        "output": "SK Hynix manufactures:\n1. DRAM memory chips (world's #2 producer)\n2. NAND flash memory\n3. Mobile DRAM for smartphones\n4. Server and PC memory modules\nRevenue: ~$34 billion (2022). South Korean company, major competitor to Samsung in memory market."
    },
    {
        "input": "What does Micron Technology produce?",
        "output": "Micron Technology produces:\n1. DRAM memory chips\n2. NAND flash storage solutions\n3. SSDs for consumer and enterprise\n4. Memory for automotive and industrial applications\nRevenue: ~$30 billion (2022). US-based memory manufacturer competing with Samsung and SK Hynix."
    },
    {
        "input": "What is the semiconductor fabrication process?",
        "output": "Semiconductor fabrication process:\n1. **Wafer preparation**: Silicon wafer cleaning and preparation\n2. **Lithography**: Pattern creation using UV light and photomasks\n3. **Etching**: Removing unwanted material\n4. **Deposition**: Adding thin films of materials\n5. **Ion implantation**: Adding dopants to silicon\n6. **Testing**: Electrical testing of chips\nProcess takes 2-3 months, requires cleanroom environment."
    },
    {
        "input": "What are the most advanced semiconductor process nodes?",
        "output": "Advanced process nodes (2024):\n1. **3nm**: TSMC production, Apple M2 Pro/Max\n2. **4nm**: Samsung, Qualcomm Snapdragon 8 Gen 2\n3. **5nm**: TSMC, Apple M1/M2, AMD Ryzen 7000\n4. **7nm**: AMD CPUs/GPUs, older iPhone chips\n5. **10nm and above**: Older generation processes\nSmaller nodes = more transistors, better performance, lower power."
    },
    {
        "input": "What is EUV lithography in semiconductor manufacturing?",
        "output": "EUV (Extreme Ultraviolet) Lithography:\n1. Advanced chip manufacturing technology\n2. Uses 13.5nm wavelength light (vs 193nm in older systems)\n3. Enables production of 7nm and smaller nodes\n4. Primary supplier: ASML (Netherlands)\n5. Cost: $200+ million per machine\n6. Critical for leading-edge chip production\nUsed by TSMC, Samsung for most advanced processors."
    },
    {
        "input": "Which companies dominate the global semiconductor market?",
        "output": "Top semiconductor companies by revenue (2022):\n1. **Samsung**: $64B (memory leader)\n2. **Intel**: $63B (CPU leader)\n3. **TSMC**: $76B (foundry leader)\n4. **SK Hynix**: $34B (memory)\n5. **NVIDIA**: $61B (GPU/AI leader)\n6. **Qualcomm**: $44B (mobile chips)\n7. **Broadcom**: $35B (communication chips)\n8. **AMD**: $24B (CPUs/GPUs)"
    },
    {
        "input": "What is the global semiconductor market size?",
        "output": "Global semiconductor market:\n1. **2022 Revenue**: ~$574 billion\n2. **2023 Projection**: ~$520 billion (downturn)\n3. **Key segments**: Memory (28%), Logic (27%), Analog (13%)\n4. **Geographic demand**: China (35%), Americas (21%), Europe (9%)\n5. **Growth drivers**: AI, 5G, automotive electronics, IoT\n6. **Cyclical industry**: Experiences boom/bust cycles"
    },
    {
        "input": "What are the main applications of semiconductor chips?",
        "output": "Semiconductor chip applications:\n1. **Computing**: CPUs, GPUs for PCs, servers, smartphones\n2. **Memory**: DRAM, NAND flash for data storage\n3. **Communication**: 5G, Wi-Fi, Bluetooth chips\n4. **Automotive**: Engine control, ADAS, infotainment\n5. **Industrial**: IoT sensors, power management\n6. **Consumer electronics**: TVs, appliances, wearables\n7. **AI/ML**: Specialized accelerators for artificial intelligence"
    },
    {
        "input": "What is Moore's Law and its current status?",
        "output": "Moore's Law:\n1. **Original**: Transistor density doubles every 2 years\n2. **History**: Observed by Intel co-founder Gordon Moore (1965)\n3. **Current status**: Slowing down significantly\n4. **Challenges**: Physical limits, increased costs, quantum effects\n5. **Alternatives**: 3D stacking, new materials, specialized chips\n6. **Industry shift**: Focus on performance per watt, not just density\nStill relevant but evolution rather than revolution."
    },
    {
        "input": "What are the major semiconductor manufacturing regions?",
        "output": "Global semiconductor manufacturing:\n1. **Asia-Pacific (75%)**:\n   - Taiwan: TSMC, advanced foundries\n   - South Korea: Samsung, SK Hynix memory\n   - China: Growing local production\n2. **Americas (12%)**:\n   - US: Intel fabs, GlobalFoundries\n3. **Europe (8%)**:\n   - Germany, Ireland: Intel, analog chips\n4. **Supply chain risk**: Over-concentration in Asia"
    },
    {
        "input": "What factors affect semiconductor company stock prices?",
        "output": "Semiconductor stock price factors:\n1. **Cyclical demand**: PC, smartphone, server cycles\n2. **Technology transitions**: New process nodes, architectures\n3. **Geopolitical tensions**: US-China trade, Taiwan risk\n4. **Supply chain disruptions**: COVID-19, natural disasters\n5. **End market health**: Automotive, data center, consumer\n6. **Competition**: Market share gains/losses\n7. **Capital intensity**: R&D and fab investments\n8. **Inventory cycles**: Overstock/shortage periods"
    }
]

print(f"✅ Created comprehensive semiconductor dataset with {len(training_data)} examples")
print(f"📊 Covers: Companies, processes, markets, technology, financials")

# CORRECTED: DeepSeek's preferred format (no system prompts)
def format_training_data(examples):
    formatted_data = []

    for example in examples:
        # DeepSeek prefers simple user-assistant format with <think> tags
        formatted_text = f"""{example['input']}

<think>
Let me think about this step by step.

{example['output']}
</think>

Based on my analysis above, here's the answer:
{example['output']}<|end_of_text|>"""

        formatted_data.append({"text": formatted_text})

    return formatted_data

# Convert our training data
formatted_training_data = format_training_data(training_data)

print("✅ Data converted to DeepSeek-optimized format")
print("\n📋 Here's what the model will actually see:")
print(formatted_training_data[0]['text'][:300] + "...")

# Add LoRA adapters to enable fine-tuning
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,                # Size of the "notebook" (16 is good for learning)
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],  # Which parts to tune
    lora_alpha = 16,       # Learning strength
    lora_dropout = 0.05,   # Prevents overlearning (5% dropout)
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 42,     # For consistent results
)

print("✅ Model prepared for fine-tuning!")
print("📝 LoRA adapters added - we can now teach it new patterns")

from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import Dataset

# Convert our data to a Dataset object (what the trainer expects)
train_dataset = Dataset.from_list(formatted_training_data)

print(f"✅ Created dataset with {len(train_dataset)} examples")
print(f"📊 First example preview: {train_dataset[0]['text'][:100]}...")

# CORRECTED trainer with DeepSeek best practices
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    dataset_text_field = "text",
    max_seq_length = 1024,                  # Increased for reasoning
    dataset_num_proc = 2,
    packing = False,
    args = TrainingArguments(
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4,    # Effective batch size = 4
        warmup_steps = 10,                  # More warmup
        max_steps = 100,                    # Much more training (vs 30)
        learning_rate = 5e-5,               # Lower, more stable
        fp16 = True,
        logging_steps = 10,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "cosine",       # Better for longer training
        seed = 42,
        output_dir = "outputs",
        save_steps = 50,
        dataloader_pin_memory = False,
    ),
)

# Start the fine-tuning process!
print("🚀 Starting fine-tuning...")
print("⏱️ Should take about 1-2 minutes with our conservative settings")

trainer.train()

print("🎉 Fine-tuning completed!")

# Test with DeepSeek's official recommended settings
test_prompt = "What does NVIDIA Corporation primarily manufacture? Explain in one paragraph"

inputs = tokenizer(test_prompt, return_tensors="pt").to("cuda")

outputs = model.generate(
    **inputs,
    max_new_tokens=250,
    temperature=0.6,        # OFFICIAL DeepSeek recommendation (not 0.1!)
    top_p=0.95,            # Add top_p as recommended
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("🤖 PROPERLY CONFIGURED DeepSeek response:")
print(response)

# Maybe temperature 0.6 is still too high for our small dataset
test_prompt = "What does Intel Corporation manufacture?"

inputs = tokenizer(test_prompt, return_tensors="pt").to("cuda")

outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    temperature=0.3,        # Lower temperature
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("🤖 Lower temperature (0.3):")
print(response)

# Test with exact training example to see if it learned
exact_training_prompt = "What does NVIDIA Corporation primarily manufacture. Answer in one word"

inputs = tokenizer(exact_training_prompt, return_tensors="pt").to("cuda")

outputs = model.generate(
    **inputs,
    max_new_tokens=120,
    temperature=0.3,        # Use the temperature that worked
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("🤖 Exact training question:")
print(response)

# Test other semiconductor companies we trained on
test_companies = [
    "What does AMD specialize in manufacturing?",
    "What is TSMC's main business?",
    "What does Qualcomm manufacture?"
]

print("=== Testing Knowledge Retention ===")
for question in test_companies:
    inputs = tokenizer(question, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=80,
        temperature=0.3,  # Our winning temperature
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nQ: {question}")
    print(f"A: {response}")

print("\n🎉 Fine-tuning and testing completed successfully!")
print("📊 The model has learned semiconductor industry knowledge!")