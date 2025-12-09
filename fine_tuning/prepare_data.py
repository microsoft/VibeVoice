#!/usr/bin/env python3
"""
Data Preparation Script for VEMI AI Fine-Tuning
================================================

Prepares datasets for Medical and Automobile domain fine-tuning.

Usage:
    python prepare_data.py --domain medical
    python prepare_data.py --domain automobile
    python prepare_data.py --domain all
"""

import os
import json
import argparse
import random
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm

# Create directories
DATA_DIR = Path(__file__).parent / "data"
MEDICAL_DIR = DATA_DIR / "medical"
AUTOMOBILE_DIR = DATA_DIR / "automobile"

for dir_path in [MEDICAL_DIR, AUTOMOBILE_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


def format_qa_pair(question: str, answer: str, domain: str) -> Dict[str, str]:
    """Format Q&A pair for instruction fine-tuning"""
    
    if domain == "medical":
        system = "You are VEMI AI Medical Assistant, a helpful healthcare assistant. Provide accurate, evidence-based medical information. Always recommend consulting a healthcare professional for serious concerns."
    else:
        system = "You are VEMI AI Automobile Assistant, an expert automotive technician. Provide accurate car repair advice, diagnostic help, and maintenance guidance. Always prioritize safety."
    
    return {
        "instruction": question.strip(),
        "input": "",
        "output": answer.strip(),
        "system": system
    }


def prepare_medical_data() -> List[Dict]:
    """
    Prepare medical Q&A datasets from HuggingFace.
    
    Sources:
    - lavita/MedQuAD (47K pairs)
    - pubmed_qa (PubMedQA)
    - medmcqa (Medical MCQ)
    - lavita/ChatDoctor-HealthCareMagic-100k
    """
    print("\n" + "="*60)
    print("Preparing Medical Dataset")
    print("="*60)
    
    from datasets import load_dataset
    
    all_data = []
    
    # 1. MedQuAD - High quality medical Q&A
    print("\n[1/4] Loading MedQuAD...")
    try:
        medquad = load_dataset("lavita/MedQuAD", split="train")
        for item in tqdm(medquad, desc="Processing MedQuAD"):
            if item.get("Question") and item.get("Answer"):
                all_data.append(format_qa_pair(
                    item["Question"],
                    item["Answer"],
                    "medical"
                ))
        print(f"  ✓ MedQuAD: {len(medquad)} pairs")
    except Exception as e:
        print(f"  ✗ MedQuAD failed: {e}")
    
    # 2. PubMedQA - Research-based Q&A
    print("\n[2/4] Loading PubMedQA...")
    try:
        pubmedqa = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
        for item in tqdm(pubmedqa, desc="Processing PubMedQA"):
            question = item.get("question", "")
            # Combine context with answer
            long_answer = item.get("long_answer", "")
            final_decision = item.get("final_decision", "")
            answer = f"{long_answer}\n\nConclusion: {final_decision}" if final_decision else long_answer
            
            if question and answer:
                all_data.append(format_qa_pair(question, answer, "medical"))
        print(f"  ✓ PubMedQA: {len(pubmedqa)} pairs")
    except Exception as e:
        print(f"  ✗ PubMedQA failed: {e}")
    
    # 3. MedMCQA - Medical exam questions (converted to Q&A)
    print("\n[3/4] Loading MedMCQA...")
    try:
        medmcqa = load_dataset("openlifescienceai/medmcqa", split="train")
        # Sample subset for balance
        medmcqa_sample = medmcqa.shuffle(seed=42).select(range(min(10000, len(medmcqa))))
        
        for item in tqdm(medmcqa_sample, desc="Processing MedMCQA"):
            question = item.get("question", "")
            options = [item.get("opa", ""), item.get("opb", ""), item.get("opc", ""), item.get("opd", "")]
            correct_idx = item.get("cop", 0)
            explanation = item.get("exp", "")
            
            if question and options[correct_idx]:
                answer = f"The correct answer is: {options[correct_idx]}"
                if explanation:
                    answer += f"\n\nExplanation: {explanation}"
                all_data.append(format_qa_pair(question, answer, "medical"))
        print(f"  ✓ MedMCQA: {len(medmcqa_sample)} pairs")
    except Exception as e:
        print(f"  ✗ MedMCQA failed: {e}")
    
    # 4. ChatDoctor/HealthCareMagic - Patient-doctor conversations
    print("\n[4/4] Loading HealthCareMagic...")
    try:
        healthcaremagic = load_dataset("lavita/ChatDoctor-HealthCareMagic-100k", split="train")
        # Sample for balance
        hcm_sample = healthcaremagic.shuffle(seed=42).select(range(min(15000, len(healthcaremagic))))
        
        for item in tqdm(hcm_sample, desc="Processing HealthCareMagic"):
            question = item.get("input", "")
            answer = item.get("output", "")
            
            if question and answer:
                all_data.append(format_qa_pair(question, answer, "medical"))
        print(f"  ✓ HealthCareMagic: {len(hcm_sample)} pairs")
    except Exception as e:
        print(f"  ✗ HealthCareMagic failed: {e}")
    
    # Shuffle and deduplicate
    print("\nProcessing final dataset...")
    random.shuffle(all_data)
    
    # Remove duplicates based on instruction
    seen = set()
    unique_data = []
    for item in all_data:
        key = item["instruction"][:100]  # First 100 chars as key
        if key not in seen:
            seen.add(key)
            unique_data.append(item)
    
    print(f"\n✓ Total Medical Q&A pairs: {len(unique_data)}")
    
    # Save
    output_file = MEDICAL_DIR / "medical_qa.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(unique_data, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved to: {output_file}")
    
    return unique_data


def prepare_automobile_data() -> List[Dict]:
    """
    Prepare automobile Q&A datasets.
    
    Uses hybrid approach:
    1. AutoAIQnA from Kaggle (if available)
    2. Synthetic Q&A from car specifications
    3. OBD-II diagnostic codes
    4. Common car repair Q&A
    """
    print("\n" + "="*60)
    print("Preparing Automobile Dataset")
    print("="*60)
    
    all_data = []
    
    # 1. Synthetic Car Specifications Q&A
    print("\n[1/4] Generating Car Specifications Q&A...")
    car_specs_qa = generate_car_specs_qa()
    all_data.extend(car_specs_qa)
    print(f"  ✓ Car Specs Q&A: {len(car_specs_qa)} pairs")
    
    # 2. OBD-II Diagnostic Codes Q&A
    print("\n[2/4] Generating OBD-II Codes Q&A...")
    obd_qa = generate_obd_codes_qa()
    all_data.extend(obd_qa)
    print(f"  ✓ OBD-II Q&A: {len(obd_qa)} pairs")
    
    # 3. Common Car Repair Q&A
    print("\n[3/4] Generating Common Repair Q&A...")
    repair_qa = generate_repair_qa()
    all_data.extend(repair_qa)
    print(f"  ✓ Repair Q&A: {len(repair_qa)} pairs")
    
    # 4. Car Maintenance Q&A
    print("\n[4/4] Generating Maintenance Q&A...")
    maintenance_qa = generate_maintenance_qa()
    all_data.extend(maintenance_qa)
    print(f"  ✓ Maintenance Q&A: {len(maintenance_qa)} pairs")
    
    # Shuffle
    random.shuffle(all_data)
    
    print(f"\n✓ Total Automobile Q&A pairs: {len(all_data)}")
    
    # Save
    output_file = AUTOMOBILE_DIR / "automobile_qa.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved to: {output_file}")
    
    return all_data


def generate_car_specs_qa() -> List[Dict]:
    """Generate Q&A pairs from car specifications"""
    
    # Sample car database (in production, load from actual database)
    cars = [
        {"make": "Toyota", "model": "Camry", "year": 2024, "engine": "2.5L 4-cylinder", "hp": 203, "mpg_city": 28, "mpg_hwy": 39, "transmission": "8-speed automatic"},
        {"make": "Honda", "model": "Civic", "year": 2024, "engine": "2.0L 4-cylinder", "hp": 158, "mpg_city": 31, "mpg_hwy": 40, "transmission": "CVT"},
        {"make": "Ford", "model": "F-150", "year": 2024, "engine": "3.5L V6 EcoBoost", "hp": 400, "mpg_city": 18, "mpg_hwy": 24, "transmission": "10-speed automatic"},
        {"make": "Chevrolet", "model": "Silverado", "year": 2024, "engine": "5.3L V8", "hp": 355, "mpg_city": 16, "mpg_hwy": 22, "transmission": "8-speed automatic"},
        {"make": "Tesla", "model": "Model 3", "year": 2024, "engine": "Electric Motor", "hp": 283, "mpg_city": 138, "mpg_hwy": 126, "transmission": "Single-speed"},
        {"make": "BMW", "model": "3 Series", "year": 2024, "engine": "2.0L Turbo 4-cylinder", "hp": 255, "mpg_city": 26, "mpg_hwy": 36, "transmission": "8-speed automatic"},
        {"make": "Mercedes-Benz", "model": "C-Class", "year": 2024, "engine": "2.0L Turbo 4-cylinder", "hp": 255, "mpg_city": 23, "mpg_hwy": 33, "transmission": "9-speed automatic"},
        {"make": "Audi", "model": "A4", "year": 2024, "engine": "2.0L TFSI", "hp": 201, "mpg_city": 25, "mpg_hwy": 34, "transmission": "7-speed S tronic"},
        {"make": "Hyundai", "model": "Elantra", "year": 2024, "engine": "2.0L 4-cylinder", "hp": 147, "mpg_city": 33, "mpg_hwy": 43, "transmission": "IVT"},
        {"make": "Kia", "model": "Sportage", "year": 2024, "engine": "2.5L 4-cylinder", "hp": 187, "mpg_city": 25, "mpg_hwy": 32, "transmission": "8-speed automatic"},
    ]
    
    qa_pairs = []
    
    templates = [
        ("What are the specifications of the {year} {make} {model}?",
         "The {year} {make} {model} features a {engine} engine producing {hp} horsepower. It comes with a {transmission} transmission and achieves {mpg_city} MPG in the city and {mpg_hwy} MPG on the highway."),
        ("How much horsepower does the {year} {make} {model} have?",
         "The {year} {make} {model} produces {hp} horsepower from its {engine} engine."),
        ("What is the fuel economy of the {year} {make} {model}?",
         "The {year} {make} {model} achieves {mpg_city} MPG in city driving and {mpg_hwy} MPG on the highway."),
        ("What transmission does the {year} {make} {model} use?",
         "The {year} {make} {model} is equipped with a {transmission} transmission."),
        ("What engine is in the {year} {make} {model}?",
         "The {year} {make} {model} is powered by a {engine} engine that produces {hp} horsepower."),
    ]
    
    for car in cars:
        for q_template, a_template in templates:
            question = q_template.format(**car)
            answer = a_template.format(**car)
            qa_pairs.append(format_qa_pair(question, answer, "automobile"))
    
    return qa_pairs


def generate_obd_codes_qa() -> List[Dict]:
    """Generate Q&A pairs for OBD-II diagnostic codes"""
    
    obd_codes = [
        {"code": "P0300", "desc": "Random/Multiple Cylinder Misfire Detected", "cause": "faulty spark plugs, ignition coils, fuel injectors, or vacuum leaks", "fix": "Inspect and replace spark plugs, check ignition coils, verify fuel pressure, and check for vacuum leaks."},
        {"code": "P0171", "desc": "System Too Lean (Bank 1)", "cause": "vacuum leak, faulty MAF sensor, clogged fuel filter, or weak fuel pump", "fix": "Check for vacuum leaks, clean or replace MAF sensor, replace fuel filter, and test fuel pump pressure."},
        {"code": "P0420", "desc": "Catalyst System Efficiency Below Threshold", "cause": "failing catalytic converter, oxygen sensor issues, or exhaust leaks", "fix": "Test oxygen sensors, check for exhaust leaks, and replace catalytic converter if confirmed failed."},
        {"code": "P0442", "desc": "Evaporative Emission System Leak Detected (Small Leak)", "cause": "loose gas cap, cracked EVAP hose, or faulty purge valve", "fix": "Tighten or replace gas cap, inspect EVAP hoses for cracks, and test purge valve."},
        {"code": "P0455", "desc": "Evaporative Emission System Leak Detected (Large Leak)", "cause": "missing gas cap, disconnected EVAP hose, or faulty charcoal canister", "fix": "Check gas cap, inspect all EVAP system connections, and test charcoal canister."},
        {"code": "P0128", "desc": "Coolant Thermostat Below Regulating Temperature", "cause": "stuck open thermostat, low coolant level, or faulty coolant temperature sensor", "fix": "Replace thermostat, check coolant level, and test coolant temperature sensor."},
        {"code": "P0401", "desc": "Exhaust Gas Recirculation Flow Insufficient", "cause": "clogged EGR passages, faulty EGR valve, or carbon buildup", "fix": "Clean EGR passages, test and replace EGR valve if faulty, and decarbonize intake manifold."},
        {"code": "P0507", "desc": "Idle Air Control System RPM Higher Than Expected", "cause": "vacuum leak, dirty throttle body, or faulty IAC valve", "fix": "Check for vacuum leaks, clean throttle body, and test or replace IAC valve."},
        {"code": "P0700", "desc": "Transmission Control System Malfunction", "cause": "transmission fluid issues, faulty solenoids, or TCM failure", "fix": "Check transmission fluid level and condition, scan for specific transmission codes, and inspect solenoids."},
        {"code": "P0016", "desc": "Crankshaft/Camshaft Position Correlation Bank 1 Sensor A", "cause": "timing chain stretch, faulty VVT solenoid, or sensor issues", "fix": "Inspect timing chain tension, test VVT solenoid, and check crank/cam sensors."},
    ]
    
    qa_pairs = []
    
    for code_info in obd_codes:
        # What does code mean?
        qa_pairs.append(format_qa_pair(
            f"What does OBD code {code_info['code']} mean?",
            f"OBD-II code {code_info['code']} indicates: {code_info['desc']}. Common causes include {code_info['cause']}.",
            "automobile"
        ))
        
        # How to fix?
        qa_pairs.append(format_qa_pair(
            f"How do I fix OBD code {code_info['code']}?",
            f"To fix code {code_info['code']} ({code_info['desc']}): {code_info['fix']}",
            "automobile"
        ))
        
        # What causes?
        qa_pairs.append(format_qa_pair(
            f"What causes OBD code {code_info['code']}?",
            f"Code {code_info['code']} ({code_info['desc']}) is typically caused by {code_info['cause']}.",
            "automobile"
        ))
    
    return qa_pairs


def generate_repair_qa() -> List[Dict]:
    """Generate common car repair Q&A pairs"""
    
    repairs = [
        {
            "q": "How do I change my car's oil?",
            "a": "To change your car's oil: 1) Warm up the engine for 2-3 minutes. 2) Lift the car safely using jack stands. 3) Place a drain pan under the oil pan. 4) Remove the drain plug and let oil drain completely. 5) Replace the drain plug with a new washer. 6) Remove and replace the oil filter. 7) Add the correct type and amount of new oil. 8) Check the oil level with the dipstick. 9) Start the engine and check for leaks."
        },
        {
            "q": "How often should I change my brake pads?",
            "a": "Brake pads typically last 30,000 to 70,000 miles depending on driving habits, vehicle type, and pad quality. Signs you need new brake pads include: squealing or squeaking noises, grinding sounds, longer stopping distances, brake pedal vibration, or visible wear indicators. Most brake pads have a minimum thickness of 3-4mm before replacement is needed."
        },
        {
            "q": "Why is my car overheating?",
            "a": "Common causes of car overheating include: 1) Low coolant level - check for leaks in hoses, radiator, or water pump. 2) Faulty thermostat - stuck closed prevents coolant flow. 3) Broken water pump - listen for bearing noise. 4) Clogged radiator - debris blocking airflow. 5) Failed radiator fan - check if it turns on at temperature. 6) Blown head gasket - white smoke from exhaust, coolant in oil. Pull over immediately if overheating to prevent engine damage."
        },
        {
            "q": "How do I jump start a car?",
            "a": "To jump start a car: 1) Position cars close but not touching. 2) Turn off both vehicles. 3) Connect red cable to dead battery positive (+). 4) Connect other red end to good battery positive (+). 5) Connect black cable to good battery negative (-). 6) Connect other black end to unpainted metal in dead car (ground). 7) Start the working car. 8) Wait 2-3 minutes. 9) Try starting the dead car. 10) Remove cables in reverse order. Drive for at least 20 minutes to charge the battery."
        },
        {
            "q": "What does it mean when my check engine light is on?",
            "a": "A check engine light indicates the engine control module (ECM) has detected a problem. It could be minor (loose gas cap) or serious (catalytic converter failure). Steps to take: 1) Check if the gas cap is tight. 2) Note if the light is steady (less urgent) or flashing (serious - reduce speed immediately). 3) Have the codes scanned with an OBD-II reader - most auto parts stores do this free. 4) Address the specific code with appropriate repairs. Don't ignore it as it can affect fuel economy and emissions."
        },
        {
            "q": "How do I check my car's tire pressure?",
            "a": "To check tire pressure: 1) Find the recommended PSI on the driver's door jamb sticker or owner's manual (not on the tire sidewall - that's maximum). 2) Check tires when cold (not driven for 3+ hours). 3) Remove the valve cap. 4) Press a tire gauge firmly onto the valve stem. 5) Read the pressure. 6) Add or release air as needed. 7) Replace the valve cap. Check monthly and before long trips. Proper pressure improves fuel economy, handling, and tire life."
        },
        {
            "q": "Why is my car making a squealing noise when I start it?",
            "a": "A squealing noise at startup is usually caused by: 1) Worn or loose serpentine/drive belt - most common cause, especially in cold weather. 2) Failing belt tensioner - can't maintain proper belt tension. 3) Misaligned pulleys - belt slipping on components. 4) Worn water pump or alternator bearings. Check the belt for cracks, glazing, or looseness. A belt dressing can temporarily quiet it, but replacement is the proper fix. Most belts last 60,000-100,000 miles."
        },
        {
            "q": "How do I change a flat tire?",
            "a": "To change a flat tire: 1) Pull to a safe, flat location and turn on hazards. 2) Apply parking brake. 3) Place wheel wedges behind tires. 4) Remove hubcap and loosen lug nuts (don't remove yet). 5) Place jack under vehicle frame. 6) Raise vehicle until flat tire is 6 inches off ground. 7) Remove lug nuts and flat tire. 8) Mount spare tire. 9) Hand-tighten lug nuts in a star pattern. 10) Lower vehicle and tighten lug nuts fully (80-100 ft-lbs). 11) Check spare tire pressure. Drive carefully - most spares are limited to 50 mph."
        },
    ]
    
    qa_pairs = []
    for item in repairs:
        qa_pairs.append(format_qa_pair(item["q"], item["a"], "automobile"))
    
    # Add variations
    variations = [
        ("My car won't start, what should I check?", "If your car won't start, check: 1) Battery - test voltage (should be 12.6V+), check terminals for corrosion. 2) Starter - listen for clicking sound. 3) Fuel - ensure you have gas, check fuel pump (turn key to ON, listen for hum). 4) Ignition - try starting in neutral (automatic) or with clutch pressed (manual). 5) Immobilizer - make sure you're using the correct key. 6) Check engine light codes if the car cranks but won't start."),
        ("What's the difference between synthetic and conventional oil?", "Synthetic oil is chemically engineered for better performance: 1) Better high/low temperature stability. 2) Longer change intervals (7,500-15,000 miles vs 3,000-5,000). 3) Improved engine protection and cleanliness. 4) Better fuel economy. 5) Higher cost ($6-12/qt vs $3-5/qt). Conventional oil is refined from crude oil and works fine for most vehicles. Check your owner's manual - some engines require synthetic."),
        ("How do I know if my battery is dying?", "Signs of a dying car battery: 1) Slow engine crank when starting. 2) Dim headlights, especially at idle. 3) Electrical issues (power windows slow, radio resets). 4) Check engine or battery warning light. 5) Swollen battery case. 6) Bad smell (sulfur/rotten eggs). 7) Battery is 3-5 years old. Test with a multimeter: 12.6V or higher = good, 12.4V = 75% charged, 12.2V = 50% charged, below 12V = discharged or failing."),
        ("What causes my steering wheel to shake?", "Steering wheel shaking causes vary by speed: At highway speeds (55+ mph): 1) Unbalanced tires - most common. 2) Bent wheel rim. 3) Worn tires with flat spots. At low speeds or braking: 1) Warped brake rotors - shake when braking. 2) Stuck brake caliper. While turning: 1) Worn CV joints or axle. 2) Worn tie rod ends. 3) Bad wheel bearing. Get tires balanced first - it's the cheapest fix and most common cause."),
    ]
    
    for q, a in variations:
        qa_pairs.append(format_qa_pair(q, a, "automobile"))
    
    return qa_pairs


def generate_maintenance_qa() -> List[Dict]:
    """Generate car maintenance Q&A pairs"""
    
    maintenance = [
        ("How often should I change my oil?", "Oil change intervals depend on oil type and driving conditions: Conventional oil: every 3,000-5,000 miles. Synthetic blend: every 5,000-7,500 miles. Full synthetic: every 7,500-15,000 miles. Severe conditions (frequent short trips, dusty areas, towing) may require more frequent changes. Always follow your owner's manual recommendations. Modern cars often have oil life monitors that track actual driving conditions."),
        ("When should I replace my timing belt?", "Timing belts typically need replacement between 60,000-100,000 miles depending on your vehicle. Check your owner's manual for exact intervals. Signs of a failing timing belt include: ticking noise from engine, engine misfires, oil leaking near motor, engine won't turn over. Timing belt failure can cause severe engine damage on interference engines - don't skip this maintenance. Cost is usually $500-1,000 including water pump replacement."),
        ("How often should I rotate my tires?", "Rotate tires every 5,000-7,500 miles or with every oil change. Regular rotation ensures even wear and extends tire life. Front tires typically wear faster due to steering and weight. Rotation patterns vary: Front-to-rear (same side), X-pattern (diagonal swap), or specific patterns for directional tires. Check your owner's manual for recommended pattern."),
        ("When should I replace spark plugs?", "Spark plug replacement intervals: Copper plugs: 30,000 miles. Platinum plugs: 60,000 miles. Iridium plugs: 100,000+ miles. Signs of worn plugs include: rough idle, poor acceleration, decreased fuel economy, hard starting, and engine misfires. Always replace with the correct type specified in your manual. While replacing, inspect plug wires or coil boots for wear."),
        ("How do I maintain my car battery?", "Battery maintenance tips: 1) Keep terminals clean - remove corrosion with baking soda and water. 2) Ensure tight connections. 3) Check fluid level in non-sealed batteries. 4) Test battery annually after 3 years. 5) Drive regularly - short trips don't fully charge battery. 6) Avoid draining with accessories when engine off. 7) Park in moderate temperatures when possible. Average battery life is 3-5 years."),
        ("What fluids should I check regularly?", "Check these fluids monthly: 1) Engine oil - use dipstick, should be between marks. 2) Coolant - check overflow tank when cold. 3) Brake fluid - check reservoir, should be between MIN/MAX. 4) Power steering fluid - check reservoir or dipstick. 5) Transmission fluid - check with engine warm, in Park. 6) Windshield washer fluid - keep full for visibility. Low levels may indicate leaks - investigate before topping off."),
        ("How often should I replace my air filter?", "Engine air filter: Replace every 15,000-30,000 miles or annually. Cabin air filter: Replace every 15,000-25,000 miles. Check more frequently in dusty conditions. Signs of a dirty engine filter: reduced acceleration, black exhaust smoke, decreased fuel economy, and check engine light. A clogged cabin filter causes weak AC/heat, musty odors, and foggy windows. Both are easy DIY replacements."),
        ("When should I flush my coolant?", "Coolant (antifreeze) should be flushed every 30,000-60,000 miles or every 5 years, whichever comes first. Some long-life coolants last 100,000+ miles. Signs you need a flush: rusty or discolored coolant, debris or particles visible, overheating issues, or sweet smell from engine. Always use the correct coolant type for your vehicle - mixing types can cause damage. Never open the radiator cap when hot."),
    ]
    
    qa_pairs = []
    for q, a in maintenance:
        qa_pairs.append(format_qa_pair(q, a, "automobile"))
    
    return qa_pairs


def main():
    parser = argparse.ArgumentParser(description="Prepare datasets for VEMI AI fine-tuning")
    parser.add_argument("--domain", type=str, choices=["medical", "automobile", "all"], 
                        default="all", help="Domain to prepare data for")
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("VEMI AI Data Preparation")
    print("="*60)
    
    if args.domain in ["medical", "all"]:
        medical_data = prepare_medical_data()
        print(f"\n✓ Medical dataset ready: {len(medical_data)} examples")
    
    if args.domain in ["automobile", "all"]:
        auto_data = prepare_automobile_data()
        print(f"\n✓ Automobile dataset ready: {len(auto_data)} examples")
    
    print("\n" + "="*60)
    print("Data preparation complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Run: python train_medical.py")
    print("2. Run: python train_automobile.py")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
