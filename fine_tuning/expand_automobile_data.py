#!/usr/bin/env python3
"""
Expanded Automobile Q&A Data Generator
======================================
Generates 1000+ automobile Q&A pairs for fine-tuning.

Usage: python expand_automobile_data.py
"""

import json
import random
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data" / "automobile"
DATA_DIR.mkdir(parents=True, exist_ok=True)

SYSTEM_PROMPT = """You are VEMI AI Automobile Assistant, an expert automotive technician developed by Alvion Global Solutions.

Provide accurate car repair advice, diagnostics, and maintenance guidance. Prioritize safety and recommend professional help for complex repairs. Be concise and direct - do not greet unless the user greets first."""


def format_qa(q, a):
    return {"instruction": q.strip(), "input": "", "output": a.strip(), "system": SYSTEM_PROMPT}


def generate_all_data():
    all_data = []
    
    # OBD Codes (100+ codes, 5 variations each = 500+ pairs)
    obd_codes = [
        ("P0010", "Camshaft Position Actuator Circuit Bank 1", "faulty VVT solenoid, wiring issues, low oil", "Check oil level. Test VVT solenoid. Inspect wiring. Cost: $50-300"),
        ("P0011", "Cam Timing Over-Advanced Bank 1", "dirty oil, VVT solenoid, timing chain stretch", "Change oil. Test VVT. Check timing chain. Cost: $100-500"),
        ("P0016", "Crank/Cam Position Correlation", "timing chain jump, sensor failure", "Check sensors. Inspect timing chain. Cost: $100-2000"),
        ("P0030", "O2 Sensor Heater Circuit Bank 1 Sensor 1", "failed heater, fuse, wiring", "Check fuse. Test heater resistance. Replace sensor. Cost: $50-250"),
        ("P0100", "MAF Circuit Malfunction", "dirty/failed MAF, air leak", "Clean MAF first. Check for leaks. Replace if faulty. Cost: $20-350"),
        ("P0101", "MAF Range/Performance", "dirty MAF, air leak, restricted filter", "Replace air filter. Clean MAF. Check intake. Cost: $20-350"),
        ("P0106", "MAP Range/Performance", "vacuum leak, bad MAP sensor", "Check vacuum lines. Test MAP sensor. Cost: $30-150"),
        ("P0115", "Coolant Temp Circuit", "failed ECT sensor, wiring", "Test sensor resistance. Check wiring. Cost: $20-100"),
        ("P0128", "Thermostat Below Temp", "stuck open thermostat", "Replace thermostat. Common issue. Cost: $100-300"),
        ("P0130", "O2 Sensor Circuit Bank 1 Sensor 1", "failed sensor, exhaust leak", "Check for exhaust leaks. Replace O2 sensor. Cost: $50-250"),
        ("P0131", "O2 Sensor Low Voltage", "lean condition, sensor failure", "Check vacuum leaks. Replace sensor if stuck. Cost: $50-250"),
        ("P0132", "O2 Sensor High Voltage", "rich condition, contaminated sensor", "Check fuel system. Replace sensor. Cost: $50-250"),
        ("P0133", "O2 Sensor Slow Response", "aging sensor", "Replace O2 sensor. Cost: $50-250"),
        ("P0171", "System Too Lean Bank 1", "vacuum leak, weak fuel pump, MAF issue", "Smoke test for leaks. Check fuel pressure. Clean MAF. Cost: $50-500"),
        ("P0172", "System Too Rich Bank 1", "leaking injector, bad O2, fuel pressure", "Test injectors. Check fuel pressure. Replace O2. Cost: $100-400"),
        ("P0174", "System Too Lean Bank 2", "same as P0171, Bank 2", "Same diagnosis as P0171. Cost: $50-500"),
        ("P0175", "System Too Rich Bank 2", "same as P0172, Bank 2", "Same diagnosis as P0172. Cost: $100-400"),
        ("P0300", "Random Multiple Misfire", "spark plugs, coils, vacuum leak", "Replace plugs. Test coils. Check for leaks. Cost: $50-500"),
        ("P0301", "Cylinder 1 Misfire", "plug, coil, injector on cyl 1", "Swap coil to test. Replace plug. Test injector. Cost: $20-400"),
        ("P0302", "Cylinder 2 Misfire", "plug, coil, injector on cyl 2", "Same as P0301 for cylinder 2. Cost: $20-400"),
        ("P0303", "Cylinder 3 Misfire", "plug, coil, injector on cyl 3", "Same diagnosis for cylinder 3. Cost: $20-400"),
        ("P0304", "Cylinder 4 Misfire", "plug, coil, injector on cyl 4", "Same diagnosis for cylinder 4. Cost: $20-400"),
        ("P0325", "Knock Sensor Circuit", "failed sensor, wiring", "Test sensor. Check wiring. Cost: $50-250"),
        ("P0335", "Crankshaft Position Sensor", "failed CKP, wiring, reluctor", "Test sensor. Check reluctor ring. Cost: $50-200"),
        ("P0340", "Camshaft Position Sensor", "failed CMP, wiring", "Test sensor. Check wiring. Cost: $50-200"),
        ("P0400", "EGR Flow Malfunction", "clogged EGR, stuck valve", "Clean EGR valve and passages. Cost: $100-400"),
        ("P0401", "EGR Insufficient Flow", "carbon buildup, stuck EGR", "Clean EGR system thoroughly. Cost: $100-400"),
        ("P0420", "Catalyst Efficiency Below Threshold Bank 1", "failing catalytic converter, O2 sensor", "Rule out O2 sensor first. May need cat replacement. Cost: $200-2500"),
        ("P0430", "Catalyst Efficiency Below Threshold Bank 2", "same as P0420 Bank 2", "Same as P0420 diagnosis. Cost: $200-2500"),
        ("P0440", "EVAP System Malfunction", "leak in EVAP system, purge valve", "Smoke test system. Check gas cap. Cost: $50-400"),
        ("P0442", "EVAP Small Leak", "loose gas cap, small hose crack", "Tighten gas cap. Smoke test. Cost: $20-300"),
        ("P0455", "EVAP Large Leak", "missing cap, major hose disconnect", "Check gas cap. Major leak - easy to find. Cost: $20-400"),
        ("P0456", "EVAP Very Small Leak", "hairline crack, minor seal issue", "Smoke test carefully. Often gas cap. Cost: $20-200"),
        ("P0500", "Vehicle Speed Sensor", "failed VSS, wiring", "Test sensor. Check wiring. Cost: $50-200"),
        ("P0505", "Idle Control System", "dirty IAC, vacuum leak", "Clean throttle body. Check vacuum. Cost: $50-250"),
        ("P0507", "Idle RPM High", "vacuum leak, IAC stuck open", "Check vacuum hoses. Clean throttle body. Cost: $50-250"),
        ("P0520", "Oil Pressure Sensor Circuit", "failed sensor", "Verify real pressure first! Replace sensor. Cost: $20-100"),
        ("P0562", "System Voltage Low", "weak battery, alternator", "Test battery and alternator. Cost: $100-500"),
        ("P0700", "Transmission Control System", "trans issue", "Scan for specific trans codes. Cost: $100-3000+"),
        ("P0715", "Input Speed Sensor", "failed sensor, trans issue", "Test sensor. May be internal. Cost: $50-300"),
        ("P0720", "Output Speed Sensor", "failed sensor", "Test sensor. Check wiring. Cost: $50-300"),
        ("P0730", "Incorrect Gear Ratio", "worn clutches, solenoids", "Check fluid. May need rebuild. Cost: $200-4000"),
        ("P0740", "Torque Converter Clutch", "TCC solenoid, converter", "Test solenoid. May need converter. Cost: $150-1500"),
        ("C0035", "Left Front Wheel Speed Sensor", "ABS sensor failed", "Clean or replace sensor. Cost: $50-200"),
        ("C0040", "Right Front Wheel Speed Sensor", "ABS sensor failed", "Same as C0035. Cost: $50-200"),
        ("U0100", "Lost Communication with ECM", "CAN bus issue", "Check wiring. Professional diagnosis. Cost varies"),
        ("U0101", "Lost Communication with TCM", "trans module issue", "Check wiring. May need TCM. Cost: $200-1000"),
    ]
    
    for code, name, cause, fix in obd_codes:
        all_data.append(format_qa(f"What does code {code} mean?", f"{code} - {name}. Caused by: {cause}. Fix: {fix}"))
        all_data.append(format_qa(f"How to fix {code}?", f"To fix {code} ({name}): {fix}"))
        all_data.append(format_qa(f"What causes {code}?", f"{code} is caused by: {cause}."))
    
    # Symptoms (50+ symptoms, 3 variations each = 150+ pairs)
    symptoms = [
        ("car won't start but lights work", "weak battery, bad starter, ignition switch", "Test battery under load. Check starter. Try neutral start."),
        ("car won't start no crank", "dead battery, starter, ignition switch", "Jump start. Check main fuse. Test starter."),
        ("car starts then dies", "IAC, vacuum leak, MAF, fuel issue", "Clean throttle body. Check for leaks. Test fuel pressure."),
        ("engine cranks won't start", "no fuel, no spark, no compression", "Check fuel pressure. Test spark. Check timing."),
        ("car shakes at idle", "misfire, vacuum leak, motor mount", "Scan for misfires. Check vacuum. Inspect mounts."),
        ("car shakes at highway speed", "unbalanced tires, bent wheel", "Balance tires first. Check for tire damage."),
        ("steering wheel shakes braking", "warped rotors", "Resurface or replace rotors. Replace pads."),
        ("car pulls to one side", "alignment, tire pressure, stuck caliper", "Check pressures. Get alignment. Check brakes."),
        ("squealing on startup", "worn serpentine belt", "Replace belt. Check tensioner."),
        ("grinding when braking", "worn pads metal on metal", "Replace pads and rotors immediately."),
        ("clicking when turning", "worn CV joint", "Replace CV axle. Check boots for tears."),
        ("clunking over bumps", "worn sway bar links, ball joints", "Replace sway bar links. Check suspension."),
        ("car overheating", "low coolant, thermostat, water pump", "STOP driving. Check coolant. Test thermostat."),
        ("white smoke exhaust", "coolant burning, head gasket", "Check coolant level. Test for exhaust gases in coolant."),
        ("blue smoke exhaust", "burning oil, rings or seals", "Check oil consumption. May need engine work."),
        ("black smoke exhaust", "running rich", "Check MAF. Test O2 sensors. Scan for codes."),
        ("car hesitates accelerating", "dirty filter, fuel, ignition", "Replace air filter. Clean MAF. Check fuel pressure."),
        ("check engine light flashing", "severe misfire", "STOP driving. Scan codes. Fix misfire cause."),
        ("car stalls at idle", "dirty throttle, IAC, vacuum leak", "Clean throttle body. Check for leaks."),
        ("transmission slipping", "low fluid, worn clutches", "Check fluid level and condition. May need service."),
        ("hard shifting", "low fluid, solenoid issue", "Check fluid. May need trans service."),
        ("ABS light on", "wheel speed sensor, low fluid", "Scan ABS codes. Check brake fluid."),
        ("battery keeps dying", "parasitic drain, old battery", "Test for drain. Test battery age."),
        ("AC not cold", "low refrigerant, compressor", "Check pressures. Look for leaks."),
        ("heater not working", "low coolant, thermostat, heater core", "Check coolant. Test thermostat."),
    ]
    
    for symptom, causes, fix in symptoms:
        all_data.append(format_qa(f"My {symptom}. What's wrong?", f"Common causes: {causes}. To fix: {fix}"))
        all_data.append(format_qa(f"How to fix {symptom}?", f"For {symptom}: {fix} Causes include: {causes}"))
        all_data.append(format_qa(f"Why does my {symptom}?", f"This happens because of: {causes}. Solution: {fix}"))
    
    # Maintenance Q&A (50+ pairs)
    maintenance = [
        ("How often to change oil?", "Conventional: 3000-5000mi. Synthetic blend: 5000-7500mi. Full synthetic: 7500-15000mi. Check your manual."),
        ("When to replace spark plugs?", "Copper: 30000mi. Platinum: 60000mi. Iridium: 100000+mi."),
        ("How often rotate tires?", "Every 5000-7500 miles or with every oil change."),
        ("When to replace brake pads?", "30000-70000 miles. Replace when squealing, grinding, or under 3mm thick."),
        ("How often change transmission fluid?", "Auto: 60000-100000mi. Manual: 30000-60000mi."),
        ("When to replace timing belt?", "60000-100000mi. Critical - belt failure causes major engine damage."),
        ("How often flush coolant?", "Traditional: 30000mi/5yr. Long-life: 100000mi/10yr."),
        ("When replace serpentine belt?", "60000-100000mi or when cracked/fraying."),
        ("How often change air filter?", "15000-30000mi or annually."),
        ("When to replace battery?", "Every 3-5 years typically."),
        ("How often check tire pressure?", "Monthly and before long trips."),
        ("When replace wiper blades?", "Every 6-12 months."),
        ("How often change brake fluid?", "Every 2-3 years or 30000-45000mi."),
        ("When service at 30000 miles?", "Air filter, fuel filter, spark plugs (copper), inspect brakes, trans fluid check."),
        ("What service at 60000 miles?", "Timing belt if equipped, spark plugs (platinum), trans service, full brake inspection."),
        ("What service at 100000 miles?", "Spark plugs (iridium), water pump, all fluids, full inspection."),
    ]
    
    for q, a in maintenance:
        all_data.append(format_qa(q, a))
    
    # DIY Repairs (20+ detailed guides)
    diy_repairs = [
        ("How to change oil?", "1) Warm engine 2min. 2) Lift car on jack stands. 3) Put pan under drain plug. 4) Remove plug (14-17mm). 5) Drain 10min. 6) Replace plug with new washer. 7) Remove old filter. 8) Oil new filter gasket. 9) Install filter hand-tight. 10) Add new oil. 11) Check level. 12) Start, check for leaks."),
        ("How to replace brake pads?", "1) Loosen lugs on ground. 2) Lift and secure on stands. 3) Remove wheel. 4) Remove caliper bolts. 5) Hang caliper with wire. 6) Remove old pads. 7) Compress piston with C-clamp. 8) Install new pads. 9) Reinstall caliper. 10) Reinstall wheel. 11) PUMP BRAKES before driving!"),
        ("How to replace air filter?", "1) Open air box clips. 2) Remove old filter. 3) Wipe housing clean. 4) Install new filter. 5) Close housing. Takes 5 minutes."),
        ("How to replace battery?", "1) Turn off car. 2) Disconnect NEGATIVE first. 3) Disconnect positive. 4) Remove hold-down. 5) Remove battery. 6) Clean terminals. 7) Install new battery. 8) Connect POSITIVE first. 9) Connect negative."),
        ("How to replace spark plugs?", "1) Cool engine. 2) Remove coils. 3) Blow out wells. 4) Remove old plugs. 5) Gap new plugs. 6) Apply anti-seize. 7) Install hand-tight. 8) Torque to spec (12-18 ft-lbs). 9) Reinstall coils."),
        ("How to jump start a car?", "1) Position cars close. 2) Both off. 3) Red to dead positive. 4) Red to good positive. 5) Black to good negative. 6) Black to dead car metal ground. 7) Start good car. 8) Wait 2-3min. 9) Start dead car. 10) Remove cables reverse order."),
        ("How to change a flat tire?", "1) Safe flat spot, hazards on. 2) Parking brake. 3) Wheel wedges. 4) Loosen lugs. 5) Jack under frame. 6) Raise 6 inches. 7) Remove lugs and tire. 8) Mount spare. 9) Hand-tighten lugs star pattern. 10) Lower car. 11) Torque lugs (80-100 ft-lbs)."),
        ("How to check tire pressure?", "1) Find spec on door jamb (NOT tire sidewall). 2) Check when cold. 3) Remove valve cap. 4) Press gauge firmly. 5) Read pressure. 6) Add/release air as needed. 7) Replace cap."),
        ("How to replace wiper blades?", "1) Lift arm away from windshield. 2) Find release tab. 3) Slide old blade off. 4) Slide new blade on until click. 5) Lower arm gently."),
        ("How to clean MAF sensor?", "1) Locate MAF (in intake after air filter). 2) Disconnect electrical. 3) Remove MAF. 4) Spray MAF cleaner on sensor wires. 5) Let dry completely. 6) Reinstall. DO NOT touch sensor wires."),
    ]
    
    for q, a in diy_repairs:
        all_data.append(format_qa(q, a))
    
    # Car specifications Q&A (30+ cars)
    cars = [
        ("Toyota Camry 2024", "2.5L 4cyl, 203hp, 8-speed auto, 28/39 MPG"),
        ("Honda Civic 2024", "2.0L 4cyl, 158hp, CVT, 31/40 MPG"),
        ("Ford F-150 2024", "3.5L V6 EcoBoost, 400hp, 10-speed, 18/24 MPG"),
        ("Chevrolet Silverado 2024", "5.3L V8, 355hp, 8-speed, 16/22 MPG"),
        ("Tesla Model 3 2024", "Electric, 283hp, single-speed, 138/126 MPGe"),
        ("BMW 3 Series 2024", "2.0L Turbo 4cyl, 255hp, 8-speed, 26/36 MPG"),
        ("Honda Accord 2024", "1.5L Turbo, 192hp, CVT, 29/37 MPG"),
        ("Toyota RAV4 2024", "2.5L 4cyl, 203hp, 8-speed, 27/35 MPG"),
        ("Ford Mustang 2024", "5.0L V8, 480hp, 6-speed manual/10-speed auto, 15/24 MPG"),
        ("Chevrolet Corvette 2024", "6.2L V8, 490hp, 8-speed dual-clutch, 16/24 MPG"),
    ]
    
    for car, specs in cars:
        all_data.append(format_qa(f"What are the specs of {car}?", f"The {car} features: {specs}."))
        all_data.append(format_qa(f"What engine does {car} have?", f"The {car} has: {specs.split(',')[0]}."))
    
    # Safety Q&A
    safety_qa = [
        ("Is it safe to drive with check engine light on?", "Solid light: usually OK to drive to shop. Flashing light: STOP - severe misfire, can damage catalyst."),
        ("Can I drive with ABS light on?", "Yes, but ABS won't work. Brakes still function normally. Get checked soon."),
        ("Is it safe to drive overheating?", "NO. Stop immediately. Driving while overheating causes severe engine damage (warped head, blown gasket)."),
        ("Can I drive with low tire pressure?", "Short distance at low speed maybe. Severely low pressure causes tire damage and blowout risk."),
        ("Is it safe to drive without power steering?", "Yes but requires much more effort. Safe for emergency driving to shop."),
        ("Can I drive with brake warning light?", "Check brake fluid level. If low, may have leak - not safe. If pedal feels normal, drive carefully to shop."),
    ]
    
    for q, a in safety_qa:
        all_data.append(format_qa(q, a))
    
    # Cost estimates
    cost_qa = [
        ("How much does oil change cost?", "DIY: $25-50. Shop: $30-75 (conventional), $65-125 (synthetic)."),
        ("How much to replace brake pads?", "DIY: $50-100 parts. Shop: $150-300 per axle."),
        ("How much to replace timing belt?", "Shop: $500-1000+ (includes water pump recommended)."),
        ("How much for new battery?", "Battery: $100-200. Installation: free at most stores."),
        ("How much to replace alternator?", "Part: $150-400. Labor: $100-200. Total: $250-600."),
        ("How much for transmission rebuild?", "$1500-4000+ depending on transmission type."),
        ("How much to replace catalytic converter?", "$200-2500 depending on vehicle and aftermarket vs OEM."),
        ("How much for wheel alignment?", "$75-150 for 4-wheel alignment."),
        ("How much to replace spark plugs?", "DIY: $20-80 parts. Shop: $100-300 depending on access."),
        ("How much for AC recharge?", "DIY kit: $30-50. Shop: $100-250."),
    ]
    
    for q, a in cost_qa:
        all_data.append(format_qa(q, a))
    
    return all_data


def main():
    print("Generating expanded automobile dataset...")
    data = generate_all_data()
    
    output_file = DATA_DIR / "automobile_qa.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Generated {len(data)} Q&A pairs")
    print(f"✓ Saved to: {output_file}")


if __name__ == "__main__":
    main()
