from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
import re

# ==========================================================
# LOAD MODEL (Loaded Once)
# ==========================================================

MODEL_NAME = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)


# ==========================================================
# DEVICE KNOWLEDGE BASE
# ==========================================================

DEVICE_INFO = {
    "fridge": {"level": "Medium"},
    "oven": {"level": "High"},
    "dishwasher": {"level": "Medium-High"},
    "heater": {"level": "High"},
    "microwave": {"level": "Low-Medium"},
    "air conditioning": {"level": "Very High"},
    "computer": {"level": "Low-Medium"},
    "tv": {"level": "Low"},
    "washing machine": {"level": "Medium-High"},
    "lights": {"level": "Low"},
}


# ==========================================================
# MAIN CHATBOT FUNCTION
# ==========================================================

def energy_chatbot(question, df=None):

    question_lower = question.lower()

    # 1️⃣ Detect device mentioned
    detected_device = detect_device(question_lower)

    # 2️⃣ Device-specific question
    if detected_device:
        return handle_device_question(detected_device, question_lower, df)

    # 3️⃣ Data-based analytics
    if df is not None and not df.empty:
        stats = calculate_statistics(df)

        if any(word in question_lower for word in ["most energy", "highest consumption"]):
            return f"📊 The appliance consuming the most energy is **{stats['highest']}** with an average of {stats['highest_val']:.2f} kWh."

        if any(word in question_lower for word in ["least", "most efficient", "lowest consumption"]):
            return f"📊 The most energy-efficient appliance is **{stats['lowest']}** with {stats['lowest_val']:.2f} kWh average usage."

        if "total" in question_lower:
            return f"📊 Total recorded energy consumption is **{stats['total']:.2f} kWh**."

        if "average" in question_lower:
            return f"📊 The overall average energy consumption is **{stats['average']:.2f} kWh**."

        if "peak" in question_lower and "hour" in question_lower:
            if stats.get("peak_hour"):
                return f"⏰ Your peak energy usage occurs around **{stats['peak_hour']}**."

    # 4️⃣ Generic energy saving
    if any(word in question_lower for word in ["save energy", "reduce bill", "save electricity"]):
        return generic_energy_tips()

    # 5️⃣ Fallback to LLM
    return generate_llm_response(question)


# ==========================================================
# DEVICE DETECTION
# ==========================================================

def detect_device(question_lower):
    for device in DEVICE_INFO.keys():
        if device in question_lower:
            return device
    return None


# ==========================================================
# HANDLE DEVICE QUESTIONS
# ==========================================================

def handle_device_question(device, question_lower, df=None):

    level = DEVICE_INFO[device]["level"]

    # High or low question
    if "high" in question_lower or "low" in question_lower:
        return f"⚡ {device.title()} has **{level}** electricity consumption compared to other household appliances."

    # Efficient usage question
    if any(word in question_lower for word in ["how to use", "efficient", "tips"]):
        return generate_efficiency_tips(device)

    # Consumption question
    if any(word in question_lower for word in ["consume", "how much", "electricity"]):
        response = f"⚡ {device.title()} generally has **{level}** electricity usage."

        if df is not None and not df.empty:
            df_device = df[df['appliance'].str.lower() == device]
            if not df_device.empty:
                avg_energy = df_device['energy'].mean()
                response += f"\n📊 In your home, it consumes an average of **{avg_energy:.2f} kWh**."

        return response

    # Default device summary
    return generate_device_summary(device, df)


# ==========================================================
# DEVICE SUMMARY
# ==========================================================

def generate_device_summary(device, df=None):

    response = f"""
🔌 **{device.title()} Energy Overview**

⚡ Consumption Level: {DEVICE_INFO[device]['level']}

💡 Ask me:
• How to use {device} efficiently?
• Is {device} high electricity consumption?
• How much electricity does {device} consume?
"""

    if df is not None and not df.empty:
        df_device = df[df['appliance'].str.lower() == device]
        if not df_device.empty:
            avg_energy = df_device['energy'].mean()
            response += f"\n📊 Your average usage: {avg_energy:.2f} kWh"

    return response


# ==========================================================
# EFFICIENCY TIPS
# ==========================================================

def generate_efficiency_tips(device):

    tips_dict = {
        "fridge": "Keep door closed, clean coils, set correct temperature (3-5°C).",
        "oven": "Avoid frequent door opening, use convection mode, cook multiple items together.",
        "dishwasher": "Run full loads, use eco mode, air dry instead of heat dry.",
        "heater": "Set thermostat to 20°C, insulate room, use timer mode.",
        "microwave": "Use for reheating small portions instead of oven.",
        "air conditioning": "Set temperature to 24°C, clean filters monthly.",
        "computer": "Enable power saving mode and shut down when idle.",
        "tv": "Reduce brightness and turn off completely when not watching.",
        "washing machine": "Use cold water wash and full loads.",
        "lights": "Use LED bulbs and turn off when not needed."
    }

    return f"""
💡 **How to Use {device.title()} Efficiently**

{tips_dict.get(device, "Use appliance wisely and maintain it regularly.")}

Reducing usage time and maintaining appliances properly can lower electricity bills significantly.
"""


# ==========================================================
# DATA STATISTICS
# ==========================================================

def calculate_statistics(df):

    stats = {}
    stats['total'] = df['energy'].sum()
    stats['average'] = df['energy'].mean()

    grouped = df.groupby('appliance')['energy'].mean()

    stats['highest'] = grouped.idxmax()
    stats['highest_val'] = grouped.max()

    stats['lowest'] = grouped.idxmin()
    stats['lowest_val'] = grouped.min()

    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        peak_hour = df.groupby(df['timestamp'].dt.hour)['energy'].mean().idxmax()
        stats['peak_hour'] = f"{int(peak_hour)}:00"

    return stats


# ==========================================================
# GENERIC TIPS
# ==========================================================

def generic_energy_tips():
    return """
💡 **General Energy Saving Tips**

• Use LED lighting
• Avoid standby power (use smart power strips)
• Run appliances during off-peak hours
• Maintain appliances regularly
• Use inverter AC and energy-efficient devices
• Monitor your energy usage monthly

Small changes can reduce 15–30% of your electricity bill.
"""


# ==========================================================
# LLM FALLBACK
# ==========================================================

def generate_llm_response(question):

    prompt = f"""
You are a professional smart home energy assistant.
Answer clearly and concisely.

Question: {question}
"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(**inputs, max_length=200, temperature=0.6)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
