"""
Driver Behaviour Models — Daily WhatsApp Report
"""

import os
import json
import datetime
import textwrap
import requests
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from twilio.rest import Client

TWILIO_ACCOUNT_SID  = os.environ["TWILIO_ACCOUNT_SID"]
TWILIO_AUTH_TOKEN   = os.environ["TWILIO_AUTH_TOKEN"]
TWILIO_FROM         = os.environ["TWILIO_WHATSAPP_FROM"]
TWILIO_TO           = os.environ["TWILIO_WHATSAPP_TO"]
MEDIA_HOST_URL      = os.environ.get("MEDIA_HOST_URL", "")
HF_TOKEN            = os.environ.get("HF_TOKEN", "")
OUTPUT_IMAGE        = Path("table_output.png")

BASELINE_MODELS = [
    {"source": "Hugging Face", "name": "mosesb/drowsiness-detection-yolo-cls", "arch": "YOLOv11x-cls", "type": "Classification", "behaviors": "Fatigue · Eyes · Yawning", "size": "~60 MB", "perf": "97.8% Acc", "dataset": "Combined drowsiness", "license": "MIT"},
    {"source": "Hugging Face", "name": "mosesb/drowsiness-detection-mobileViT-v2", "arch": "MobileViT-v2-200", "type": "Classification", "behaviors": "Fatigue · Eyes · Yawning", "size": "~80 MB", "perf": "96.1% Acc", "dataset": "Combined drowsiness", "license": "MIT"},
    {"source": "Hugging Face", "name": "chbh7051/vit-driver-drowsiness-detection", "arch": "ViT-Base-patch16-224", "type": "Classification", "behaviors": "Fatigue · Eyes · Yawning", "size": "~330 MB", "perf": "Eval reported", "dataset": "chbh7051/driver-drowsiness", "license": "Apache 2.0"},
    {"source": "Hugging Face", "name": "TEEN-D/Driver-Drowsiness-Detection", "arch": "Custom CNN + VGG16", "type": "Classification", "behaviors": "Fatigue · Eyes · Yawning", "size": "~140 MB", "perf": "91.6% Acc", "dataset": "Drowsy Detection Dataset", "license": "MIT"},
    {"source": "Roboflow", "name": "Driver Monitoring System v3", "arch": "YOLOv8", "type": "Object Detection", "behaviors": "Phone · Fatigue · Yawning", "size": "~6-22 MB", "perf": "mAP varies", "dataset": "2,099 images (2025)", "license": "CC BY 4.0"},
    {"source": "Roboflow", "name": "Driver behaviors (Jui)", "arch": "YOLOv8", "type": "Object Detection", "behaviors": "Seatbelt · Cigarette · Phone", "size": "~6 MB", "perf": "~85-90% mAP50", "dataset": "9,901 images", "license": "CC BY 4.0"},
    {"source": "Roboflow", "name": "DMD (Driver Monitoring)", "arch": "YOLOv8", "type": "Object Detection", "behaviors": "Distracted · Fatigue · Eyes", "size": "~22 MB", "perf": "~89-92% mAP50", "dataset": "9,739 images", "license": "CC BY 4.0"},
    {"source": "Roboflow", "name": "Driver Monitoring 2 (pikitti)", "arch": "YOLOv8", "type": "Object Detection", "behaviors": "Phone · Distracted", "size": "~6 MB", "perf": "N/A", "dataset": "8,768 images", "license": "CC BY 4.0"},
    {"source": "Roboflow", "name": "abnormal driver behaviour v1", "arch": "YOLOv8", "type": "Object Detection", "behaviors": "Phone · Eyes · Yawn · Seat · Distracted", "size": "~6-22 MB", "perf": "N/A", "dataset": "Custom (2024)", "license": "CC BY 4.0"},
    {"source": "Roboflow", "name": "detect smoking behavior (truong)", "arch": "YOLOv8", "type": "Object Detection", "behaviors": "Smoking", "size": "~6 MB", "perf": "N/A", "dataset": "853 images (2024)", "license": "CC BY 4.0"},
    {"source": "GitHub", "name": "roihan12/traffic-accident-detection", "arch": "YOLOv8", "type": "Object Detection", "behaviors": "Traffic incidents", "size": "~6-130 MB", "perf": "~90-95% mAP50", "dataset": "Custom traffic", "license": "MIT"},
    {"source": "GitHub", "name": "DebajyotiTalukder2001/Traffic-Monitoring-System", "arch": "YOLOv8s", "type": "Detection + Tracking", "behaviors": "Traffic · Speed violation", "size": "~22 MB", "perf": "~95% avg acc", "dataset": "COCO + custom", "license": "MIT"},
    {"source": "Research", "name": "ME-YOLOv8 (Debsi et al. 2024)", "arch": "YOLOv8 + MHSA + ECA", "type": "Object Detection", "behaviors": "Phone · Eyes · Yawn · Seat · Fatigue", "size": "~30-50 MB", "perf": "87-93% mAP50", "dataset": "DDFDD + StateFarm + YawDD", "license": "Research only"},
    {"source": "Research", "name": "DAHD-YOLO (MDPI Sensors 2025)", "arch": "YOLOv8 + DBCA + AFGCA", "type": "Object Detection", "behaviors": "Smoking · Phone · Distracted", "size": "~30-60 MB", "perf": "> YOLOv8 baseline", "dataset": "Custom + Kaggle cigarette", "license": "CC BY 4.0"},
    {"source": "Research", "name": "YOLO-fastest-xl (IEEE 2021)", "arch": "YOLO-fastest-xl", "type": "Object Detection", "behaviors": "Distracted · Phone · Fatigue · Yawning", "size": "~4 MB (edge)", "perf": "F1=91.84% mAP=95.81%", "dataset": "Custom dashboard cam", "license": "Research only"},
    {"source": "Research", "name": "Video Swin Transformer (2023)", "arch": "Video Swin Transformer", "type": "Video Classification", "behaviors": "Distracted (9 classes) · Fatigue", "size": "~200 MB", "perf": "97.5% Acc (DMD)", "dataset": "DMD + NTHU-DDD", "license": "Research only"},
]

SOURCE_COLORS = {
    "Hugging Face": (245, 166, 35),
    "Roboflow":     (41, 182, 246),
    "GitHub":       (100, 200, 120),
    "Research":     (200, 140, 230),
}

COL_MAP = {
    "#": 36, "Source": 105, "Model Name": 280, "Architecture": 165,
    "Type": 130, "Covered Behaviours": 210, "Size": 100,
    "Performance": 145, "Dataset": 175, "License": 90,
}

def load_font(size, bold=False):
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for fp in font_paths:
        if Path(fp).exists():
            try:
                return ImageFont.truetype(fp, size)
            except:
                pass
    return ImageFont.load_default()

def wrap_text(text, max_chars):
    return textwrap.wrap(text, width=max_chars) or [""]

def perf_color(perf):
    p = perf.lower()
    if "n/a" in p or "unknown" in p:
        return (140, 150, 170)
    try:
        val = float("".join(c for c in p.split("%")[0] if c.isdigit() or c == "."))
        if val >= 95: return (100, 200, 130)
        if val >= 88: return (255, 210, 70)
        return (255, 160, 60)
    except:
        return (80, 160, 240)

def render_table_image(models, output_path):
    COL_WIDTHS = list(COL_MAP.values())
    COL_KEYS   = list(COL_MAP.keys())
    total_width = sum(COL_WIDTHS) + 60
    PAD_LEFT, PAD_TOP, ROW_H, HEADER_H = 24, 90, 52, 44
    total_height = PAD_TOP + HEADER_H + len(models) * ROW_H + 80

    img  = Image.new("RGB", (total_width, total_height), (15, 17, 23))
    draw = ImageDraw.Draw(img)

    f_title  = load_font(20, bold=True)
    f_sub    = load_font(12)
    f_header = load_font(11, bold=True)
    f_body   = load_font(11)
    f_bold   = load_font(11, bold=True)
    f_small  = load_font(10)

    today = datetime.date.today().strftime("%d %B %Y")

    draw.rectangle([0, 0, total_width, PAD_TOP - 4], fill=(22, 28, 55))
    draw.text((PAD_LEFT, 14), "  Driver Behaviour Events - Open Source AI Models", font=f_title, fill=(255,255,255))
    draw.text((PAD_LEFT, 46), f"Sources: Hugging Face · Roboflow Universe · GitHub · Published Research     |     Updated: {today}     |     Total models: {len(models)}", font=f_sub, fill=(140,150,170))
    draw.line([(0, PAD_TOP-4), (total_width, PAD_TOP-4)], fill=(60,130,220), width=2)

    x = PAD_LEFT
    draw.rectangle([0, PAD_TOP-2, total_width, PAD_TOP+HEADER_H], fill=(22,28,55))
    for i, key in enumerate(COL_KEYS):
        draw.text((x+4, PAD_TOP+12), key.upper(), font=f_header, fill=(160,190,230))
        x += COL_WIDTHS[i]
    draw.line([(0, PAD_TOP+HEADER_H), (total_width, PAD_TOP+HEADER_H)], fill=(50,70,120), width=1)

    for idx, m in enumerate(models):
        y  = PAD_TOP + HEADER_H + idx * ROW_H
        bg = (26,29,45) if idx % 2 == 0 else (19,22,32)
        draw.rectangle([0, y, total_width, y+ROW_H], fill=bg)
        if m.get("is_new"):
            draw.rectangle([0, y, 4, y+ROW_H], fill=(100,220,130))

        x = PAD_LEFT
        draw.text((x+4, y+17), str(idx+1), font=f_small, fill=(140,150,170))
        x += COL_WIDTHS[0]

        sc = SOURCE_COLORS.get(m["source"], (140,150,170))
        draw.rounded_rectangle([x, y+14, x+COL_WIDTHS[1]-8, y+34], radius=8, fill=(sc[0]//4, sc[1]//4, sc[2]//4))
        draw.text((x+5, y+17), m["source"], font=f_small, fill=sc)
        x += COL_WIDTHS[1]

        for li, line in enumerate(wrap_text(m["name"], 32)[:2]):
            draw.text((x+4, y+8+li*14), line, font=f_bold if li==0 else f_small, fill=(255,255,255) if li==0 else (140,150,170))
        x += COL_WIDTHS[2]

        for li, line in enumerate(wrap_text(m["arch"], 18)[:2]):
            draw.text((x+4, y+12+li*13), line, font=f_small, fill=(180,200,220))
        x += COL_WIDTHS[3]

        for li, line in enumerate(wrap_text(m["type"], 16)[:2]):
            draw.text((x+4, y+12+li*13), line, font=f_small, fill=(144,164,174))
        x += COL_WIDTHS[4]

        for li, line in enumerate(wrap_text(m["behaviors"], 24)[:2]):
            draw.text((x+4, y+12+li*13), line, font=f_small, fill=(200,230,210))
        x += COL_WIDTHS[5]

        draw.text((x+4, y+17), m["size"], font=f_small, fill=(140,150,170))
        x += COL_WIDTHS[6]

        pc = perf_color(m["perf"])
        for li, line in enumerate(wrap_text(m["perf"], 16)[:2]):
            draw.text((x+4, y+12+li*13), line, font=f_bold if li==0 else f_small, fill=pc)
        x += COL_WIDTHS[7]

        for li, line in enumerate(wrap_text(m["dataset"], 20)[:2]):
            draw.text((x+4, y+12+li*13), line, font=f_small, fill=(144,164,174))
        x += COL_WIDTHS[8]

        draw.text((x+4, y+17), m["license"], font=f_small, fill=(120,144,156))
        draw.line([(0, y+ROW_H), (total_width, y+ROW_H)], fill=(30,35,55), width=1)

    fy = PAD_TOP + HEADER_H + len(models) * ROW_H + 10
    draw.line([(0, fy), (total_width, fy)], fill=(40,55,90), width=1)
    draw.text((PAD_LEFT, fy+12), "HF=Hugging Face  RF=Roboflow  GH=GitHub  RS=Research  |  Green>=95%  Yellow>=88%  Orange<88%  Grey=N/A", font=f_small, fill=(140,150,170))
    draw.text((PAD_LEFT, fy+28), "Generated automatically · GitHub Actions · Sent via Twilio WhatsApp · 20:00 AST (Doha)", font=f_small, fill=(70,80,100))

    img.save(output_path, "PNG", optimize=True)
    print(f"[OK] Image saved: {output_path}  ({total_width}x{total_height}px)")

def fetch_new_hf_models(existing_names):
    headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
    queries = ["driver drowsiness", "driver monitoring", "driver distraction", "driver fatigue"]
    new_models = []
    seen = set()
    for q in queries:
        try:
            resp = requests.get("https://huggingface.co/api/models",
                params={"search": q, "limit": 10, "sort": "lastModified", "direction": -1},
                headers=headers, timeout=15)
            if resp.status_code != 200:
                continue
            for m in resp.json():
                mid = m.get("id", "")
                if mid in seen or mid in existing_names:
                    continue
                seen.add(mid)
                new_models.append({
                    "source": "Hugging Face", "name": mid, "arch": "Unknown",
                    "type": "Unknown", "behaviors": "Driver monitoring",
                    "size": "N/A", "perf": "N/A", "dataset": "N/A",
                    "license": m.get("cardData", {}).get("license", "N/A"),
                    "is_new": True,
                })
        except Exception as e:
            print(f"[HF warn] {q}: {e}")
    return new_models

def send_whatsapp(image_path, model_count, new_count):
    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    today  = datetime.date.today().strftime("%d %B %Y")
    body = (
        f"*Driver Behaviour AI Models - Daily Update*\n"
        f"Date: {today}  |  20:00 AST (Doha)\n\n"
        f"Total models tracked: *{model_count}*\n"
        f"New models found today: *{new_count}*\n\n"
        f"Sources: Hugging Face · Roboflow · GitHub · Research\n"
        f"Events: Yawning · Phone · Fatigue · Smoking · Empty seat · Eyes closed · Distracted · Traffic\n\n"
        f"_Automated daily report_"
    )
    kwargs = {"from_": TWILIO_FROM, "to": TWILIO_TO, "body": body}
    if MEDIA_HOST_URL:
        kwargs["media_url"] = [MEDIA_HOST_URL]
    message = client.messages.create(**kwargs)
    print(f"[OK] WhatsApp sent - SID: {message.sid}  Status: {message.status}")

def main():
    print(f"\n{'='*55}")
    print(f"  Daily Run  [{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC]")
    print(f"{'='*55}\n")

    all_models     = list(BASELINE_MODELS)
    existing_names = {m["name"] for m in all_models}

    print("[1/4] Checking Hugging Face for new models...")
    new_hf = fetch_new_hf_models(existing_names)
    if new_hf:
        print(f"      Found {len(new_hf)} new models!")
        all_models.extend(new_hf)
    else:
        print("      No new models found today.")

    new_count = len([m for m in all_models if m.get("is_new")])

    print(f"[2/4] Rendering table image ({len(all_models)} models)...")
    render_table_image(all_models, OUTPUT_IMAGE)

    print("[3/4] Sending WhatsApp via Twilio...")
    send_whatsapp(OUTPUT_IMAGE, len(all_models), new_count)

    print("\n[OK] Done!\n")

if __name__ == "__main__":
    main()
