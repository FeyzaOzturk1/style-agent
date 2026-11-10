"""
Fotoğraf Stil Danışmanı Ajanı  (Google Gemini ücretsiz API sürümü)
-----------------------------------------------------------------
Fotoğrafı Gemini'ye gösterir, Lightroom tarzı ayar değerlerini alır,
değerleri kendisi uygular, sonucu kontrol edip gerekirse düzeltir.
Çıktılar: düzenlenmiş JPG, değerler (JSON) ve Lightroom ön ayarı (.xmp).

Kurulum:
    pip install google-genai pillow numpy
    API anahtarı: https://aistudio.google.com/apikey  (ücretsiz)
    Windows (CMD):        set GEMINI_API_KEY=anahtarınız
    Windows (PowerShell): $env:GEMINI_API_KEY="anahtarınız"
    Mac/Linux:            export GEMINI_API_KEY="anahtarınız"

Kullanım:
    python stil_ajani_gemini.py foto.jpg
    python stil_ajani_gemini.py foto.jpg "sıcak, sinematik, hafif soluk"

Not: Ücretsiz katmanda gönderdiğiniz içerik Google tarafından ürünlerini
geliştirmek için kullanılabilir. Çok özel fotoğrafları göndermeyin.
"""
import io
import json
import os
import sys
import time
from pathlib import Path
import time

import numpy as np
from google import genai
from google.genai import types
from PIL import Image, ImageFilter, ImageOps
from pydantic import BaseModel, Field

# Ücretsiz katmanda çalışan bir model. Değişirse GEMINI_MODEL ile geçersiz kılın.
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
MAX_PASSES = 2  # ilk düzenleme + en fazla 1 düzeltme turu
client = genai.Client()  # GEMINI_API_KEY ortam değişkenini okur

# ----------------------------------------------------------------------
# 1) Çıktı şeması ve sistem talimatı
# ----------------------------------------------------------------------
class EditParams(BaseModel):
    analysis: str = Field(description="Short analysis in Turkish: photo type, lighting, chosen vibe, key edits.")
    exposure: float = Field(description="Exposure in EV, range -2 to 2")
    contrast: float = Field(description="Range -100 to 100")
    highlights: float = Field(description="Range -100 to 100")
    shadows: float = Field(description="Range -100 to 100")
    whites: float = Field(description="Range -100 to 100")
    blacks: float = Field(description="Range -100 to 100")
    temperature: float = Field(description="Relative; + warmer, - cooler. Range -100 to 100")
    tint: float = Field(description="+ magenta, - green. Range -100 to 100")
    vibrance: float = Field(description="Range -100 to 100")
    saturation: float = Field(description="Range -100 to 100")
    clarity: float = Field(description="Range -100 to 100")
    sharpness: float = Field(description="Range 0 to 100")
    vignette: float = Field(description="- darkens edges, + lightens. Range -100 to 100")
    done: bool = Field(description="True if the edited result is good. Always false in the first pass.")

LIMITS = {"exposure": (-2, 2), "sharpness": (0, 100)}

def clamp_params(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if k in ("analysis", "done"):
            out[k] = v
        else:
            lo, hi = LIMITS.get(k, (-100, 100))
            out[k] = float(min(max(float(v), lo), hi))
    return out

SYSTEM = (
    "You are an experienced photo retoucher and colorist who edits in Adobe Lightroom. "
    "Edit the given photo so it looks as striking and appealing as possible, with a mood that "
    "suits its subject and light. Draw on your general knowledge of popular, well-liked editing "
    "styles on social media, but do not claim to have seen current top-performing posts.\n\n"
    "THINK IN LIGHTROOM SLIDERS, in Lightroom's own order:\n"
    "1. Identify the photo: subject, lighting, palette, mood, technical problems.\n"
    "2. Choose a vibe that strengthens what the photo already is (golden hour: warm and glowing; "
    "city/night: cinematic, deep blacks; landscape: rich natural greens/blues; portrait: soft, clean "
    "skin, slight warmth; street: textured, slightly faded; food: warm, appetizing; minimal: clean, "
    "low saturation; foggy/moody: muted, cool). Never fight the photo.\n"
    "3. Basic panel: White Balance (temperature, tint), Exposure (EV), Contrast, Highlights, Shadows, "
    "Whites, Blacks. Then Presence: Clarity, Vibrance, Saturation. Then Effects: Vignette. "
    "Then Detail: Sharpening.\n\n"
    "LIGHTROOM RULES OF THUMB:\n"
    "- Fix first, then stylize. Start with exposure and white balance.\n"
    "- Exposure is usually within +/-0.7 EV. If you raise exposure, lower Highlights; if you lower "
    "Highlights a lot, consider raising Whites slightly to keep punch.\n"
    "- Recover bright skies with Highlights -20 to -80; open shadows with Shadows +10 to +60.\n"
    "- Set Blacks slightly negative and Whites slightly positive for depth, but never clip.\n"
    "- Prefer Vibrance over Saturation; Saturation beyond +/-20 usually looks fake.\n"
    "- Clarity above +30 or Sharpness above 60 creates halos and noise; use sparingly. "
    "Negative Clarity gives a soft, dreamy look for portraits.\n"
    "- Most sliders should stay within +/-40 unless a real problem demands more. "
    "When unsure, choose smaller values.\n"
    "- Avoid over-editing signs: orange/red skin, plastic skies, clipped whites, crushed blacks.\n\n"
    "If the user specifies a style, follow it, adapting intensity to the photo. Otherwise pick the "
    "best vibe yourself.\n\n"
    "SELF-CHECK: in the second pass compare the result with the original. Is the mood stronger? "
    "Does the subject stand out? Any clipped highlights or crushed blacks? Natural skin tones? "
    "If there is a problem, send corrected absolute values; if good, set done=true and send the same values.\n\n"
    "OUTPUT: always respond with a single JSON object that matches the schema. Write the analysis field in Turkish: photo type "
    "and lighting, the vibe chosen and why, and the 2-3 most important edits. Values are always "
    "absolute relative to the original photo, using Lightroom's slider scales."
)

# ----------------------------------------------------------------------
# 2) Düzenleme motoru ve Lightroom ön ayarı
# ----------------------------------------------------------------------
def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)

def apply_edit(img: Image.Image, p: dict) -> Image.Image:
    a = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0

    # Pozlama
    a = a * (2.0 ** p["exposure"])

    # Kontrast (orta gri etrafında)
    a = (a - 0.5) * (1 + p["contrast"] / 100.0) + 0.5
    a = np.clip(a, 0, 1)

    # Ton aralıkları (parlaklığa göre maskeler)
    lum = (0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2])
    delta = (
        smoothstep(0.5, 1.0, lum) * (p["highlights"] / 100.0) * 0.25
        + (1 - smoothstep(0.0, 0.5, lum)) * (p["shadows"] / 100.0) * 0.25
        + smoothstep(0.75, 1.0, lum) * (p["whites"] / 100.0) * 0.15
        + (1 - smoothstep(0.0, 0.25, lum)) * (p["blacks"] / 100.0) * 0.15
    )
    a = np.clip(a + delta[..., None], 0, 1)

    # Beyaz dengesi
    t, ti = p["temperature"] / 100.0, p["tint"] / 100.0
    a[..., 0] += t * 0.08
    a[..., 2] -= t * 0.08
    a[..., 1] -= ti * 0.05
    a = np.clip(a, 0, 1)

    # Doygunluk ve canlılık
    gray = (0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2])[..., None]
    a = gray + (a - gray) * (1 + p["saturation"] / 100.0)
    chroma = (a.max(axis=2) - a.min(axis=2))[..., None]
    gray = (0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2])[..., None]
    a = gray + (a - gray) * (1 + (p["vibrance"] / 100.0) * (1 - chroma))
    a = np.clip(a, 0, 1)

    # Vinyet
    if abs(p["vignette"]) > 0.5:
        h, w = a.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        r = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2) / 1.414
        mask = smoothstep(0.35, 1.0, r)[..., None]
        a = np.clip(a * (1 + (p["vignette"] / 100.0) * 0.6 * mask), 0, 1)

    out = Image.fromarray((a * 255 + 0.5).astype(np.uint8))

    # Netlik (büyük yarıçaplı keskinleştirme) ve keskinlik
    if abs(p["clarity"]) > 0.5:
        r = max(out.size) / 80
        blur = np.asarray(out.filter(ImageFilter.GaussianBlur(r)), dtype=np.float32)
        arr = np.asarray(out, dtype=np.float32)
        arr = np.clip(arr + (arr - blur) * (p["clarity"] / 100.0) * 0.6, 0, 255)
        out = Image.fromarray(arr.astype(np.uint8))
    if p["sharpness"] > 0.5:
        out = out.filter(ImageFilter.UnsharpMask(radius=1.5, percent=int(p["sharpness"] * 1.5), threshold=2))
    return out

def write_xmp_preset(p: dict, path: Path, name: str):
    import uuid
    def s(v, dec=0):
        return f"{v:+.{dec}f}"
    attrs = {
        "Exposure2012": s(p["exposure"], 2),
        "Contrast2012": s(p["contrast"]),
        "Highlights2012": s(p["highlights"]),
        "Shadows2012": s(p["shadows"]),
        "Whites2012": s(p["whites"]),
        "Blacks2012": s(p["blacks"]),
        "Clarity2012": s(p["clarity"]),
        "Vibrance": s(p["vibrance"]),
        "Saturation": s(p["saturation"]),
        "Sharpness": str(int(round(p["sharpness"] * 1.5))),  # Lightroom: 0-150
        "PostCropVignetteAmount": s(p["vignette"]),
        # Göreli beyaz dengesi: fotoğrafın kendi sıcaklığına eklenir
        "IncrementalTemperature": s(p["temperature"]),
        "IncrementalTint": s(p["tint"]),
    }
    attr_lines = "\n".join(f'   crs:{k}="{v}"' for k, v in attrs.items())
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
   xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/"
   crs:PresetType="Normal"
   crs:UUID="{str(uuid.uuid4()).upper()}"
   crs:SupportsAmount="False"
   crs:SupportsColor="True"
   crs:SupportsMonochrome="False"
   crs:Version="15.0"
   crs:ProcessVersion="11.0"
{attr_lines}>
   <crs:Name>
    <rdf:Alt>
     <rdf:li xml:lang="x-default">{name}</rdf:li>
    </rdf:Alt>
   </crs:Name>
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
"""
    path.write_text(xml, encoding="utf-8")


# ----------------------------------------------------------------------
# 3) Ajan döngüsü (Gemini)
# ----------------------------------------------------------------------
def img_part(img: Image.Image, max_side=1568) -> types.Part:
    im = img.copy()
    im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=90)
    return types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg")

def ask_gemini(contents) -> dict:
    """Gemini'ye sor. Ücretsiz katmandaki hız limitine (429) takılırsa bekleyip tekrar dener."""
    for attempt in range(1, 5):
        try:
            resp = client.models.generate_content(
                model=MODEL,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM,
                    response_mime_type="application/json",
                    response_schema=EditParams,
                    temperature=0.4,
                ),
            )
            text = resp.text.strip()
            text = text.removeprefix("```json").removesuffix("```").strip()
            return clamp_params(json.loads(text))
        except Exception as e:
            msg = str(e)
            if ("429" in msg or "RESOURCE_EXHAUSTED" in msg) and attempt < 4:
                wait = 20 * attempt
                print(f"Hız limiti, {wait} sn bekleniyor...")
                time.sleep(wait)
                continue
            raise
def ask_gemini_with_retry(contents, retries=3):

    for attempt in range(retries):

        try:

            return ask_gemini(contents)

        except Exception as e:

            if "503" in str(e) or "UNAVAILABLE" in str(e):

                wait = 2 ** (attempt + 1)

                print(f"Gemini yoğun. {wait} saniye bekleniyor...")

                time.sleep(wait)

            else:

                raise

    raise Exception("Gemini API 3 denemeden sonra yanıt vermedi.")
def run(path: str, style_request: str = ""):
    original = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    request = "Edit this photo."
    if style_request:
        request += f" Requested style: {style_request}"

    params = ask_gemini_with_retry([img_part(original), request])
    edited = apply_edit(original, params)
    print(f"\n--- Tur 1 ---\n{params['analysis']}")

    if MAX_PASSES > 1:
        review = [
            "Original photo:", img_part(original),
            "Edited result, produced with these absolute values: "
            + json.dumps({k: v for k, v in params.items() if k not in ("analysis", "done")}),
            img_part(edited),
            "Compare the result with the original. If it is good, set done=true and return the same "
            "values. Otherwise return corrected absolute values (relative to the ORIGINAL photo). "
            + (f"Requested style: {style_request}" if style_request else ""),
        ]
        params2 = ask_gemini_with_retry(review)
        print(f"\n--- Tur 2 ---\n{params2['analysis']}")
        params = params2
        edited = apply_edit(original, params)

    src = Path(path)
    out_img = src.with_name(src.stem + "_edit.jpg")
    out_json = src.with_name(src.stem + "_degerler.json")
    out_xmp = src.with_name(src.stem + "_preset.xmp")
    edited.save(out_img, quality=95)  # tam çözünürlükte uygulanmış nihai sonuç
    out_json.write_text(json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8")
    write_xmp_preset(params, out_xmp, f"Ajan - {src.stem}")

    print("\n=== Lightroom / Photoshop için değerler ===")
    for k, v in params.items():
        if k not in ("analysis", "done"):
            print(f"{k:12s} {v:+.2f}" if k == "exposure" else f"{k:12s} {v:+.0f}")
    print(f"\nDüzenlenmiş fotoğraf: {out_img}\nDeğerler: {out_json}\nLightroom ön ayarı: {out_xmp}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit('Kullanım: python stil_ajani_gemini.py foto.jpg ["istenen stil"]')
    run(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "")
