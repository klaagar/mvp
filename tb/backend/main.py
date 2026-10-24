"""TAXBuddy AI backend: FastAPI + Gemini (default) or Claude. Run: uvicorn main:app --reload"""
import os, json, time, base64, pathlib, collections
import httpx
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

PROVIDER = os.getenv("PROVIDER", "gemini")  # gemini | anthropic
GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")  # check the current model name in Google AI Studio
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")
ORIGINS = [o for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o]
RATE_PER_MIN = int(os.getenv("RATE_PER_MIN", "20"))
MAX_IMG = 6 * 1024 * 1024

RULES = (pathlib.Path(__file__).parent / "knowledge" / "tax_rules.md").read_text(encoding="utf-8")
SYSTEM = f"""คุณคือ TAXBuddy AI ผู้ช่วยภาษีเงินได้บุคคลธรรมดาของไทย พูดภาษาไทยสไตล์ลูกหลาน สุภาพ กระชับ ไม่เกิน 6 ประโยค
- ใช้ตัวเลขภาษีของผู้ใช้จาก CONTEXT เท่านั้น ห้ามคำนวณภาษีใหม่เอง
- ตอบเรื่องกฎจากฐานความรู้ด้านล่าง ถ้าไม่มีในฐานความรู้หรือไม่แน่ใจ ให้บอกตรงๆ และแนะนำตรวจกับกรมสรรพากร (rd.go.th)
- ห้ามอ้างว่าเป็นคำปรึกษาทางกฎหมายหรือภาษีที่รับรองผล
ฐานความรู้:
{RULES}"""

OCR_PROMPT = """อ่านเอกสารภาษีไทยในรูปนี้ ตอบเป็น JSON เท่านั้น รูปแบบ
{"doc":"withholding|fund|insurance|other","income_type":"salary|freelance|commission|business|other|null","income":number|null,"withheld":number|null,"product":"rmf|ssf|pension|life|health|home|null","amount":number|null,"confidence":"high|low"}
ห้ามเดาตัวเลขที่อ่านไม่ออก ให้ใส่ null"""

app = FastAPI(title="TAXBuddy AI")
if ORIGINS:
    app.add_middleware(CORSMiddleware, allow_origins=ORIGINS, allow_methods=["POST"], allow_headers=["*"])

hits: dict[str, collections.deque] = collections.defaultdict(collections.deque)
def limit(req: Request):
    ip = req.headers.get("x-forwarded-for", req.client.host).split(",")[0].strip()
    q, now = hits[ip], time.time()
    while q and now - q[0] > 60: q.popleft()
    if len(q) >= RATE_PER_MIN: raise HTTPException(429, "ส่งคำขอถี่เกินไป รอสักครู่")
    q.append(now)

async def llm(system: str, messages: list[dict], image: tuple[str, bytes] | None = None, json_mode=False) -> str:
    async with httpx.AsyncClient(timeout=60) as c:
        if PROVIDER == "anthropic":
            if not ANTHROPIC_KEY: raise HTTPException(500, "ยังไม่ได้ตั้ง ANTHROPIC_API_KEY")
            msgs = [dict(m) for m in messages]
            if image:
                msgs[-1]["content"] = [
                    {"type": "image", "source": {"type": "base64", "media_type": image[0], "data": base64.b64encode(image[1]).decode()}},
                    {"type": "text", "text": msgs[-1]["content"]}]
            r = await c.post("https://api.anthropic.com/v1/messages",
                headers={"x-api-key": ANTHROPIC_KEY, "anthropic-version": "2023-06-01"},
                json={"model": ANTHROPIC_MODEL, "max_tokens": 800, "system": system, "messages": msgs})
            if r.status_code != 200: raise HTTPException(502, "AI ตอบกลับผิดพลาด")
            return r.json()["content"][0]["text"]
        if not GEMINI_KEY: raise HTTPException(500, "ยังไม่ได้ตั้ง GEMINI_API_KEY")
        contents = [{"role": "user" if m["role"] == "user" else "model", "parts": [{"text": m["content"]}]} for m in messages]
        if image:
            contents[-1]["parts"].insert(0, {"inline_data": {"mime_type": image[0], "data": base64.b64encode(image[1]).decode()}})
        body = {"systemInstruction": {"parts": [{"text": system}]}, "contents": contents,
                "generationConfig": {"temperature": 0.3, **({"responseMimeType": "application/json"} if json_mode else {})}}
        r = await c.post(f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent",
                         headers={"x-goog-api-key": GEMINI_KEY}, json=body)
        if r.status_code != 200: raise HTTPException(502, "AI ตอบกลับผิดพลาด")
        try: return r.json()["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError): raise HTTPException(502, "AI ไม่ส่งคำตอบ")

class Msg(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(max_length=2000)

class ChatIn(BaseModel):
    messages: list[Msg] = Field(min_length=1, max_length=10)
    context: str = Field(default="{}", max_length=4000)

@app.post("/api/chat")
async def chat(body: ChatIn, req: Request):
    limit(req)
    if body.messages[-1].role != "user": raise HTTPException(400, "ข้อความสุดท้ายต้องเป็นของผู้ใช้")
    msgs = [m.model_dump() for m in body.messages]
    msgs[-1]["content"] = f"CONTEXT (ตัวเลขของผู้ใช้): {body.context}\n\nคำถาม: {msgs[-1]['content']}"
    return {"text": (await llm(SYSTEM, msgs)).strip()}

@app.post("/api/ocr")
async def ocr(req: Request, file: UploadFile = File(...)):
    limit(req)
    if file.content_type not in ("image/jpeg", "image/png", "image/webp"): raise HTTPException(400, "รองรับเฉพาะ JPG, PNG, WebP")
    data = await file.read()
    if len(data) > MAX_IMG: raise HTTPException(413, "ไฟล์ใหญ่เกิน 6 MB")
    txt = await llm("คุณเป็นเครื่องมืออ่านเอกสารภาษี ตอบ JSON เท่านั้น", [{"role": "user", "content": OCR_PROMPT}], (file.content_type, data), True)
    try: return json.loads(txt.strip().removeprefix("```json").removesuffix("```").strip())
    except ValueError: raise HTTPException(502, "อ่านผลจาก AI ไม่ได้ ลองใหม่")

app.mount("/", StaticFiles(directory=pathlib.Path(__file__).parent / "static", html=True), name="static")
