from moviepy.editor import VideoFileClip
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import whisper
import cv2
import mediapipe as mp
import math
import matplotlib.pyplot as plt
import os
from reportlab.lib.utils import ImageReader
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

RETORIKA_BLUE = colors.HexColor("#1f4fa3")   # azul principal
RETORIKA_LIGHT = colors.HexColor("#e9f0fb")  # azul claro fondo
RETORIKA_DARK = colors.HexColor("#0f2f6b")   # azul oscuro



VIDEO_PATH = "video.mp4"
AUDIO_PATH = "audio.wav"
PDF_PATH = "informe.pdf"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGO_PATH = os.path.join(BASE_DIR, "retorika_logo.png")


# 1) VIDEO -> AUDIO
def extract_audio(video_path: str, audio_path: str) -> float:
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path, codec="pcm_s16le")  # WAV
    return float(video.duration)


# 2) WHISPER 

def transcribe_with_whisper(audio_path: str, model_size: str = "base") -> dict:
    audio, sr = sf.read(audio_path)

    # float32
    audio = audio.astype(np.float32)

    # estéreo -> mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1).astype(np.float32)

    # resample a 16k
    target_sr = 16000
    if sr != target_sr:
        audio = resample_poly(audio, target_sr, sr).astype(np.float32)

    model = whisper.load_model(model_size)
    result = model.transcribe(audio, fp16=False, language="es")
    return result

# 3) MÉTRICAS + SCORING
def count_fillers_es(text: str) -> dict:
    t = " " + text.lower().replace("\n", " ") + " "
    fillers = [
        " eh ", " em ", " eee ", " esteee ", " este ", " o sea ", " pues ",
        " vale ", " digamos ", " bueno ", " entonces ", " en plan ",
        " como que ", " tipo ", " ¿no? ", " no? ", " verdad "
    ]
    counts = {}
    for f in fillers:
        counts[f.strip()] = t.count(f)
    counts = dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))
    return counts


def compute_pause_metrics(segments: list) -> dict:
    if not segments or len(segments) < 2:
        return {"num_pausas": 0, "pausa_max_s": 0.0, "pausa_media_s": 0.0}

    gaps = []
    for i in range(1, len(segments)):
        prev_end = float(segments[i - 1].get("end", 0.0))
        cur_start = float(segments[i].get("start", 0.0))
        gap = max(0.0, cur_start - prev_end)
        if gap >= 0.6:
            gaps.append(gap)

    if not gaps:
        return {"num_pausas": 0, "pausa_max_s": 0.0, "pausa_media_s": 0.0}

    return {
        "num_pausas": len(gaps),
        "pausa_max_s": float(max(gaps)),
        "pausa_media_s": float(sum(gaps) / len(gaps)),
    }


def lexical_diversity(text: str) -> float:
    words = [w.strip(".,;:¡!¿?()[]\"'").lower() for w in text.split()]
    words = [w for w in words if w]
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def analyze(result: dict, duration_sec: float) -> dict:
    text = result.get("text", "").strip()
    segments = result.get("segments", []) or []

    words = text.split()
    num_words = len(words)
    minutes = max(duration_sec / 60.0, 1e-6)
    wpm = num_words / minutes

    fillers = count_fillers_es(text)
    top_fillers = [(k, v) for k, v in fillers.items() if v > 0][:6]
    filler_total = sum(fillers.values())

    pauses = compute_pause_metrics(segments)
    diversity = lexical_diversity(text)

    #  Scoring 0-10
    # velocidad (120-170 ideal)
    if wpm < 105:
        speed_score, speed_label = 5, "Lenta"
    elif 105 <= wpm < 120:
        speed_score, speed_label = 7, "Algo lenta"
    elif 120 <= wpm <= 170:
        speed_score, speed_label = 9, "Adecuada"
    elif 170 < wpm <= 190:
        speed_score, speed_label = 7, "Algo rápida"
    else:
        speed_score, speed_label = 5, "Muy rápida"

    # fluidez: muletillas + pausas
    fillers_per_min = filler_total / minutes if minutes > 0 else filler_total
    fluency_score = 9
    if fillers_per_min > 10:
        fluency_score = 5
    elif fillers_per_min > 6:
        fluency_score = 6
    elif fillers_per_min > 3:
        fluency_score = 7

    if pauses["pausa_max_s"] >= 3.0:
        fluency_score = max(5, fluency_score - 2)
    elif pauses["pausa_max_s"] >= 2.0:
        fluency_score = max(6, fluency_score - 1)

    # claridad
    clarity_score = 8
    if diversity < 0.30:
        clarity_score = 6
    elif diversity < 0.38:
        clarity_score = 7
    elif diversity > 0.55:
        clarity_score = 9

    # estructura: marcadores
    t = text.lower()
    structure_markers = sum([
        ("hoy voy a" in t) or ("voy a hablar" in t) or ("presentar" in t),
        ("primero" in t) or ("en primer lugar" in t),
        ("después" in t) or ("luego" in t) or ("a continuación" in t),
        ("por último" in t) or ("para terminar" in t) or ("en conclusión" in t),
    ])
    structure_score = 6 + min(4, structure_markers)

    final_score = round((speed_score + fluency_score + clarity_score + structure_score) / 4, 1)

    return {
        "text": text,
        "segments": segments,
        "duration_sec": duration_sec,
        "num_words": num_words,
        "wpm": wpm,
        "diversity": diversity,
        "fillers_total": filler_total,
        "top_fillers": top_fillers,
        "pauses": pauses,
        "scores": {
            "speed": (speed_score, speed_label),
            "fluency": fluency_score,
            "clarity": clarity_score,
            "structure": structure_score,
            "final": final_score,
        }
    }

def build_recommendations(analysis: dict) -> list:
    wpm = analysis["wpm"]
    pauses = analysis["pauses"]
    fillers_total = analysis["fillers_total"]
    duration_min = max(analysis["duration_sec"] / 60.0, 1e-6)
    fillers_per_min = fillers_total / duration_min
    diversity = analysis["diversity"]

    recs = []

    if wpm < 120:
        recs.append("Aumenta ligeramente la velocidad: apunta a 130–160 palabras/min (prueba con un metrónomo o leyendo en voz alta).")
    elif wpm > 180:
        recs.append("Reduce la velocidad: introduce pausas cortas al final de frases y enfatiza 1–2 palabras clave por idea.")

    if pauses["pausa_max_s"] >= 2.5:
        recs.append("Hay pausas largas: ensaya transiciones ('pasemos a…', 'la idea clave es…') para evitar silencios >2.5s.")
    elif pauses["num_pausas"] == 0:
        recs.append("Casi no hay pausas: añade micro-pausas (0.3–0.6s) tras conceptos importantes para mejorar comprensión.")

    if fillers_per_min > 6:
        recs.append("Reducir muletillas: grábate 2 minutos, marca cada 'eh/este/o sea', y repite sustituyéndolo por una pausa corta.")
    elif fillers_per_min > 3:
        recs.append("Muletillas moderadas: trabaja respiración y ritmo; una pausa breve suele sonar más profesional que un 'eh'.")

    if diversity < 0.35:
        recs.append("Mejora la variedad de vocabulario: evita repetir las mismas 3–5 palabras y usa sinónimos o ejemplos concretos.")
    else:
        recs.append("Buen nivel de vocabulario: refuerza con ejemplos y una frase-resumen por sección para mayor impacto.")

    recs.append("Estructura recomendada: 1) contexto en 1 frase, 2) 3 ideas clave (Primero/Segundo/Tercero), 3) cierre con conclusión + llamada a la acción.")
    return recs


# 4) BODY LANGUAGE (MediaPipe Pose)

mp_pose = mp.solutions.pose


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def analyze_body_language(video_path: str, sample_fps: int = 5) -> dict:
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        pose.close()
        return {"ok": False, "error": "No se pudo abrir el video"}

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    step = max(1, int(round(fps / sample_fps)))

    torso_centers = []
    shoulder_tilts = []
    hand_motion = []

    prev_wrists = None
    frames_used = 0
    frames_total = 0

    L_SH = mp_pose.PoseLandmark.LEFT_SHOULDER.value
    R_SH = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
    L_HIP = mp_pose.PoseLandmark.LEFT_HIP.value
    R_HIP = mp_pose.PoseLandmark.RIGHT_HIP.value
    L_WR = mp_pose.PoseLandmark.LEFT_WRIST.value
    R_WR = mp_pose.PoseLandmark.RIGHT_WRIST.value

    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames_total += 1

        if i % step != 0:
            i += 1
            continue
        i += 1

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        if not res.pose_landmarks:
            continue

        lm = res.pose_landmarks.landmark

        if (lm[L_SH].visibility < 0.5 or lm[R_SH].visibility < 0.5 or
                lm[L_HIP].visibility < 0.5 or lm[R_HIP].visibility < 0.5):
            continue

        lsh = (lm[L_SH].x, lm[L_SH].y)
        rsh = (lm[R_SH].x, lm[R_SH].y)
        lhip = (lm[L_HIP].x, lm[L_HIP].y)
        rhip = (lm[R_HIP].x, lm[R_HIP].y)

        cx = (lsh[0] + rsh[0] + lhip[0] + rhip[0]) / 4.0
        cy = (lsh[1] + rsh[1] + lhip[1] + rhip[1]) / 4.0
        torso_centers.append((cx, cy))

        shoulder_tilts.append(abs(lsh[1] - rsh[1]))

        if lm[L_WR].visibility >= 0.5 and lm[R_WR].visibility >= 0.5:
            lw = (lm[L_WR].x, lm[L_WR].y)
            rw = (lm[R_WR].x, lm[R_WR].y)
            if prev_wrists is not None:
                dm = _dist(lw, prev_wrists[0]) + _dist(rw, prev_wrists[1])
                hand_motion.append(dm)
            prev_wrists = (lw, rw)

        frames_used += 1

    cap.release()
    pose.close()

    if frames_used < 10:
        return {"ok": False, "error": "Muy pocos frames útiles para analizar pose"}

    xs = np.array([p[0] for p in torso_centers], dtype=np.float32)
    ys = np.array([p[1] for p in torso_centers], dtype=np.float32)
    stability = float(np.sqrt(xs.var() + ys.var()))

    tilt = float(np.mean(shoulder_tilts))

    gestures = float(np.mean(hand_motion)) if hand_motion else 0.0

    # scoring
    if stability < 0.005:
        stab_score, stab_label = 9, "Muy estable"
    elif stability < 0.010:
        stab_score, stab_label = 8, "Estable"
    elif stability < 0.018:
        stab_score, stab_label = 6, "Algo inquieto"
    else:
        stab_score, stab_label = 5, "Inquieto"

    if tilt < 0.010:
        posture_score, posture_label = 9, "Hombros alineados"
    elif tilt < 0.020:
        posture_score, posture_label = 8, "Buena alineación"
    elif tilt < 0.035:
        posture_score, posture_label = 6, "Ligera inclinación"
    else:
        posture_score, posture_label = 5, "Inclinación notable"

    if gestures < 0.010:
        gestures_score, gestures_label = 6, "Poca gesticulación"
    elif gestures < 0.030:
        gestures_score, gestures_label = 9, "Gestos adecuados"
    elif gestures < 0.050:
        gestures_score, gestures_label = 7, "Gestos intensos"
    else:
        gestures_score, gestures_label = 5, "Gestos excesivos"

    body_score = round((stab_score + posture_score + gestures_score) / 3, 1)

    return {
        "ok": True,
        "frames_used": frames_used,
        "stability": stability,
        "tilt": tilt,
        "gestures": gestures,
        "scores": {
            "stability": (stab_score, stab_label),
            "posture": (posture_score, posture_label),
            "gestures": (gestures_score, gestures_label),
            "body_score": body_score
        }
    }

# 5) RADAR CHART

def radar_chart_png(labels, values, out_png, title=""):
    assert len(labels) == len(values)

    angles = [n / float(len(labels)) * 2 * math.pi for n in range(len(labels))]
    angles += angles[:1]
    vals = list(values) + [values[0]]

    fig = plt.figure(figsize=(5, 5), dpi=150)
    ax = plt.subplot(111, polar=True)

    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8)

    ax.set_rlabel_position(0)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(["2", "4", "6", "8", "10"], fontsize=7)
    ax.set_ylim(0, 10)

    ax.plot(angles, vals, linewidth=2)
    ax.fill(angles, vals, alpha=0.25)

    if title:
        ax.set_title(title, fontsize=10, pad=12)

    plt.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

def draw_logo_on_pages(canvas, doc):
    
    if not os.path.exists(LOGO_PATH):
        return

    canvas.saveState()
    logo_w = 3.2 * cm
    logo_h = 1.2 * cm
    x = doc.pagesize[0] - doc.rightMargin - logo_w
    y = doc.pagesize[1] - doc.topMargin + 0.3 * cm  
    canvas.drawImage(
        ImageReader(LOGO_PATH),
        x, y,
        width=logo_w,
        height=logo_h,
        mask='auto'
    )

    canvas.restoreState()

# 6) PDF
def generate_pdf(analysis: dict) -> None:
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Title2", fontSize=18, leading=22, spaceAfter=12))
    styles.add(ParagraphStyle(name="H2", fontSize=13, leading=16, spaceBefore=10, spaceAfter=6))
    styles.add(ParagraphStyle(name="Body2", fontSize=10.5, leading=14))

    doc = SimpleDocTemplate(
        PDF_PATH,
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm
    )
    story = []

    score = analysis["scores"]["final"]
    story.append(Paragraph("Informe de análisis de comunicación", styles["Title2"]))
    story.append(Paragraph(f"<b>Puntuación global:</b> {score}/10", styles["Body2"]))
    story.append(Spacer(1, 10))

    #BODY TABLE
    story.append(Paragraph("Lenguaje corporal (Pose)", styles["H2"]))
    body = analysis.get("body", {"ok": False, "error": "No analizado"})
    if body.get("ok"):
        b = body["scores"]
        data = [
            ["Estabilidad", f"{b['stability'][0]}/10 ({b['stability'][1]})"],
            ["Postura", f"{b['posture'][0]}/10 ({b['posture'][1]})"],
            ["Gestos", f"{b['gestures'][0]}/10 ({b['gestures'][1]})"],
            ["Score corporal", f"{b['body_score']}/10"],
            ["Frames analizados", str(body["frames_used"])],
        ]
        tb = Table([["Dimensión", "Resultado"]] + data, colWidths=[7 * cm, 7 * cm])
        tb.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), RETORIKA_BLUE),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("BACKGROUND", (0, 1), (-1, -1), colors.whitesmoke),
            ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
        ]))
        story.append(tb)
    else:
        story.append(Paragraph(f"No se pudo analizar pose: {body.get('error', 'Error desconocido')}", styles["Body2"]))
    story.append(Spacer(1, 12))

    #METRICS TABLE
    speed_score, speed_label = analysis["scores"]["speed"]
    pauses = analysis["pauses"]
    metrics_data = [
        ["Duración (s)", f"{analysis['duration_sec']:.1f}"],
        ["Palabras", f"{analysis['num_words']}"],
        ["Velocidad (WPM)", f"{analysis['wpm']:.1f} ({speed_label})"],
        ["Pausas (>=0.6s)", f"{pauses['num_pausas']}"],
        ["Pausa máxima (s)", f"{pauses['pausa_max_s']:.2f}"],
        ["Pausa media (s)", f"{pauses['pausa_media_s']:.2f}"],
        ["Diversidad léxica", f"{analysis['diversity']:.2f}"],
        ["Muletillas (total)", f"{analysis['fillers_total']}"],
    ]
    table = Table([["Métrica", "Valor"]] + metrics_data, colWidths=[7 * cm, 7 * cm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), RETORIKA_DARK),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("BACKGROUND", (0, 1), (-1, -1), colors.whitesmoke),
        ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(table)
    story.append(Spacer(1, 12))

    #SCORES TABLE
    story.append(Paragraph("Puntuaciones", styles["H2"]))
    scores = analysis["scores"]
    scores_data = [
        ["Velocidad", f"{scores['speed'][0]}/10"],
        ["Fluidez", f"{scores['fluency']}/10"],
        ["Claridad", f"{scores['clarity']}/10"],
        ["Estructura", f"{scores['structure']}/10"],
        ["Global", f"{scores['final']}/10"],
    ]
    t2 = Table([["Dimensión", "Puntuación"]] + scores_data, colWidths=[7 * cm, 7 * cm])
    t2.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), RETORIKA_BLUE),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("BACKGROUND", (0, 1), (-1, -1), colors.lightgrey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
    ]))
    story.append(t2)
    story.append(Spacer(1, 12))

    #TOP FILLERS
    story.append(Paragraph("Muletillas detectadas (top)", styles["H2"]))
    top = analysis["top_fillers"]
    if top:
        mf = Table([["Muletilla", "Conteo"]] + [[k, str(v)] for k, v in top], colWidths=[7 * cm, 7 * cm])
        mf.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), RETORIKA_BLUE),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("BACKGROUND", (0, 1), (-1, -1), colors.whitesmoke),
        ]))
        story.append(mf)
    else:
        story.append(Paragraph("No se detectaron muletillas frecuentes en el texto.", styles["Body2"]))
    story.append(Spacer(1, 12))

    #RECOMMENDATIONS 
    story.append(Paragraph("Recomendaciones", styles["H2"]))
    recs = build_recommendations(analysis)
    for r in recs:
        story.append(Paragraph(f"• {r}", styles["Body2"]))
    story.append(Spacer(1, 12))

    #  RADARS 
    story.append(PageBreak())
    story.append(Paragraph("RADIOGRAFÍA COMUNICATIVA", styles["Title2"]))
    story.append(Spacer(1, 8))

    # Body radar
    body = analysis.get("body", {"ok": False})
    if body.get("ok"):
        b = body["scores"]
        body_labels = ["Postura", "Estabilidad", "Gestos"]
        body_vals = [float(b["posture"][0]), float(b["stability"][0]), float(b["gestures"][0])]
        radar_chart_png(body_labels, body_vals, "radar_body.png", "Lenguaje corporal")
        story.append(Paragraph("Lenguaje corporal", styles["H2"]))
        story.append(Image("radar_body.png", width=13 * cm, height=13 * cm))
        story.append(Spacer(1, 12))
    else:
        story.append(Paragraph("Lenguaje corporal: no disponible (no se pudo analizar pose).", styles["Body2"]))
        story.append(Spacer(1, 12))

    # Voice radar (convertimos pausas a score 0..10)
    pauses = analysis["pauses"]
    # si hay pocas pausas grandes, mejor. Heurística simple:
    # - num_pausas 0..3 => bien, >6 => peor
    # - pausa_max > 2.5 => penaliza
    pause_score = 9
    if pauses["num_pausas"] > 6:
        pause_score = 6
    elif pauses["num_pausas"] > 3:
        pause_score = 7
    if pauses["pausa_max_s"] >= 3.0:
        pause_score = max(5, pause_score - 2)
    elif pauses["pausa_max_s"] >= 2.0:
        pause_score = max(6, pause_score - 1)

    voice_labels = ["Velocidad", "Pausas", "Claridad", "Fluidez", "Estructura"]
    voice_vals = [
        float(analysis["scores"]["speed"][0]),
        float(pause_score),
        float(analysis["scores"]["clarity"]),
        float(analysis["scores"]["fluency"]),
        float(analysis["scores"]["structure"]),
    ]
    radar_chart_png(voice_labels, voice_vals, "radar_voice.png", "Lenguaje paraverbal")
    story.append(Paragraph("Lenguaje paraverbal", styles["H2"]))
    story.append(Image("radar_voice.png", width=13 * cm, height=13 * cm))

    #TRANSCRIPTION
    story.append(PageBreak())
    story.append(Paragraph("Transcripción (con timestamps)", styles["Title2"]))

    segs = analysis["segments"] or []
    if segs:
        for s in segs[:400]:
            st = float(s.get("start", 0.0))
            et = float(s.get("end", 0.0))
            tx = (s.get("text", "") or "").strip()
            story.append(Paragraph(f"[{st:6.2f}s – {et:6.2f}s] {tx}", styles["Body2"]))
    else:
        story.append(Paragraph(analysis["text"], styles["Body2"]))
    
    doc.build(story, onFirstPage=draw_logo_on_pages, onLaterPages=draw_logo_on_pages)

#  MAIN

def main():
    duration = extract_audio(VIDEO_PATH, AUDIO_PATH)
    result = transcribe_with_whisper(AUDIO_PATH, model_size="base")

    analysis = analyze(result, duration)

    body = analyze_body_language(VIDEO_PATH)
    analysis["body"] = body

    # mezclar score final con body_score
    if analysis["body"].get("ok"):
        body_score = analysis["body"]["scores"]["body_score"]
        old_final = analysis["scores"]["final"]
        analysis["scores"]["final"] = round((old_final * 0.7) + (body_score * 0.3), 1)

    generate_pdf(analysis)
    print("✅ Informe generado:", PDF_PATH)


if __name__ == "__main__":
    main()