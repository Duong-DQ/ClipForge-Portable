import os
import uuid
import threading
from flask import Flask, render_template, request, jsonify, send_file
from dotenv import load_dotenv

load_dotenv()

from database import app, db
from models import Job
import gemini_client

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def get_mime_type(filename: str) -> str:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "mp4"
    mime_map = {
        "mp4": "video/mp4",
        "mov": "video/quicktime",
        "avi": "video/x-msvideo",
        "mkv": "video/x-matroska",
        "webm": "video/webm",
    }
    return mime_map.get(ext, "video/mp4")


def run_job(job_id: str, api_key: str, custom_prompt: str = ""):
    """Background thread: process a single job."""
    with app.app_context():
        job = Job.query.get(job_id)
        if not job:
            return

        video_path = os.path.join(UPLOAD_DIR, job.video_filename)

        try:
            # Step 1: Upload to Gemini
            job.status = "uploading"
            db.session.commit()

            mime_type = get_mime_type(job.video_filename)
            video_file = gemini_client.upload_video_to_gemini(video_path, mime_type, api_key)

            # Step 2: Generate rewritten SRT
            job.status = "processing"
            db.session.commit()

            rewritten_srt = gemini_client.generate_rewritten_srt(video_file, job.srt_content, api_key, custom_prompt)

            # Step 3: Save result
            result_path = os.path.join(RESULTS_DIR, f"{job_id}.srt")
            with open(result_path, "w", encoding="utf-8") as f:
                f.write(rewritten_srt)

            job.result_srt = rewritten_srt
            job.status = "done"
            db.session.commit()

        except Exception as e:
            job.status = "failed"
            job.error = str(e)
            db.session.commit()
        finally:
            # Cleanup uploaded video
            if os.path.exists(video_path):
                os.remove(video_path)


@app.route("/")
def index():
    jobs = Job.query.order_by(Job.created_at.desc()).all()
    return render_template("index.html", jobs=jobs)


@app.route("/jobs/<job_id>")
def view_job(job_id):
    job = Job.query.get_or_404(job_id)
    return render_template("result.html", job=job)


@app.route("/api/jobs", methods=["POST"])
def create_job():
    video_file = request.files.get("video")
    srt_file = request.files.get("srt")
    api_key = request.form.get("api_key", "").strip()
    custom_prompt = request.form.get("custom_prompt", "").strip()

    if not video_file or not srt_file:
        return jsonify({"error": "Both video and SRT file are required"}), 400
    if not api_key:
        return jsonify({"error": "API key is required"}), 400

    job_id = str(uuid.uuid4())

    # Save video
    video_filename = f"{job_id}_{video_file.filename}"
    video_path = os.path.join(UPLOAD_DIR, video_filename)
    video_file.save(video_path)

    # Read SRT
    srt_content = srt_file.read().decode("utf-8")

    # Video size in MB
    video_size_mb = os.path.getsize(video_path) / (1024 * 1024)

    # Create job
    job = Job(
        id=job_id,
        status="pending",
        video_filename=video_filename,
        srt_filename=srt_file.filename,
        srt_content=srt_content,
        video_size_mb=round(video_size_mb, 2),
    )
    db.session.add(job)
    db.session.commit()

    # Start background processing
    thread = threading.Thread(target=run_job, args=(job_id, api_key, custom_prompt), daemon=True)
    thread.start()

    return jsonify({"id": job_id, "status": "pending"}), 201


@app.route("/api/jobs/<job_id>/status")
def job_status(job_id):
    job = Job.query.get_or_404(job_id)
    return jsonify({
        "id": job.id,
        "status": job.status,
        "error": job.error,
    })


@app.route("/api/jobs/<job_id>/result")
def job_result(job_id):
    job = Job.query.get_or_404(job_id)
    if job.status != "done":
        return jsonify({"error": f"Job not done. Status: {job.status}"}), 400
    return jsonify({"id": job.id, "result_srt": job.result_srt})


@app.route("/api/jobs/<job_id>/download")
def job_download(job_id):
    job = Job.query.get_or_404(job_id)
    if job.status != "done" or not job.result_srt:
        return jsonify({"error": "No result available"}), 400

    result_path = os.path.join(RESULTS_DIR, f"{job_id}.srt")
    if os.path.exists(result_path):
        return send_file(result_path, as_attachment=True, download_name=f"rewritten_{job_id[:8]}.srt")
    return jsonify({"error": "Result file not found"}), 404


@app.route("/api/jobs")
def list_jobs():
    jobs = Job.query.order_by(Job.created_at.desc()).all()
    return jsonify([j.to_dict() for j in jobs])


if __name__ == "__main__":
    app.run(debug=True, port=5000)
