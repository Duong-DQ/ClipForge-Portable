from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class Job(db.Model):
    __tablename__ = "jobs"

    id = db.Column(db.String(36), primary_key=True)
    status = db.Column(db.String(20), default="pending")  # pending, uploading, processing, done, failed
    video_filename = db.Column(db.String(255), nullable=False)
    srt_filename = db.Column(db.String(255), nullable=False)
    srt_content = db.Column(db.Text, nullable=False)
    result_srt = db.Column(db.Text, nullable=True)
    error = db.Column(db.Text, nullable=True)
    video_size_mb = db.Column(db.Float, nullable=True)
    created_at = db.Column(db.DateTime, default=db.func.now())
    updated_at = db.Column(db.DateTime, default=db.func.now(), onupdate=db.func.now())

    def to_dict(self):
        return {
            "id": self.id,
            "status": self.status,
            "video_filename": self.video_filename,
            "srt_filename": self.srt_filename,
            "video_size_mb": self.video_size_mb,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
