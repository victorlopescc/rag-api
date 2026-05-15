"""Resolve um LID para o aluno correspondente no banco."""
import logging

from sqlalchemy.orm import Session

from database import Student
from services.evolution_mongo import find_message_ids_for_lid
from services.whatsapp import normalize_lid

logger = logging.getLogger(__name__)


def resolve_student_by_lid(lid: str, db: Session) -> Student | None:
    """
    Tenta nesta ordem:
      1. students.lid (vínculo já gravado)
      2. Cruzar os ACKs do Mongo com students.pending_welcome_id e,
         ao achar o dono, gravar o LID para as próximas consultas.
    """
    lid = normalize_lid(lid)

    student = db.query(Student).filter(Student.lid == lid).first()
    if student:
        return student

    ack_ids = find_message_ids_for_lid(lid)
    if not ack_ids:
        return None

    student = db.query(Student).filter(
        Student.pending_welcome_id.in_(ack_ids)
    ).first()
    if not student:
        return None

    student.lid = lid
    student.pending_welcome_id = None
    db.commit()
    logger.info(f"LID {lid} vinculado ao aluno {student.phone_number}")
    return student
