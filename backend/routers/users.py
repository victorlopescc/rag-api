import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from auth import require_api_key
from database import Student, get_db
from services.evolution_client import evolution_client
from services.whatsapp import build_welcome_text, normalize_phone

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/users", tags=["users"])


class StudentRegister(BaseModel):
    full_name: str
    matricula: str
    phone_number: str
    data_consent: bool = True  # obrigatório no frontend; default True p/ compat.


class StudentOut(BaseModel):
    id: str
    full_name: str
    matricula: str
    phone_number: str
    lid: str | None
    active: bool
    data_consent: bool

    class Config:
        from_attributes = True


def _to_out(student: Student) -> StudentOut:
    return StudentOut(
        id=str(student.id),
        full_name=student.full_name,
        matricula=student.matricula,
        phone_number=student.phone_number,
        lid=student.lid,
        active=student.active,
        data_consent=bool(student.data_consent),
    )


def _upsert_student(db: Session, data: StudentRegister, phone: str) -> Student:
    student = db.query(Student).filter(Student.phone_number == phone).first()
    if student:
        student.full_name = data.full_name
        student.matricula = data.matricula
        student.active = True
        student.data_consent = data.data_consent
        student.lid = None
        student.pending_welcome_id = None
        db.commit()
        db.refresh(student)
        logger.info(f"Aluno atualizado: {phone}")
        return student

    student = Student(
        full_name=data.full_name,
        matricula=data.matricula,
        phone_number=phone,
        data_consent=data.data_consent,
    )
    db.add(student)
    db.commit()
    db.refresh(student)
    logger.info(f"Aluno cadastrado: {phone}")
    return student


@router.post("/register", response_model=StudentOut)
async def register_student(
    data: StudentRegister,
    db: Session = Depends(get_db),
):
    """
    Cadastra ou atualiza um aluno e envia boas-vindas pelo WhatsApp.
    O key.id da mensagem é salvo em pending_welcome_id para, mais tarde,
    vincular o LID do aluno via ACK (veja services.lid_resolver).
    """
    if not data.data_consent:
        raise HTTPException(
            status_code=400,
            detail=(
                "É necessário aceitar o uso dos dados para pesquisa acadêmica "
                "para concluir o cadastro."
            ),
        )
    phone = normalize_phone(data.phone_number)
    student = _upsert_student(db, data, phone)

    try:
        welcome_id = await evolution_client.send_text(
            phone, build_welcome_text(data.full_name)
        )
        if welcome_id:
            student.pending_welcome_id = welcome_id
            db.commit()
            db.refresh(student)
    except Exception as e:
        logger.error(f"Erro ao enviar boas-vindas: {e}")

    return _to_out(student)


@router.get("", response_model=list[StudentOut])
def list_students(db: Session = Depends(get_db)):
    students = db.query(Student).order_by(Student.created_at.desc()).all()
    return [_to_out(s) for s in students]


@router.delete("/{phone_number}", dependencies=[Depends(require_api_key)])
def delete_student(phone_number: str, db: Session = Depends(get_db)):
    student = db.query(Student).filter(Student.phone_number == phone_number).first()
    if not student:
        raise HTTPException(status_code=404, detail="Aluno não encontrado.")
    db.delete(student)
    db.commit()
    return {"detail": f"Aluno {phone_number} removido com sucesso."}
