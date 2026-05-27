import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from auth import require_api_key
from config import settings
from database import Student, get_db
from services.whatsapp import (
    build_registration_link,
    generate_registration_token,
    normalize_phone,
)

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


class StudentRegisterOut(StudentOut):
    """Saída do endpoint de cadastro. Inclui o link wa.me que o aluno
    precisa clicar pra ativar o cadastro enviando a primeira mensagem.

    Quando ``registration_completed_at`` está preenchido, significa que
    o aluno JÁ ativou (clicou no link e mandou a mensagem). Nesses casos
    ``whatsapp_link`` pode ser ``None`` (cadastro já completo).
    """
    whatsapp_link: str | None
    registration_completed: bool


class StudentStatusOut(BaseModel):
    """Endpoint de polling: o frontend consulta periodicamente pra saber
    se o aluno já ativou (clicou no link e mandou a mensagem)."""
    id: str
    registration_completed: bool


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


def _to_register_out(student: Student) -> StudentRegisterOut:
    """Igual a ``_to_out`` mas inclui o link wa.me e o status de ativação.

    Só monta o link quando o aluno AINDA NÃO ativou — o frontend usa
    isso pra decidir entre mostrar o botão "Abrir WhatsApp" ou
    confirmar que o cadastro já tá completo.
    """
    completed = student.registration_completed_at is not None
    link = None
    if not completed and student.registration_token and settings.bot_phone_number:
        link = build_registration_link(
            bot_phone=settings.bot_phone_number,
            token=student.registration_token,
            full_name=student.full_name,
        )
    return StudentRegisterOut(
        id=str(student.id),
        full_name=student.full_name,
        matricula=student.matricula,
        phone_number=student.phone_number,
        lid=student.lid,
        active=student.active,
        data_consent=bool(student.data_consent),
        whatsapp_link=link,
        registration_completed=completed,
    )


def _generate_unique_token(db: Session, max_tries: int = 5) -> str:
    """Gera um token único checando contra colisões no banco.

    Como o token tem 6 chars do alfabeto reduzido (~10⁹ combinações), a
    chance de colisão é desprezível até dezenas de milhares de alunos —
    mas mantemos o retry defensivo. Levanta ``RuntimeError`` se 5
    tentativas consecutivas colidirem (cenário praticamente impossível).
    """
    for _ in range(max_tries):
        token = generate_registration_token()
        existing = (
            db.query(Student.id)
            .filter(Student.registration_token == token)
            .first()
        )
        if not existing:
            return token
    raise RuntimeError("Não foi possível gerar um token único após várias tentativas.")


def _upsert_student(db: Session, data: StudentRegister, phone: str) -> Student:
    """Cria ou atualiza um Student. Em ambos os casos garante que existe
    um ``registration_token`` válido e ``registration_completed_at = None``.

    Reabertura: se o aluno já existe E já completou o cadastro antes,
    geramos um token novo mesmo assim (caso ele tenha trocado de número
    ou perdido o histórico de WhatsApp). O fluxo do webhook
    naturalmente trata o re-vínculo na próxima mensagem.
    """
    student = db.query(Student).filter(Student.phone_number == phone).first()
    token = _generate_unique_token(db)
    if student:
        student.full_name = data.full_name
        student.matricula = data.matricula
        student.active = True
        student.data_consent = data.data_consent
        student.lid = None
        student.pending_welcome_id = None
        student.registration_token = token
        student.registration_completed_at = None
        db.commit()
        db.refresh(student)
        logger.info(f"Aluno atualizado: {phone} (token regenerado)")
        return student

    student = Student(
        full_name=data.full_name,
        matricula=data.matricula,
        phone_number=phone,
        data_consent=data.data_consent,
        registration_token=token,
    )
    db.add(student)
    try:
        db.commit()
    except IntegrityError:
        # Race condition raríssima: outro request criou o mesmo phone
        # entre o ``query`` e o ``commit``. Reabre como update.
        db.rollback()
        return _upsert_student(db, data, phone)
    db.refresh(student)
    logger.info(f"Aluno cadastrado: {phone} (token={token})")
    return student


@router.post("/register", response_model=StudentRegisterOut)
async def register_student(
    data: StudentRegister,
    db: Session = Depends(get_db),
):
    """Cadastra ou atualiza um aluno e devolve o link wa.me que ele deve
    clicar pra enviar a primeira mensagem.

    Diferente do fluxo legado, NÃO enviamos boas-vindas proativamente —
    isso era detectado como spam pela Meta. Agora o ALUNO inicia a
    conversa via ``wa.me/<bot>?text=...código: ABC123``, e o webhook
    detecta o token, casa com este Student, captura o LID/phone reais
    e responde com a mensagem de boas-vindas.
    """
    if not data.data_consent:
        raise HTTPException(
            status_code=400,
            detail=(
                "É necessário aceitar o uso dos dados para pesquisa acadêmica "
                "para concluir o cadastro."
            ),
        )
    if not settings.bot_phone_number:
        # Defensivo: sem o número do bot, o link wa.me não funciona.
        # Falhar cedo é melhor que devolver um link quebrado pro aluno.
        raise HTTPException(
            status_code=500,
            detail=(
                "BOT_PHONE_NUMBER não configurado no servidor. "
                "Avise o administrador."
            ),
        )

    phone = normalize_phone(data.phone_number)
    student = _upsert_student(db, data, phone)
    return _to_register_out(student)


@router.get("/{student_id}/status", response_model=StudentStatusOut)
def get_registration_status(student_id: str, db: Session = Depends(get_db)):
    """Polling endpoint: o frontend chama periodicamente após o cadastro
    pra detectar quando o aluno enviou a primeira mensagem pelo WhatsApp.

    Útil pra mostrar uma confirmação visual ("✅ Cadastro ativado!") sem
    forçar o usuário a recarregar a página.
    """
    student = db.query(Student).filter(Student.id == student_id).first()
    if not student:
        raise HTTPException(status_code=404, detail="Aluno não encontrado.")
    return StudentStatusOut(
        id=str(student.id),
        registration_completed=student.registration_completed_at is not None,
    )


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
