from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.orm import Session
import pyotp
from . import schemas, models, auth
from .database import get_db

router = APIRouter(prefix="/auth", tags=["auth"])

@router.post("/signup", response_model=schemas.UserOut)
def signup(data: schemas.UserCreate, db: Session = Depends(get_db)):
    user = db.query(models.User).filter_by(username=data.username).first()
    if user: raise HTTPException(400, "Username đã tồn tại")
    totp_secret = pyotp.random_base32()
    db_user = models.User(
        username=data.username,
        email=data.email,
        full_name=data.full_name,
        hashed_password=auth.hash_password(data.password),
        totp_secret=totp_secret
    )
    db.add(db_user); db.commit(); db.refresh(db_user)
    return db_user

@router.post("/login", response_model=schemas.Token)
def login(form: schemas.LoginForm, db: Session = Depends(get_db)):
    user = db.query(models.User).filter_by(username=form.username).first()
    if not user or not auth.verify_password(form.password, user.hashed_password):
        raise HTTPException(401, "Sai credentials")
    # Sau khi login, yêu cầu 2FA
    return {"access_token": auth.create_access_token({"sub": user.username}), "token_type":"bearer", "2fa_required": True}

@router.post("/2fa-verify")
def verify_2fa(request: Request, data: schemas.TOTPVerify, db: Session = Depends(get_db)):
    token = request.headers.get("Authorization").split()[1]
    payload = auth.decode_access_token(token)
    if not payload: raise HTTPException(401, "Token invalid")
    user = db.query(models.User).filter_by(username=payload["sub"]).first()
    if not pyotp.TOTP(user.totp_secret).verify(data.otp):
        raise HTTPException(401, "OTP không hợp lệ")
    # Trả về token có flag 2FA đã hoàn thành
    return {"access_token": auth.create_access_token({"sub": user.username, "2fa": True}), "token_type":"bearer"}
