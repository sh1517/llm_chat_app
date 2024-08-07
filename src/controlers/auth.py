from fastapi import APIRouter
from fastapi import Depends
from fastapi.security import OAuth2PasswordRequestForm
from datetime import timedelta

from models.user_model import User

from services.auth import authenticator
from services.exceptions.auth_exception import CredentialsException

router = APIRouter()

ACCESS_TOKEN_EXPIRE_MINUTES = 30

@router.post('/token')
async def login(credentials: OAuth2PasswordRequestForm = Depends()):
    user_information = authenticator.authenticate_user(credentials.username, credentials.password)

    if not user_information:
        raise CredentialsException()
    
    token_expire_time = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = authenticator.create_access_token(
        data={"sub": user_information.username}, expires_delta=token_expire_time
    )

    return {"access_token": access_token, "token_type": "bearer"}

@router.get("/verify")
async def token_verify(current_user: User = Depends(authenticator.get_current_user)):
    return current_user

@router.get("/get_hashed_password")
async def return_hased_password(password: str):
    return authenticator.get_password_hash(password)