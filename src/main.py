import uvicorn

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from controlers import bedrock, auth

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix='/auth')
app.include_router(bedrock.router, prefix='/bedrock')

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)