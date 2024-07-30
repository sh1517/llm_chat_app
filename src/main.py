import uvicorn

from fastapi import FastAPI

from controlers import bedrock

app = FastAPI()

app.include_router(bedrock.router, prefix='/bedrock')

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)