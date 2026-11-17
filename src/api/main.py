from fastapi import FastAPI
from src.schemas.hospital import HospitalCapacityUpdateRequest

app = FastAPI(
    title="ReliefWeave AI",
    description="Emergency Response and Resource Coordination Platform",
    version="0.1.0"
)


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "application": "ReliefWeave AI"
    }
    
@app.patch("/hospital/capacity")
def update_hospital_capacity(
    request: HospitalCapacityUpdateRequest
):
    return request