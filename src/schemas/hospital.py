from pydantic import BaseModel,Field
'''Model: The foundation for creating Pydantic models.
Field: Lets us define additional validation rules and metadata for individual fields.'''

class HospitalCapacityUpdateRequest(BaseModel):
    hospital_id: int
    available_beds: int = Field(ge=0, description="Number of currently available beds")