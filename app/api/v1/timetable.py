"""
POST /v1/timetable/generate        — Generate a single-section AI timetable.
POST /v1/timetable/generate-bulk   — Generate timetables for multiple sections
                                      with cross-section teacher conflict tracking.

These endpoints mirror the LLM_server Flask app (app.py) exactly, now integrated
into the FastAPI Shiksha-Intelligence-AI-server so a single hosted service handles
both the chat/RAG features and the timetable generation.
"""
from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from openai import AsyncOpenAI

from app.core.config import settings
from app.core.logging import logger
from app.schemas.ai_config import AIConfig
from app.services.session_service import SessionService

router = APIRouter()

# ---------------------------------------------------------------------------
# Lazy-initialised async OpenAI client
# ---------------------------------------------------------------------------

_openai_client: AsyncOpenAI | None = None


def _get_openai_client() -> AsyncOpenAI:
    global _openai_client
    if _openai_client is None:
        if not settings.OPENAI_API_KEY:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="OPENAI_API_KEY is not configured on the server.",
            )
        _openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    return _openai_client


# ---------------------------------------------------------------------------
# Pydantic request / response schemas (kept inline — timetable-specific)
# ---------------------------------------------------------------------------


class TeacherEntry(BaseModel):
    name: str
    subjects: list[str]


class GenerateTimetableRequest(BaseModel):
    subjects: list[str] = Field(..., min_length=1)
    teachers: list[TeacherEntry] = Field(..., min_length=1)
    subjects_per_day: int = Field(..., gt=0)
    user_query: str = Field(..., min_length=1)


class SectionEntry(BaseModel):
    sectionId: str
    className: str
    sectionName: str
    subjects: list[str]
    teachers: list[TeacherEntry]
    subjects_per_day: int = 6


class BulkGenerateTimetableRequest(BaseModel):
    sections: list[SectionEntry] = Field(..., min_length=1)
    user_query: str = "Create optimized timetables."


# ---------------------------------------------------------------------------
# Prompt builders (ported verbatim from LLM_server/app.py)
# ---------------------------------------------------------------------------


def _create_timetable_prompt(
    subjects: list,
    teachers: list,
    subjects_per_day: int,
    user_query: str,
) -> str:
    return f"""You are an expert school timetable scheduler. Your task is to create an optimized weekly timetable (Monday to Saturday) based on the provided data and constraints.

## Input Data:

### Subjects for the class:
{json.dumps(subjects, indent=2)}

### Teachers and the subjects they teach:
{json.dumps(teachers, indent=2)}

### Number of subjects/periods per day: {subjects_per_day}

### User Query and Constraints:
{user_query}

## Your Task:
1. Create a complete timetable from Monday to Saturday.
2. Each day must have exactly {subjects_per_day} periods/subjects.
3. Strictly follow ALL constraints mentioned in the user query.
4. Ensure no teacher has conflicting schedules (teaching two subjects at the same time).
5. Try to distribute subjects evenly across the week.
6. Optimize for student learning (avoid same subject consecutively unless specified).

## Response Format:
You MUST respond with a valid JSON object in exactly this format:

If successful:
{{
    "success": true,
    "timetable": {{
        "Monday": [
            {{"period": 1, "subject": "Subject Name", "teacher": "Teacher Name"}},
            {{"period": 2, "subject": "Subject Name", "teacher": "Teacher Name"}},
            ...
        ],
        "Tuesday": [...],
        "Wednesday": [...],
        "Thursday": [...],
        "Friday": [...],
        "Saturday": [...]
    }},
    "notes": "Any important notes about the timetable"
}}

If constraints cannot be satisfied:
{{
    "success": false,
    "error": "Detailed explanation of why the constraints cannot be met",
    "conflicting_constraints": ["List of specific constraints that conflict"]
}}

IMPORTANT:
- Respond ONLY with the JSON object, no additional text.
- Ensure all subjects from the input list are scheduled appropriately.
- Each teacher can only teach their assigned subjects.
- Validate all constraints before generating the timetable.
"""


def _create_bulk_timetable_prompt(
    subjects: list,
    teachers: list,
    subjects_per_day: int,
    user_query: str,
    teacher_bookings: dict,
    class_label: str,
) -> str:
    availability_constraints = ""
    if teacher_bookings:
        lines = []
        for teacher_name, day_periods in teacher_bookings.items():
            for day, periods in day_periods.items():
                period_str = ", ".join(str(p) for p in sorted(periods))
                lines.append(
                    f"  - {teacher_name} is UNAVAILABLE on {day} at period(s): {period_str}"
                )
        if lines:
            availability_constraints = (
                "\n### CRITICAL — Teacher Availability Constraints (from already-generated classes):\n"
                "The following teachers are already booked in other classes at these slots. "
                "You MUST NOT assign them to these periods. Pick a DIFFERENT teacher who can teach "
                "the same subject, or rearrange.\n"
                + "\n".join(lines)
                + "\n"
            )

    return f"""You are an expert school timetable scheduler. Create an optimized weekly timetable (Monday to Saturday) for **{class_label}**.

## Input Data:

### Subjects for this class:
{json.dumps(subjects, indent=2)}

### Teachers and the subjects they teach:
{json.dumps(teachers, indent=2)}

### Number of periods per day: {subjects_per_day}

### User Query and Constraints:
{user_query}
{availability_constraints}
## Your Task:
1. Create a complete timetable from Monday to Saturday.
2. Each day must have exactly {subjects_per_day} periods.
3. Strictly follow ALL constraints, ESPECIALLY the teacher availability constraints above.
4. Ensure no teacher has conflicting schedules.
5. Distribute subjects evenly across the week.
6. Optimize for student learning.

## Response Format:
Respond with ONLY a valid JSON object:

If successful:
{{
    "success": true,
    "timetable": {{
        "Monday": [
            {{"period": 1, "subject": "Subject Name", "teacher": "Teacher Name"}},
            ...
        ],
        "Tuesday": [...], "Wednesday": [...], "Thursday": [...], "Friday": [...], "Saturday": [...]
    }},
    "notes": "Any important notes"
}}

If constraints cannot be satisfied:
{{
    "success": false,
    "error": "Detailed explanation",
    "conflicting_constraints": ["List of conflicts"]
}}
"""


def _update_teacher_bookings(bookings: dict, timetable: dict) -> dict:
    """Record which teachers are booked in a generated timetable."""
    for day_name, periods in timetable.items():
        for period in periods:
            teacher = period.get("teacher", "")
            period_num = period.get("period", 0)
            if teacher and period_num:
                bookings.setdefault(teacher, {}).setdefault(day_name, [])
                if period_num not in bookings[teacher][day_name]:
                    bookings[teacher][day_name].append(period_num)
    return bookings


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/generate", tags=["Timetable"])
async def generate_timetable(body: GenerateTimetableRequest):
    """
    Generate an AI-optimised timetable for a single class/section.

    Returns a Monday-to-Saturday timetable with per-period subject + teacher assignments,
    or a structured error if the constraints cannot be satisfied.
    """
    teachers_raw = [t.model_dump() for t in body.teachers]
    prompt = _create_timetable_prompt(
        body.subjects, teachers_raw, body.subjects_per_day, body.user_query
    )

    try:
        client = _get_openai_client()

        # ── Resolve model from ai-config (Redis), fallback to env default ──
        raw_config = await SessionService.get_ai_config()
        active_model = (
            AIConfig(**raw_config).active_model if raw_config else settings.ACTIVE_MODEL
        )

        response = await client.chat.completions.create(
            model=active_model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert school timetable scheduler. Always respond with valid JSON only.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        result = json.loads(response.choices[0].message.content)
    except json.JSONDecodeError as exc:
        logger.error("Timetable LLM response not valid JSON", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"success": False, "error": f"LLM returned invalid JSON: {exc}", "conflicting_constraints": []},
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Timetable generation failed", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"success": False, "error": f"LLM API error: {exc}", "conflicting_constraints": []},
        )

    if result.get("success", False):
        return result
    # Constraint failure — return 422 with the LLM's explanation (matches LLM_server behaviour)
    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail=result,
    )


@router.post("/generate-bulk", tags=["Timetable"])
async def generate_timetable_bulk(body: BulkGenerateTimetableRequest):
    """
    Generate AI-optimised timetables for multiple sections sequentially,
    tracking teacher bookings across sections to prevent conflicts.

    Returns per-section results with success/failure status.
    """
    results = []
    teacher_bookings: dict = {}

    for section in body.sections:
        section_id = section.sectionId
        class_label = f"{section.className} - Section {section.sectionName}"
        teachers_raw = [t.model_dump() for t in section.teachers]

        if not section.subjects or not section.teachers:
            results.append(
                {
                    "sectionId": section_id,
                    "className": section.className,
                    "sectionName": section.sectionName,
                    "success": False,
                    "error": "No subjects or teachers configured for this section.",
                }
            )
            continue

        try:
            prompt = _create_bulk_timetable_prompt(
                section.subjects,
                teachers_raw,
                section.subjects_per_day,
                body.user_query,
                teacher_bookings,
                class_label,
            )
            client = _get_openai_client()

            # ── Resolve model from ai-config (Redis), fallback to env default ──
            raw_config = await SessionService.get_ai_config()
            active_model = (
                AIConfig(**raw_config).active_model if raw_config else settings.ACTIVE_MODEL
            )

            response = await client.chat.completions.create(
                model=active_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert school timetable scheduler. Always respond with valid JSON only.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                response_format={"type": "json_object"},
            )
            result = json.loads(response.choices[0].message.content)

            if result.get("success", False):
                timetable = result.get("timetable", {})
                teacher_bookings = _update_teacher_bookings(teacher_bookings, timetable)
                results.append(
                    {
                        "sectionId": section_id,
                        "className": section.className,
                        "sectionName": section.sectionName,
                        "success": True,
                        "timetable": timetable,
                        "notes": result.get("notes", ""),
                    }
                )
            else:
                results.append(
                    {
                        "sectionId": section_id,
                        "className": section.className,
                        "sectionName": section.sectionName,
                        "success": False,
                        "error": result.get("error", "LLM could not generate timetable."),
                        "conflicting_constraints": result.get("conflicting_constraints", []),
                    }
                )

        except Exception as exc:
            logger.error("Bulk timetable section failed", section_id=section_id, error=str(exc))
            results.append(
                {
                    "sectionId": section_id,
                    "className": section.className,
                    "sectionName": section.sectionName,
                    "success": False,
                    "error": f"LLM error: {exc}",
                }
            )

    success_count = sum(1 for r in results if r.get("success"))
    return {
        "success": success_count > 0,
        "totalSections": len(results),
        "successCount": success_count,
        "failedCount": len(results) - success_count,
        "results": results,
    }
