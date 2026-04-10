"""
Data Summarizer — Converts raw ERP JSON into compact, token-efficient text.

Key principle: The LLM never sees raw JSON. It sees tight bullet summaries.
This is the primary token-saving mechanism in the pipeline.
"""
from __future__ import annotations

from typing import Any


def summarize_dashboard_intelligence(data: dict[str, Any]) -> str:
    """
    Converts IntelligenceResponseDTO → compact bullet summary.
    Saves ~80% tokens vs raw JSON.
    """
    lines: list[str] = ["## Student Snapshot"]

    # Profile
    profile = data.get("profile", {})
    if profile:
        lines.append(
            f"Student: {profile.get('fullName', 'N/A')} | "
            f"Class: {profile.get('className', '?')} {profile.get('sectionName', '?')} | "
            f"Enrolment: {profile.get('enrollmentNumber', '?')}"
        )

    # Attendance
    pulse = data.get("academicPulse", {})
    attend = pulse.get("predictiveAttendance", {})
    if attend:
        pct = attend.get("percentage", "?")
        status = attend.get("status", "?")
        threshold = attend.get("thresholdPercentage", 75)
        attended = attend.get("attendedClasses", "?")
        total = attend.get("totalClasses", "?")
        lines.append(
            f"Attendance: {pct}% ({attended}/{total} classes) — Status: {status} "
            f"[Threshold: {threshold}%]"
        )

    # Live schedule
    live = pulse.get("liveAcademicContext", {})
    if live:
        current = live.get("currentClass", "None right now")
        nxt = live.get("nextClass", "None scheduled")
        lines.append(f"Current Class: {current}")
        lines.append(f"Next Class: {nxt}")

    # Finance
    finance = data.get("financeHealth", {})
    if finance and not finance.get("temporarilyUnavailable", False):
        due = finance.get("totalDue", 0)
        earliest = finance.get("earliestDueDate", "N/A")
        if float(due) > 0:
            lines.append(f"Pending Fees: ₹{due} (earliest due: {earliest})")
        else:
            lines.append("Fees: All clear — no pending dues")
    elif finance.get("temporarilyUnavailable"):
        lines.append("Fees: Data temporarily unavailable")

    return "\n".join(lines)


def summarize_attendance_records(records: list[dict[str, Any]]) -> str:
    """Converts attendance record list → compact tabular summary."""
    if not records:
        return "No attendance records found for the requested period."

    lines = [f"## Attendance Records ({len(records)} entries)"]
    for r in records[:10]:  # Cap at 10 to keep tokens down
        date = r.get("date", "?")
        status = r.get("attendanceTypeShortCode", "?")
        subject = r.get("subjectName", "")
        lines.append(f"  • {date} — {status}{' (' + subject + ')' if subject else ''}")

    if len(records) > 10:
        lines.append(f"  ... and {len(records) - 10} more records.")

    return "\n".join(lines)


def summarize_marks(records: list[dict[str, Any]]) -> str:
    """Converts mark records → short performance summary."""
    if not records:
        return "No marks found for this exam schedule."

    lines = [f"## Exam Marks ({len(records)} subjects)"]
    for r in records:
        subj = r.get("subjectName", r.get("subjectCode", "?"))
        obtained = r.get("marksObtained", "?")
        max_marks = r.get("maxMarks", "?")
        grade = r.get("grade", "")
        lines.append(
            f"  • {subj}: {obtained}/{max_marks}{' — ' + grade if grade else ''}"
        )
    return "\n".join(lines)


def summarize_invoices(invoices: list[dict[str, Any]]) -> str:
    """Converts invoice list → short fee summary."""
    if not invoices:
        return "No invoices found."

    total_due = sum(
        float(inv.get("totalAmount", 0))
        for inv in invoices
        if inv.get("status") in ("PENDING", "UNPAID", "OVERDUE")
    )

    lines = [f"## Fee Invoices ({len(invoices)} total | Pending: ₹{total_due:.2f})"]
    for inv in invoices[:5]:
        status = inv.get("status", "?")
        amount = inv.get("totalAmount", "?")
        due = inv.get("dueDate", "?")
        lines.append(f"  • ₹{amount} — {status} (due: {due})")

    if len(invoices) > 5:
        lines.append(f"  ... and {len(invoices) - 5} more invoices.")

    return "\n".join(lines)
