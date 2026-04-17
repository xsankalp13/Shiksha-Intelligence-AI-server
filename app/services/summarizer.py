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


def summarize_class_performance(data: dict[str, Any]) -> str:
    """Converts class-level marks data → compact teacher summary."""
    if not data:
        return "No class performance data available."

    lines = ["## Class Performance Summary"]

    class_avg = data.get("classAverage", "?")
    pass_rate = data.get("passRate", "?")
    lines.append(f"Class Average: {class_avg}% | Pass Rate: {pass_rate}%")

    # Subject-wise averages
    subject_avgs = data.get("subjectAverages", [])
    if subject_avgs:
        lines.append("\nSubject Averages:")
        for s in subject_avgs[:8]:
            lines.append(f"  • {s.get('subject', '?')}: {s.get('average', '?')}%")

    # Top performers
    top = data.get("topStudents", [])
    if top:
        lines.append(f"\nTop Students: {', '.join(s.get('name', '?') for s in top[:3])}")

    # Bottom performers (needs attention)
    bottom = data.get("needsAttention", [])
    if bottom:
        lines.append(f"Needs Attention: {', '.join(s.get('name', '?') for s in bottom[:3])}")

    return "\n".join(lines)


def summarize_class_attendance(summary: dict[str, Any], at_risk: list[dict[str, Any]]) -> str:
    """Converts class attendance + at-risk list → compact teacher summary."""
    lines = ["## Class Attendance Summary"]

    total = summary.get("totalStudents", "?")
    present_pct = summary.get("presentPercentage", "?")
    absent_today = summary.get("absentToday", "?")
    lines.append(
        f"Total Students: {total} | Present Today: {present_pct}% | Absent: {absent_today}"
    )

    threshold = summary.get("threshold", 75)
    at_risk_count = len(at_risk) if at_risk else summary.get("atRiskCount", 0)
    lines.append(f"Students Below {threshold}% Attendance: {at_risk_count}")

    if at_risk:
        lines.append("\nAt-Risk Students (lowest attendance first):")
        for student in at_risk[:5]:
            name = student.get("studentName", "?")
            pct = student.get("attendancePercentage", "?")
            lines.append(f"  • {name}: {pct}%")
        if len(at_risk) > 5:
            lines.append(f"  ... and {len(at_risk) - 5} more students at risk.")

    return "\n".join(lines)
