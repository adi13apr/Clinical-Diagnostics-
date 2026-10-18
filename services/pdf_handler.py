from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from datetime import datetime
from xml.sax.saxutils import escape
import io
from services.calibration import REFER_MESSAGE, REFER_THRESHOLD, STATUS_CONFIDENT, STATUS_REFER

class PDFReportGenerator:
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.story = []

    def generate_diagnosis_report(self, patient_name, diagnosis, confidence, probabilities, radiologist_email, timestamp,
                                  status=None, report=None, report_source=None):
        """Generate a PDF report for a single diagnosis

        status: 'confident', 'refer' or None (records saved before triage existed)
        report: optional agent Report (summary, recommendation, limitations, disclaimer)
        report_source: optional line describing who wrote the report
        """
        
        # Create a BytesIO buffer
        pdf_buffer = io.BytesIO()
        
        # Create PDF document
        doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
        story = []
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1f77b4'),
            spaceAfter=30,
            alignment=1  # Center
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=self.styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#333333'),
            spaceAfter=12,
            spaceBefore=12
        )
        
        normal_style = self.styles['Normal']
        
        # Title
        story.append(Paragraph("🫁 Clinical Diagnostic Report", title_style))
        story.append(Spacer(1, 0.3*inch))

        # Referral banner first, so a "refer" result can't be missed
        if status == STATUS_REFER:
            referral_style = ParagraphStyle('Referral', parent=normal_style, fontSize=12, leading=16,
                                            textColor=colors.HexColor('#8B0000'))
            referral_table = Table([[Paragraph(
                f"<b>REFERRAL: CLINICIAN REVIEW REQUIRED</b><br/>{escape(REFER_MESSAGE.capitalize())}. "
                f"The calibrated confidence ({confidence:.1f}%) is below the {REFER_THRESHOLD:.0%} threshold, "
                f"so this AI result must not be relied on without clinician review.", referral_style)]],
                colWidths=[6*inch])
            referral_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#FDECEA')),
                ('BOX', (0, 0), (-1, -1), 2, colors.HexColor('#C62828')),
                ('TOPPADDING', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
                ('LEFTPADDING', (0, 0), (-1, -1), 10),
            ]))
            story.append(referral_table)
            story.append(Spacer(1, 0.3*inch))

        # Patient Information Section
        story.append(Paragraph("Patient Information", heading_style))
        patient_data = [
            ['Patient ID:', patient_name],
            ['Radiologist:', radiologist_email],
            ['Report Date:', timestamp],
        ]
        patient_table = Table(patient_data, colWidths=[2*inch, 4*inch])
        patient_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f0f0f0')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ]))
        story.append(patient_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Diagnosis Section
        story.append(Paragraph("Diagnosis Results", heading_style))
        
        # Map diagnosis to severity color
        severity_colors = {
            'Normal': colors.HexColor('#00D084'),
            'Pneumonia': colors.HexColor('#FFA500'),
            'Tuberculosis': colors.HexColor('#FF0000')
        }
        
        diagnosis_color = severity_colors.get(diagnosis, colors.grey)
        
        triage_labels = {
            STATUS_REFER: 'REFER - clinician review recommended',
            STATUS_CONFIDENT: f'Confident (calibrated confidence >= {REFER_THRESHOLD:.0%})',
        }
        diagnosis_data = [
            ['Primary Diagnosis:', diagnosis],
            ['Confidence Score:', f"{confidence:.1f}%"],
            ['Triage Status:', triage_labels.get(status, 'Not recorded')],
        ]
        diagnosis_table = Table(diagnosis_data, colWidths=[2*inch, 4*inch])
        diagnosis_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f0f0f0')),
            ('BACKGROUND', (1, 0), (1, 0), diagnosis_color),
            ('TEXTCOLOR', (1, 0), (1, 0), colors.white),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ] + ([('TEXTCOLOR', (1, 2), (1, 2), colors.HexColor('#C62828')),
              ('FONTNAME', (1, 2), (1, 2), 'Helvetica-Bold')] if status == STATUS_REFER else [])))
        story.append(diagnosis_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Prediction Probabilities Section
        story.append(Paragraph("Prediction Probabilities", heading_style))
        
        prob_data = [['Classification', 'Probability']]
        for class_name, prob in probabilities.items():
            prob_data.append([class_name, f"{prob*100:.2f}%"])
        
        prob_table = Table(prob_data, colWidths=[3*inch, 3*inch])
        prob_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ]))
        story.append(prob_table)
        story.append(Spacer(1, 0.5*inch))

        # AI-written report (escaped: LLM text may contain &, < or >)
        if report is not None:
            story.append(Paragraph("AI-Generated Report", heading_style))
            if report_source:
                story.append(Paragraph(f"<i>{escape(report_source)}</i>", normal_style))
                story.append(Spacer(1, 0.1*inch))
            story.append(Paragraph(f"<b>Summary:</b> {escape(report.summary)}", normal_style))
            story.append(Spacer(1, 0.1*inch))
            story.append(Paragraph(f"<b>Recommendation:</b> {escape(report.recommendation)}", normal_style))
            story.append(Spacer(1, 0.1*inch))
            story.append(Paragraph("<b>Limitations:</b>", normal_style))
            for limitation in report.limitations:
                story.append(Paragraph(f"&bull; {escape(limitation)}", normal_style))
            story.append(Spacer(1, 0.1*inch))
            story.append(Paragraph(f"<i>{escape(report.disclaimer)}</i>", normal_style))
            story.append(Spacer(1, 0.4*inch))

        # Footer
        story.append(Paragraph("This report was generated by the Clinical Diagnostic System. "
                              "Please consult with a qualified radiologist for final diagnosis.", 
                              self.styles['Normal']))
        
        # Build PDF
        doc.build(story)
        
        # Return the buffer content
        pdf_buffer.seek(0)
        return pdf_buffer.getvalue()
