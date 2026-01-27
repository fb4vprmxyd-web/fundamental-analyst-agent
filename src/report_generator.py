import json
from pathlib import Path
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.enums import TA_CENTER, TA_RIGHT, TA_LEFT
import re


class ReportGenerator:
    def __init__(self, output_dir="reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save_to_json(self, analysis_data, filename=None):
        """Save analysis results to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"analysis_{analysis_data['ticker']}_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(analysis_data, f, indent=2)
        
        print(f"✓ JSON report saved: {filepath}")
        return filepath
    
    def generate_pdf(self, analysis_data, filename=None):
        """Generate a professional PDF investment report."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"report_{analysis_data['ticker']}_{timestamp}.pdf"
        
        filepath = self.output_dir / filename
        
        doc = SimpleDocTemplate(
            str(filepath),
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18,
        )
        
        # Build the document
        story = []
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1a1a1a'),
            spaceAfter=30,
            alignment=TA_CENTER,
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#2c3e50'),
            spaceAfter=12,
            spaceBefore=12,
        )
        
        subheading_style = ParagraphStyle(
            'CustomSubHeading',
            parent=styles['Heading3'],
            fontSize=12,
            textColor=colors.HexColor('#34495e'),
            spaceAfter=8,
        )
        
        # Title Page
        story.append(Spacer(1, 1.5*inch))
        story.append(Paragraph("INVESTMENT ANALYSIS REPORT", title_style))
        story.append(Spacer(1, 0.2*inch))
        story.append(Paragraph(f"{analysis_data['ticker']}", title_style))
        story.append(Spacer(1, 0.5*inch))
        
        # Report metadata
        metadata = [
            ["Report Date:", datetime.now().strftime("%B %d, %Y")],
            ["Ticker Symbol:", analysis_data['ticker']],
            ["Current Price:", f"${analysis_data['current_price']:.2f}"],
            ["Recommendation:", analysis_data['recommendation']],
        ]
        
        metadata_table = Table(metadata, colWidths=[2*inch, 3*inch])
        metadata_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(metadata_table)
        story.append(PageBreak())
        
        # Executive Summary
        story.append(Paragraph("EXECUTIVE SUMMARY", heading_style))
        story.append(Spacer(1, 0.1*inch))
        
        recommendation_color = {
            'BUY': colors.green,
            'HOLD': colors.orange,
            'SELL': colors.red,
        }.get(analysis_data['recommendation'], colors.black)
        
        exec_summary = [
            ["Recommendation:", analysis_data['recommendation']],
            ["Confidence:", analysis_data['confidence']],
            ["Target Price:", f"${analysis_data['blended_target']:.2f}"],
            ["Current Price:", f"${analysis_data['current_price']:.2f}"],
            ["Upside Potential:", f"{analysis_data['upside_potential']:.2f}%"],
        ]
        
        exec_table = Table(exec_summary, colWidths=[2*inch, 3*inch])
        exec_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#ecf0f1')),
            ('TEXTCOLOR', (1, 0), (1, 0), recommendation_color),
            ('FONTNAME', (1, 0), (1, 0), 'Helvetica-Bold'),
        ]))
        story.append(exec_table)
        story.append(Spacer(1, 0.2*inch))
        
        # DCF Valuation
        story.append(Paragraph("DISCOUNTED CASH FLOW ANALYSIS", heading_style))
        
        dcf_data = [
            ["Metric", "Value"],
            ["DCF Target Price", f"${analysis_data['dcf_target']:.2f}"],
            ["Forecast Growth (CAGR)", f"{analysis_data['dcf_valuation']['forecast_growth']:.2%}"],
            ["WACC", f"{analysis_data['dcf_valuation']['wacc']:.2%}"],
            ["Terminal Growth", f"{analysis_data['dcf_valuation']['terminal_growth']:.2%}"],
            ["Base FCFF", f"${analysis_data['dcf_valuation']['fcff_base']:,.0f}"],
            ["Enterprise Value", f"${analysis_data['dcf_valuation']['enterprise_value']:,.0f}"],
            ["Equity Value", f"${analysis_data['dcf_valuation']['equity_value']:,.0f}"],
        ]
        
        dcf_table = Table(dcf_data, colWidths=[3*inch, 2.5*inch])
        dcf_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('ALIGN', (1, 1), (1, -1), 'RIGHT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#ecf0f1')]),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ]))
        story.append(dcf_table)
        story.append(Spacer(1, 0.2*inch))
        
        # Terminal Growth Sensitivity
        if 'sensitivity' in analysis_data:
            story.append(Paragraph("Terminal Growth Sensitivity", subheading_style))
            
            sens_data = [["Terminal Growth", "Target Price", "Upside"]]
            for row in analysis_data['sensitivity']:
                sens_data.append([
                    f"{row['terminal_growth']:.1%}",
                    f"${row['intrinsic_price']:.2f}",
                    f"{((row['intrinsic_price'] - analysis_data['current_price']) / analysis_data['current_price'] * 100):+.2f}%"
                ])
            
            sens_table = Table(sens_data, colWidths=[2*inch, 2*inch, 1.5*inch])
            sens_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#ecf0f1')]),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ]))
            story.append(sens_table)
            story.append(Spacer(1, 0.2*inch))
        
        # Multiples Valuation
        story.append(PageBreak())
        story.append(Paragraph("MULTIPLES VALUATION", heading_style))
        
        mult_data = [
            ["Metric", "Current (AAPL)", "Peer Median", "Implied Price"],
        ]
        
        for metric in ['P/E', 'P/B', 'P/S', 'EV/EBITDA']:
            current = analysis_data['multiples_valuation']['current_multiples'].get(metric)
            peer = analysis_data['multiples_valuation']['peer_median_multiples'].get(metric)
            implied = analysis_data['multiples_valuation']['implied_prices'].get(metric)
            
            mult_data.append([
                metric,
                f"{current:.2f}" if current else "N/A",
                f"{peer:.2f}" if peer else "N/A",
                f"${implied:.2f}" if implied else "N/A",
            ])
        
        mult_data.append([
            "Average",
            "",
            "",
            f"${analysis_data['multiples_target']:.2f}"
        ])
        
        mult_table = Table(mult_data, colWidths=[1.5*inch, 1.5*inch, 1.5*inch, 1.5*inch])
        mult_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('ROWBACKGROUNDS', (0, 1), (-1, -2), [colors.white, colors.HexColor('#ecf0f1')]),
            ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor('#bdc3c7')),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ]))
        story.append(mult_table)
        story.append(Spacer(1, 0.3*inch))
        
            # AI Analysis
        if 'ai_report' in analysis_data and analysis_data['ai_report']:
            story.append(PageBreak())
            story.append(Paragraph("AI-POWERED ANALYSIS", heading_style))
            story.append(Spacer(1, 0.1*inch))
            
            # Convert markdown to ReportLab-friendly format
            ai_text = analysis_data['ai_report']
            
            # Convert markdown bold to HTML bold
            import re
            ai_text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', ai_text)
            ai_text = re.sub(r'__(.+?)__', r'<b>\1</b>', ai_text)
            
            # Convert markdown italic to HTML italic  
            ai_text = re.sub(r'\*(.+?)\*', r'<i>\1</i>', ai_text)
            ai_text = re.sub(r'_(.+?)_', r'<i>\1</i>', ai_text)
            
            # Remove horizontal rules
            ai_text = ai_text.replace('---', '')
            
            paragraphs = ai_text.split('\n\n')
            
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                    
                # Handle headers
                if para.startswith('# '):
                    text = para.replace('# ', '').replace('<b>', '').replace('</b>', '')
                    story.append(Paragraph(text, heading_style))
                elif para.startswith('## '):
                    text = para.replace('## ', '').replace('<b>', '').replace('</b>', '')
                    story.append(Paragraph(text, subheading_style))
                elif para.startswith('### '):
                    text = para.replace('### ', '').replace('<b>', '').replace('</b>', '')
                    story.append(Paragraph(text, subheading_style))
                else:
                    # Regular paragraph with HTML formatting preserved
                    try:
                        story.append(Paragraph(para, styles['BodyText']))
                    except Exception as e:
                        # If paragraph fails, strip HTML and try again
                        clean = para.replace('<b>', '').replace('</b>', '').replace('<i>', '').replace('</i>', '')
                        story.append(Paragraph(clean, styles['BodyText']))
                
                story.append(Spacer(1, 0.1*inch))
        
        # Build the PDF
        try:
            doc.build(story)
            print(f"✓ PDF report saved: {filepath}")
            return filepath
        except Exception as e:
            print(f"✗ Error generating PDF: {e}")
            return None

def create_analysis_data_structure(ticker, dcf_result, multiples, blended_target, 
                                   blended_upside, sensitivity, ai_report=None):
    """Create standardized data structure for JSON export."""
    
    if blended_upside > 20:
        recommendation = "BUY"
        confidence = "High" if blended_upside > 30 else "Medium"
    elif blended_upside < -15:
        recommendation = "SELL"
        confidence = "High" if blended_upside < -25 else "Medium"
    else:
        recommendation = "HOLD"
        confidence = "Medium"
    
    return {
        "ticker": ticker,
        "report_date": datetime.now().isoformat(),
        "current_price": dcf_result['current_price'],
        "recommendation": recommendation,
        "confidence": confidence,
        "blended_target": blended_target,
        "upside_potential": blended_upside,
        "dcf_target": dcf_result['intrinsic_price'],
        "multiples_target": multiples['average_implied_price'],
        "dcf_valuation": {
            "intrinsic_price": dcf_result['intrinsic_price'],
            "forecast_growth": dcf_result['forecast_growth'],
            "wacc": dcf_result['wacc'],
            "terminal_growth": dcf_result['terminal_growth'],
            "fcff_base": dcf_result['fcff_base'],
            "enterprise_value": dcf_result['enterprise_value'],
            "net_debt": dcf_result['net_debt'],
            "equity_value": dcf_result['equity_value'],
            "shares_outstanding": dcf_result['shares_outstanding'],
        },
        "multiples_valuation": {
            "current_multiples": multiples['current_multiples'],
            "peer_median_multiples": multiples['peer_median_multiples'],
            "implied_prices": multiples['implied_prices'],
            "average_implied_price": multiples['average_implied_price'],
        },
        "sensitivity": sensitivity.to_dict('records') if sensitivity is not None else [],
        "ai_report": ai_report,
    }


if __name__ == "__main__":
    # Example usage
    from fundamental_analyzer import FundamentalAnalyzer
    
    analyzer = FundamentalAnalyzer("AAPL")
    
    # Ensure cache is valid
    if not analyzer.market_data_cache:
        analyzer.update_market_data_cache()
    
    # Run analysis
    dcf_result = analyzer.dcf_valuation()
    multiples = analyzer.multiples_valuation()
    sensitivity = analyzer.dcf_terminal_growth_sensitivity()
    
    blended = dcf_result['intrinsic_price'] * 0.6 + multiples['average_implied_price'] * 0.4
    blended_upside = ((blended - dcf_result['current_price']) / dcf_result['current_price']) * 100
    
    # Create data structure
    analysis_data = create_analysis_data_structure(
        ticker="AAPL",
        dcf_result=dcf_result,
        multiples=multiples,
        blended_target=blended,
        blended_upside=blended_upside,
        sensitivity=sensitivity,
        ai_report="Sample AI analysis would go here..."
    )
    
    # Generate reports
    generator = ReportGenerator()
    generator.save_to_json(analysis_data)
    generator.generate_pdf(analysis_data)
    
    print("\n✓ Reports generated successfully!")