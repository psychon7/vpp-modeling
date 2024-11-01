import streamlit as st
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.units import inch
import base64
from io import BytesIO
from datetime import datetime
import numpy_financial as npf

# Financial calculation functions
def calculate_npv(rate, cash_flows):
    return sum([cf / (1 + rate) ** i for i, cf in enumerate(cash_flows)])

def calculate_irr(cash_flows):
    try:
        # Check if there's at least one positive and one negative cash flow
        if any(cf > 0 for cf in cash_flows) and any(cf < 0 for cf in cash_flows):
            irr = npf.irr(cash_flows)
            # Check if IRR is a valid number
            if np.isnan(irr) or np.isinf(irr):
                return None
            return irr
        return None
    except:
        return None

def calculate_payback_period(cash_flows):
    try:
        cumulative = np.cumsum(cash_flows)
        if cumulative[-1] < 0:  # Project never pays back
            return None
        if cumulative[0] >= 0:  # Immediate payback
            return 0
        # Find the first positive cumulative cash flow
        for i in range(1, len(cumulative)):
            if cumulative[i] >= 0:
                # Linear interpolation for more accurate payback period
                return i - 1 + abs(cumulative[i-1]) / (cumulative[i] - cumulative[i-1])
        return None
    except:
        return None

def calculate_lcoe(total_capex, annual_opex, annual_energy, project_years, discount_rate):
    """Levelized Cost of Energy Storage"""
    try:
        total_energy = 0
        total_cost = total_capex
        for year in range(1, project_years + 1):
            total_energy += annual_energy / (1 + discount_rate)**year
            total_cost += annual_opex / (1 + discount_rate)**year
        return total_cost / total_energy
    except:
        return None

def calculate_debt_metrics(total_capex, debt_ratio, loan_interest, loan_term):
    """Calculate debt service coverage ratio and other debt metrics"""
    loan_amount = total_capex * debt_ratio
    # Calculate annual debt payment (Principal + Interest)
    annual_payment = loan_amount * (loan_interest * (1 + loan_interest)**loan_term) / ((1 + loan_interest)**loan_term - 1)
    return loan_amount, annual_payment

# Set page config
st.set_page_config(page_title="VPP Business Model Calculator", layout="wide")
st.title("Virtual Power Plant Business Model Calculator")

# Create tabs for different sections
tab_res, tab_com, tab_results = st.tabs(["Residential Settings", "Commercial Settings", "Results"])

# Sidebar for global parameters
with st.sidebar:
    st.header("Global Parameters")
    
    st.subheader("Project Timeline")
    project_years = st.number_input("Project Lifespan (years)", value=10, min_value=1, max_value=25, 
                                  help="Number of years the VPP project will operate")
    
    st.subheader("Financial Parameters")
    discount_rate = st.number_input("Discount Rate (%)", value=8.0,
                                  help="Rate used to discount future cash flows to present value") / 100
    inflation_rate = st.number_input("Inflation Rate (%)", value=2.0,
                                   help="Annual rate of price increase") / 100
    tax_rate = st.number_input("Tax Rate (%)", value=25.0,
                              help="Corporate tax rate applied to profits") / 100
    
    st.subheader("Battery Performance")
    degradation_rate = st.number_input("Annual Battery Degradation (%)", value=2.0, min_value=0.0, max_value=10.0,
                                     help="Annual reduction in battery capacity due to cycling and aging") / 100
    efficiency = st.number_input("Battery Round-trip Efficiency (%)", value=90.0, min_value=50.0, max_value=100.0,
                               help="Percentage of energy retained through charge-discharge cycle") / 100
    
    st.subheader("Debt Financing")
    enable_debt = st.toggle("Enable Debt Financing", value=False,
                          help="Toggle debt financing calculations on/off")
    
    # Only show debt inputs if debt financing is enabled
    if enable_debt:
        debt_ratio = st.number_input("Debt Ratio (%)", value=70.0,
                                    help="Percentage of project financed through debt") / 100
        loan_interest = st.number_input("Loan Interest Rate (%)", value=5.0,
                                      help="Annual interest rate on the loan") / 100
        loan_term = st.number_input("Loan Term (years)", value=7, min_value=1, max_value=20,
                                   help="Duration of the loan repayment period")
    else:
        debt_ratio = 0.0
        loan_interest = 0.0
        loan_term = 0

# Residential inputs
with tab_res:
    st.header("Residential System Parameters")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("System Specifications")
        res_sites = st.number_input("Number of Residential Sites", value=100, min_value=1,
                                  help="Total number of residential installations")
        res_battery_cap = st.number_input("Battery Capacity per Site (kWh)", value=13.5,
                                        help="Energy storage capacity per residential system")
        res_power = st.number_input("Power Rating per Site (kW)", value=5.0,
                                  help="Maximum power output per residential system")
        res_cycles = st.number_input("Daily Cycles - Residential", value=1.0,
                                   help="Number of complete charge-discharge cycles per day")
        
        st.subheader("Capital Expenditure")
        res_battery_cost = st.number_input("Battery Cost ($/kWh) - Residential", value=500.0,
                                         help="Cost per kWh of battery storage capacity")
        res_install_cost = st.number_input("Installation Cost per Site ($)", value=1500.0,
                                         help="Labor and equipment costs for installation")
        res_inverter_cost = st.number_input("Inverter Cost per Site ($)", value=1000.0,
                                          help="Cost of power conversion equipment")
        
    with col2:
        st.subheader("Operating Expenses")
        res_maintenance = st.number_input("Annual Maintenance per Site ($)", value=100.0,
                                        help="Yearly maintenance cost per residential system")
        res_insurance = st.number_input("Insurance (% of CapEx) - Residential", value=1.0,
                                      help="Insurance cost as percentage of capital expenditure") / 100
        
        st.subheader("Revenue Streams")
        res_energy_arbitrage = st.number_input("Energy Arbitrage ($/kWh) - Residential", value=0.10,
                                             help="Revenue from buying low and selling high")
        res_grid_services = st.number_input("Grid Services ($/kW/year) - Residential", value=50.0,
                                          help="Revenue from providing grid support services")
        res_incentives = st.number_input("Incentives per Site ($) - Residential", value=2000.0,
                                       help="Government or utility incentives per installation")

# Commercial inputs
with tab_com:
    st.header("Commercial System Parameters")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("System Specifications")
        com_sites = st.number_input("Number of Commercial Sites", value=20, min_value=1,
                                  help="Total number of commercial installations")
        com_battery_cap = st.number_input("Battery Capacity per Site (kWh)", value=100.0,
                                        help="Energy storage capacity per commercial system")
        com_power = st.number_input("Power Rating per Site (kW)", value=30.0,
                                  help="Maximum power output per commercial system")
        com_cycles = st.number_input("Daily Cycles - Commercial", value=1.5,
                                   help="Number of complete charge-discharge cycles per day")
        
        st.subheader("Capital Expenditure")
        com_battery_cost = st.number_input("Battery Cost ($/kWh) - Commercial", value=450.0,
                                         help="Cost per kWh of battery storage capacity")
        com_install_cost = st.number_input("Installation Cost per Site ($)", value=5000.0,
                                         help="Labor and equipment costs for installation")
        com_inverter_cost = st.number_input("Inverter Cost per Site ($)", value=3000.0,
                                          help="Cost of power conversion equipment")
        
    with col2:
        st.subheader("Operating Expenses")
        com_maintenance = st.number_input("Annual Maintenance per Site ($)", value=500.0,
                                        help="Yearly maintenance cost per commercial system")
        com_insurance = st.number_input("Insurance (% of CapEx) - Commercial", value=1.5,
                                      help="Insurance cost as percentage of capital expenditure") / 100
        
        st.subheader("Revenue Streams")
        com_energy_arbitrage = st.number_input("Energy Arbitrage ($/kWh) - Commercial", value=0.15,
                                             help="Revenue from buying low and selling high")
        com_demand_charge = st.number_input("Demand Charge Reduction ($/kW/month)", value=15.0,
                                          help="Savings from reducing peak demand charges")
        com_grid_services = st.number_input("Grid Services ($/kW/year) - Commercial", value=75.0,
                                          help="Revenue from providing grid support services")
        com_incentives = st.number_input("Incentives per Site ($) - Commercial", value=10000.0,
                                       help="Government or utility incentives per installation")

# Common parameters
# degradation_rate = 0.02  # 2% annual degradation
# efficiency = 0.90  # 90% round-trip efficiency

# Calculations
def calculate_metrics():
    # Capital Expenditure Calculations
    res_capex = (
        (res_battery_cost * res_battery_cap * res_sites) +
        (res_install_cost * res_sites) +
        (res_inverter_cost * res_sites)
    )
    
    com_capex = (
        (com_battery_cost * com_battery_cap * com_sites) +
        (com_install_cost * com_sites) +
        (com_inverter_cost * com_sites)
    )
    
    total_capex = res_capex + com_capex
    
    # Revenue Calculations
    res_annual_energy = res_sites * res_battery_cap * res_cycles * 365 * efficiency
    com_annual_energy = com_sites * com_battery_cap * com_cycles * 365 * efficiency
    
    # Initialize arrays for year-by-year calculations
    years = np.arange(project_years + 1)
    degradation = (1 - degradation_rate) ** years
    inflation = (1 + inflation_rate) ** years
    
    # Revenue streams
    res_revenue = np.zeros(project_years + 1)
    com_revenue = np.zeros(project_years + 1)
    
    for year in range(1, project_years + 1):
        # Residential revenue
        res_revenue[year] = (
            (res_annual_energy * res_energy_arbitrage +
             res_sites * res_power * res_grid_services) *
            degradation[year] * inflation[year]
        )
        
        # Commercial revenue
        com_revenue[year] = (
            (com_annual_energy * com_energy_arbitrage +
             com_sites * com_power * com_grid_services +
             com_sites * com_power * com_demand_charge * 12) *
            degradation[year] * inflation[year]
        )
    
    # Operating Expenses
    res_opex = (res_maintenance * res_sites + res_capex * res_insurance) * inflation
    com_opex = (com_maintenance * com_sites + com_capex * com_insurance) * inflation
    
    # Total incentives
    total_incentives = (res_incentives * res_sites) + (com_incentives * com_sites)
    
    # Cash flows
    net_cash_flows = np.zeros(project_years + 1)
    net_cash_flows[0] = -total_capex + total_incentives  # Initial investment and incentives
    
    if enable_debt:
        # Add loan proceeds to initial cash flow
        loan_amount, annual_debt_payment = calculate_debt_metrics(total_capex, debt_ratio, loan_interest, loan_term)
        net_cash_flows[0] += loan_amount
        
        # Subtract debt service payments for loan term years
        for year in range(1, project_years + 1):
            net_cash_flows[year] = res_revenue[year] + com_revenue[year] - res_opex[year] - com_opex[year]
            if year <= loan_term:
                net_cash_flows[year] -= annual_debt_payment
    else:
        # Without debt financing, just use operational cash flows
        for year in range(1, project_years + 1):
            net_cash_flows[year] = res_revenue[year] + com_revenue[year] - res_opex[year] - com_opex[year]
    
    # Calculate metrics
    npv = calculate_npv(discount_rate, net_cash_flows)
    irr = calculate_irr(net_cash_flows)
    payback = calculate_payback_period(net_cash_flows)
    roi = (sum(net_cash_flows[1:]) / total_capex) * 100
    
    # Add these new calculations before the return statement
    # Calculate LCOE
    total_annual_energy = res_annual_energy + com_annual_energy
    lcoe = calculate_lcoe(total_capex, res_opex[1] + com_opex[1], total_annual_energy, project_years, discount_rate)
    
    # Operating metrics
    ebitda = res_revenue[1] + com_revenue[1] - res_opex[1] - com_opex[1]  # Year 1 EBITDA
    
    # Profitability metrics
    gross_margin = ((res_revenue[1] + com_revenue[1]) - (res_opex[1] + com_opex[1])) / (res_revenue[1] + com_revenue[1]) * 100
    
    # Modify the metrics dictionary to include debt-related metrics only when enabled
    metrics_dict = {
        'npv': npv,
        'irr': irr,
        'payback': payback,
        'roi': roi,
        'total_capex': total_capex,
        'annual_revenue': res_revenue[1] + com_revenue[1],
        'annual_opex': res_opex[1] + com_opex[1],
        'cash_flows': net_cash_flows,
        'res_revenue': res_revenue,
        'com_revenue': com_revenue,
        'res_opex': res_opex,
        'com_opex': com_opex,
        'lcoe': lcoe,
        'ebitda': ebitda,
        'gross_margin': gross_margin
    }
    
    # Add debt metrics only if debt financing is enabled
    if enable_debt:
        metrics_dict.update({
            'loan_amount': loan_amount,
            'annual_debt_payment': annual_debt_payment,
            'equity_investment': total_capex * (1 - debt_ratio),
            'debt_service_coverage': ebitda / annual_debt_payment if annual_debt_payment > 0 else float('inf')
        })
    else:
        metrics_dict.update({
            'loan_amount': 0,
            'annual_debt_payment': 0,
            'equity_investment': total_capex,
            'debt_service_coverage': float('inf')
        })
    
    return metrics_dict

# Add these functions BEFORE "with tab_results:"
def create_pdf_report(metrics, cash_flow_df):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        spaceAfter=30
    )
    story.append(Paragraph('VPP Business Model Analysis Report', title_style))
    story.append(Paragraph(f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Key Metrics Section
    story.append(Paragraph('Key Financial Metrics', styles['Heading2']))
    story.append(Spacer(1, 10))
    
    # Core Metrics
    story.append(Paragraph('Core Metrics', styles['Heading3']))
    core_metrics = [
        f'Net Present Value: ${metrics["npv"]:,.2f}',
        f'Internal Rate of Return: {metrics["irr"]*100:.2f}%' if metrics["irr"] else "IRR: N/A",
        f'Payback Period: {metrics["payback"]:.1f} years' if metrics["payback"] else "Payback: N/A",
        f'Return on Investment: {metrics["roi"]:.2f}%',
        f'Total CapEx: ${metrics["total_capex"]:,.2f}',
        f'Annual Revenue (Year 1): ${metrics["annual_revenue"]:,.2f}'
    ]
    
    for metric in core_metrics:
        story.append(Paragraph(metric, styles['Normal']))
    story.append(Spacer(1, 10))
    
    # Operating Metrics
    story.append(Paragraph('Operating Metrics', styles['Heading3']))
    operating_metrics = [
        f'EBITDA (Year 1): ${metrics["ebitda"]:,.2f}',
        f'Gross Margin: {metrics["gross_margin"]:.1f}%',
        f'LCOE: ${metrics["lcoe"]:.4f}/kWh' if metrics["lcoe"] else "LCOE: N/A"
    ]
    
    for metric in operating_metrics:
        story.append(Paragraph(metric, styles['Normal']))
    story.append(Spacer(1, 10))
    
    # Cash Flow Table
    story.append(Paragraph('Cash Flow Summary', styles['Heading2']))
    story.append(Spacer(1, 10))
    
    # Create table data
    table_data = [['Year', 'Net Cash Flow', 'Cumulative Cash Flow']]
    for _, row in cash_flow_df[['Year', 'Net Cash Flow', 'Cumulative Cash Flow']].iterrows():
        table_data.append([
            str(row['Year']),
            f'${row["Net Cash Flow"]:,.2f}',
            f'${row["Cumulative Cash Flow"]:,.2f}'
        ])
    
    table = Table(table_data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ALIGN', (1, 1), (-1, -1), 'RIGHT'),
    ]))
    
    story.append(table)
    
    # Build PDF
    doc.build(story)
    return buffer.getvalue()

def create_excel_export(metrics, cash_flow_df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        cash_flow_df.to_excel(writer, sheet_name='Cash Flows', index=False)
        
        # Create metrics summary
        metrics_df = pd.DataFrame({
            'Metric': ['NPV', 'IRR', 'Payback Period', 'ROI', 'Total CapEx', 'Annual Revenue'],
            'Value': [
                f'${metrics["npv"]:,.2f}',
                f'{metrics["irr"]*100:.2f}%' if metrics["irr"] else "N/A",
                f'{metrics["payback"]:.1f} years' if metrics["payback"] else "N/A",
                f'{metrics["roi"]:.2f}%',
                f'${metrics["total_capex"]:,.2f}',
                f'${metrics["annual_revenue"]:,.2f}'
            ]
        })
        metrics_df.to_excel(writer, sheet_name='Key Metrics', index=False)
        
        # Format Excel
        workbook = writer.book
        worksheet = writer.sheets['Cash Flows']
        money_fmt = workbook.add_format({'num_format': '$#,##0.00'})
        worksheet.set_column('E:F', 15, money_fmt)
        
    return output.getvalue()

# Then the Results tab code follows
with tab_results:
    metrics = calculate_metrics()
    
    st.header("Financial Metrics")
    
    # Display key metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Net Present Value", f"${metrics['npv']:,.2f}", 
                 help="NPV = Sum of (Cash Flow / (1 + Discount Rate)^Year) for all years")
        
        irr_value = metrics['irr']
        irr_display = f"{irr_value*100:.2f}%" if irr_value is not None else "Not Calculable"
        st.metric("Internal Rate of Return", irr_display,
                 help="IRR is the discount rate that makes NPV = 0")
    
    with col2:
        payback = metrics['payback']
        payback_display = f"{payback:.1f} years" if payback is not None else "Not Achievable"
        st.metric("Payback Period", payback_display,
                 help="Years until cumulative cash flow becomes positive")
        
        st.metric("Return on Investment", f"{metrics['roi']:.2f}%",
                 help="ROI = (Total Returns - Initial Investment) / Initial Investment × 100")
    with col3:
        st.metric("Total CapEx", f"${metrics['total_capex']:,.2f}",
                 help="Sum of all initial capital expenditures")
        st.metric("Annual Revenue (Year 1)", f"${metrics['annual_revenue']:,.2f}",
                 help="Total revenue from all sources in first year")
    
    # Cash flow charts using Streamlit
    st.subheader("Cash Flow Analysis")
    
    # Annual cash flows
    years = np.arange(project_years + 1)
    st.bar_chart(
        pd.DataFrame({
            'Annual Cash Flow': metrics['cash_flows']
        }, index=years)
    )
    
    # Cumulative cash flow
    st.line_chart(
        pd.DataFrame({
            'Cumulative Cash Flow': np.cumsum(metrics['cash_flows'])
        }, index=years)
    )
    
    # Revenue breakdown
    st.subheader("Revenue and Cost Breakdown")
    
    revenue_df = pd.DataFrame({
        'Residential Revenue': metrics['res_revenue'],
        'Commercial Revenue': metrics['com_revenue'],
        'Residential OpEx': -metrics['res_opex'],
        'Commercial OpEx': -metrics['com_opex']
    }, index=years)
    
    st.area_chart(revenue_df)
    
    # Detailed cash flow table
    st.subheader("Detailed Cash Flow Table")
    
    cash_flow_df = pd.DataFrame({
        'Year': range(project_years + 1),
        'Residential Revenue': metrics['res_revenue'],
        'Commercial Revenue': metrics['com_revenue'],
        'Residential OpEx': -metrics['res_opex'],
        'Commercial OpEx': -metrics['com_opex'],
        'Net Cash Flow': metrics['cash_flows'],
        'Cumulative Cash Flow': np.cumsum(metrics['cash_flows'])
    })
    
    st.dataframe(cash_flow_df.style.format("${:,.2f}"))
    
    # Revenue Components Analysis
    st.subheader("Revenue Components Analysis")
    
    # Calculate revenue components
    revenue_components = pd.DataFrame({
        'Res - Energy Arbitrage': metrics['res_revenue'] * 0.7,
        'Res - Grid Services': metrics['res_revenue'] * 0.3,
        'Com - Energy Arbitrage': metrics['com_revenue'] * 0.5,
        'Com - Demand Charges': metrics['com_revenue'] * 0.3,
        'Com - Grid Services': metrics['com_revenue'] * 0.2
    }, index=years)
    
    st.area_chart(revenue_components)

        # Add new metrics section after existing metrics
    st.subheader("Additional Financial Metrics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("LCOE", f"${metrics['lcoe']:.4f}/kWh" if metrics['lcoe'] else "N/A",
                 help="Levelized Cost of Energy Storage - total cost per kWh delivered over project lifetime")
        st.metric("Gross Margin", f"{metrics['gross_margin']:.1f}%",
                 help="(Revenue - OpEx) / Revenue × 100")
    
    with col2:
        st.metric("Equity Investment", f"${metrics['equity_investment']:,.2f}",
                 help="Initial capital required from investors")
        if enable_debt:
            st.metric("Annual Debt Payment", f"${metrics['annual_debt_payment']:,.2f}",
                     help="Yearly loan payment (Principal + Interest)")
    
    with col3:
        st.metric("EBITDA (Year 1)", f"${metrics['ebitda']:,.2f}",
                 help="Earnings Before Interest, Taxes, Depreciation, and Amortization")
        if enable_debt:
            st.metric("Debt Service Coverage", f"{metrics['debt_service_coverage']:.2f}x",
                     help="EBITDA / Annual Debt Payment - measures ability to service debt")
    
    # Export section
    st.header("Export Options")
    col1, col2 = st.columns(2)
    
    with col1:
        pdf_file = create_pdf_report(metrics, cash_flow_df)
        st.download_button(
            label="Download PDF Report",
            data=pdf_file,
            file_name="vpp_business_model_report.pdf",
            mime="application/pdf"
        )
    
    with col2:
        excel_file = create_excel_export(metrics, cash_flow_df)
        st.download_button(
            label="Download Excel Model",
            data=excel_file,
            file_name="vpp_business_model.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    

