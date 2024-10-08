import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import plotly.graph_objs as go
from scipy.optimize import newton

st.set_page_config(page_title="Fixed Income: Bond Price & Yield Calculator")

# Adding the image at the top
image_url = "https://i.postimg.cc/9XRnzD6S/Screenshot-2024-05-27-at-5-20-28-PM.png"
st.image(image_url, use_column_width=True)

# Function to calculate yield to maturity using Newton's method
def calculate_ytm(price, par, coupon_rate, n_periods, freq):
    coupon = coupon_rate / 100 * par / freq
    guess = 0.05  # initial guess for YTM
    
    def bond_price(ytm):
        return sum([coupon / (1 + ytm / freq) ** t for t in range(1, n_periods + 1)]) + par / (1 + ytm / freq) ** n_periods

    def ytm_function(ytm):
        return price - bond_price(ytm)
    
    ytm = newton(ytm_function, guess)
    return ytm * 100 * freq

# Function to calculate yield to call using Newton's method
def calculate_ytc(price, par, coupon_rate, call_price, call_date, settlement_date, freq):
    coupon = coupon_rate / 100 * par / freq
    n_periods_call = (call_date - settlement_date).days // (365 // freq)
    guess = 0.05  # initial guess for YTC
    
    def bond_price(ytc):
        return sum([coupon / (1 + ytc / freq) ** t for t in range(1, n_periods_call + 1)]) + call_price / (1 + ytc / freq) ** n_periods_call

    def ytc_function(ytc):
        return price - bond_price(ytc)
    
    ytc = newton(ytc_function, guess)
    return ytc * 100 * freq

# Function to calculate bond price from yield to maturity
def calculate_price(par, coupon_rate, ytm, n_periods, freq):
    coupon = coupon_rate / 100 * par / freq
    cash_flows = [coupon] * n_periods + [par]
    discount_factors = [(1 + ytm / (100 * freq)) ** (-i) for i in range(1, n_periods + 2)]
    price = sum(cf * df for cf, df in zip(cash_flows, discount_factors))
    return price

# Function to calculate Macaulay duration
def calculate_macaulay_duration(par, coupon_rate, ytm, n_periods, freq):
    coupon = coupon_rate / 100 * par / freq
    cash_flows = [coupon] * n_periods + [par]
    discount_factors = [(1 + ytm / (100 * freq)) ** (-i) for i in range(1, n_periods + 2)]
    present_values = [cf * df for cf, df in zip(cash_flows, discount_factors)]
    durations = [t * pv for t, pv in enumerate(present_values, start=1)]
    macaulay_duration = sum(durations) / sum(present_values)
    return macaulay_duration / freq

# Function to calculate modified duration
def calculate_modified_duration(macaulay_duration, ytm, freq):
    return macaulay_duration / (1 + ytm / (100 * freq))

# Function to calculate key rate duration
def calculate_key_rate_duration(price, par, coupon_rate, ytm, n_periods, freq):
    shock = 0.01  # 1% interest rate shock
    key_rate_durations = []

    for period in range(1, n_periods + 1):
        bumped_ytm = ytm / 100 + shock
        price_up = calculate_price(par, coupon_rate, bumped_ytm * 100, n_periods, freq)
        price_down = calculate_price(par, coupon_rate, (ytm / 100 - shock) * 100, n_periods, freq)
        
        key_rate_duration = (price_down - price_up) / (2 * price * shock)
        key_rate_durations.append(key_rate_duration)
    
    return np.mean(key_rate_durations)

# Function to calculate convexity
def calculate_convexity(price, par, coupon_rate, ytm, n_periods, freq):
    coupon = coupon_rate / 100 * par / freq
    convexity_sum = 0
    for t in range(1, n_periods + 1):
        cash_flow = coupon if t < n_periods else coupon + par
        term_convexity = (cash_flow * t * (t + 1)) / ((1 + ytm / (100 * freq)) ** (t + 2))
        convexity_sum += term_convexity
    convexity = (convexity_sum / (price * freq ** 2))/10
    return convexity

# Function to calculate convexity for callable bonds
def calculate_convexity_callable(price, par, coupon_rate, call_price, call_date, ytm, settlement_date, freq):
    coupon = coupon_rate / 100 * par / freq
    n_periods_call = (call_date - settlement_date).days // (365 // freq)
    convexity_sum = 0
    for t in range(1, n_periods_call + 1):
        cash_flow = coupon if t < n_periods_call else coupon + call_price
        term_convexity = (cash_flow * t * (t + 1)) / ((1 + ytm / (100 * freq)) ** (t + 2))
        convexity_sum += term_convexity
    convexity = (convexity_sum / (price * freq ** 2))/10
    return convexity

# User inputs
bond_type = st.selectbox("Bond Type:", ["Corporate", "Treasury", "Municipal", "Agency/GSE", "Fixed Rate"])
price = st.number_input("Price:", min_value=0.0, value=98.5, step=0.01)
annual_coupon_rate = st.number_input("Annual Coupon Rate (%):", min_value=0.0, value=5.0, step=0.01)
coupon_frequency = st.selectbox("Coupon Frequency:", ["Annual", "Semi-Annual", "Quarterly", "Monthly/GSE"])
maturity_date = st.date_input("Maturity Date:", value=datetime.today().date() + relativedelta(years=10))
callable = False
error_message = ""
if bond_type == "Corporate":
    callable = st.checkbox("Callable")
    if callable:
        # Default call date is set to one year before maturity date
        call_date = st.date_input("Call Date:", value=maturity_date - relativedelta(years=1))
        call_price = st.number_input("Call Price:", min_value=0.0, value=100.0, step=0.01)
        if call_date >= maturity_date:
            error_message = "Error: Call date must be earlier than maturity date."

par_value = st.number_input("Par Value:", min_value=0.0, value=100.0, step=0.01)
quantity = st.number_input("Quantity:", min_value=1, value=10, step=1)
settlement_date = st.date_input("Settlement Date:", value=datetime.today().date())
total_markup = st.number_input("Total Markup:", min_value=0.0, value=0.0, step=0.01)
duration_type = st.selectbox("Duration Type:", ["Macaulay", "Modified", "Key Rate"])

# Show error message if call date is invalid
if error_message:
    st.error(error_message)

# Create columns for buttons
col1, col2, _ = st.columns([2, 1, 6])  # Adjusted column widths

# Calculate button
if col1.button("Calculate"):
    if error_message:
        st.error("Cannot calculate because of invalid call date.")
    else:
        # Calculations
        freq_dict = {"Annual": 1, "Semi-Annual": 2, "Quarterly": 4, "Monthly/GSE": 12}
        freq = freq_dict[coupon_frequency]
        n_periods = (maturity_date - settlement_date).days // (365 // freq)
        
        if n_periods <= 0:
            st.error("Error: Settlement date must be before the maturity date.")
        else:
            coupon_payment = annual_coupon_rate / 100 * par_value / freq
            ytm = calculate_ytm(price, par_value, annual_coupon_rate, n_periods, freq)
            ytc = None
            convexity_callable = None
            duration_callable = None
            if callable:
                ytc = calculate_ytc(price, par_value, annual_coupon_rate, call_price, call_date, settlement_date, freq)
                convexity_callable = calculate_convexity_callable(price, par_value, annual_coupon_rate, call_price, call_date, ytm, settlement_date, freq)
                duration_callable = calculate_macaulay_duration(par_value, annual_coupon_rate, ytc, n_periods, freq)
                if duration_type == "Modified":
                    duration_callable = calculate_modified_duration(duration_callable, ytc, freq)
                elif duration_type == "Key Rate":
                    duration_callable = calculate_key_rate_duration(price, par_value, annual_coupon_rate, ytc, n_periods, freq)
            
            macaulay_duration = calculate_macaulay_duration(par_value, annual_coupon_rate, ytm, n_periods, freq)
            if duration_type == "Macaulay":
                duration = macaulay_duration
            elif duration_type == "Modified":
                duration = calculate_modified_duration(macaulay_duration, ytm, freq)
            elif duration_type == "Key Rate":
                duration = calculate_key_rate_duration(price, par_value, annual_coupon_rate, ytm, n_periods, freq)
            
            convexity = calculate_convexity(price, par_value, annual_coupon_rate, ytm, n_periods, freq)
            
            accrued_interest = (datetime.now().date() - settlement_date).days / 365 * (annual_coupon_rate / 100) * par_value
            total_cost = price * quantity + total_markup

            # Create a DataFrame for the output
            output_data = {
                "Metric": ["Coupon Payment", "Number of Periods", "Accrued Interest", "Total Cost", "Yield to Maturity (YTM)", "Duration", "Convexity"],
                "Value": [f"${coupon_payment:.2f}", n_periods, f"${accrued_interest:.2f}", f"${total_cost:.2f}", f"{ytm:.2f}%", f"{duration:.2f} years", f"{convexity:.2f} years"]
            }
            if callable:
                output_data["Metric"].extend(["Yield to Call (YTC)", "Duration (Callable)", "Convexity (Callable)"])
                output_data["Value"].extend([f"{ytc:.2f}%", f"{duration_callable:.2f} years", f"{convexity_callable:.2f} years"])

            df_output = pd.DataFrame(output_data)

            # Convert DataFrame to HTML table with bold headers and white background for headers
            table_html = df_output.to_html(index=False, justify="left")
            table_html = table_html.replace('<table border="1" class="dataframe">', '<table style="width:50%; border-collapse: collapse;">')
            table_html = table_html.replace('<thead>', '<thead style="font-weight: bold; background-color: #ffffff;">')
            table_html = table_html.replace('<th>', '<th style="border: 1px solid black; padding: 8px;">')
            table_html = table_html.replace('<td>', '<td style="border: 1px solid black; padding: 8px;">')
            
            st.markdown(table_html, unsafe_allow_html=True)
            
            # Plotting the graph
            prices = np.linspace(price - 10, price + 10, 50)
            ytm_values = [calculate_ytm(p, par_value, annual_coupon_rate, n_periods, freq) for p in prices]
            ytc_values = [calculate_ytc(p, par_value, annual_coupon_rate, call_price, call_date, settlement_date, freq) for p in prices] if callable else None

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ytm_values, y=prices, mode='lines', name='Yield to Maturity'))
            if ytc_values is not None:
                fig.add_trace(go.Scatter(x=ytc_values, y=prices, mode='lines', name='Yield to Call', line=dict(dash='dash')))
            fig.update_layout(
                xaxis_title="Yield (%)",
                yaxis_title="Price $",
                legend_title="Yields",
                title="Duration"
            )
            st.plotly_chart(fig)

# Reset button
if st.button("Reset"):
    try:
        st.experimental_rerun()
    except Exception as e:
        st.error(f"Error resetting: {e}")
        
# About section
st.markdown("""
### Understanding Bond Prices and Yields:

The relationship between bond prices and yields is fundamental to bond investing. Here's a closer look at how they interact:

#### Key Concepts:

- **Inverse Relationship**: Bond prices and yields generally move in opposite directions. When a bond's price increases, its yield decreases, and vice versa.
- **Yield to Maturity (YTM)**: This is the total return expected on a bond if held until maturity. It accounts for the bond's current market price, par value, coupon interest rate, and time to maturity.
- **Yield to Call (YTC)**: For callable bonds, this is the yield assuming the bond is called (redeemed by the issuer) before its maturity date. It considers the call price and the time until the call date.
- **Yield to Worst (YTW)**: This is the lowest yield an investor can receive if the bond is called or matures early. It is the minimum between YTM and YTC.
- **Duration**: This measures the sensitivity of the bond's price to changes in interest rates. Types of duration include Macaulay Duration, Modified Duration, and Key Rate Duration.
- **Convexity**: This measures the sensitivity of the duration of the bond to changes in interest rates. It provides an estimate of the change in duration for a change in yield.

#### How to Use the Calculator:

1. **Enter Bond Details**: Input the bond's price, par value, coupon rate, and other relevant details.
2. **Calculate Yields**: The calculator computes the YTM and YTC (if the bond is callable) based on your inputs.
3. **Analyze the Chart**: The interactive chart shows how bond prices and yields relate. Hover over the chart to see specific bond and price information that updates dynamically.
4. **Review the Metrics**: The calculator provides key metrics such as coupon payment, number of periods, accrued interest, total cost, yield to maturity, duration, and convexity in a structured table. For callable bonds, additional metrics such as yield to call, callable duration, and callable convexity are also provided.

#### Practical Insights:

- **Investment Decisions**: Understanding the relationship between bond prices and yields helps in making informed investment decisions.
- **Interest Rate Movements**: Keep an eye on interest rate trends, as they significantly impact bond prices and yields.
- **Bond Characteristics**: Different bonds (corporate, municipal, treasury) have unique features and risks. Consider these when analyzing yields.

Use this calculator to explore and understand how changes in bond prices affect yields, helping you optimize your bond investment strategy.
""")
