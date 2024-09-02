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

# User inputs for multiple bonds
num_bonds = st.slider("Select number of bonds (up to 10):", min_value=1, max_value=10, value=1)

bonds = []
for i in range(num_bonds):
    st.write(f"### Bond {i+1}")
    bond_type = st.selectbox(f"Bond Type {i+1}:", ["Corporate", "Treasury", "Municipal", "Agency/GSE", "Fixed Rate"], key=f"bond_type_{i}")
    price = st.number_input(f"Price {i+1}:", min_value=0.0, value=98.5, step=0.01, key=f"price_{i}")
    annual_coupon_rate = st.number_input(f"Annual Coupon Rate {i+1} (%):", min_value=0.0, value=5.0, step=0.01, key=f"coupon_rate_{i}")
    coupon_frequency = st.selectbox(f"Coupon Frequency {i+1}:", ["Annual", "Semi-Annual", "Quarterly", "Monthly/GSE"], key=f"coupon_frequency_{i}")
    maturity_date = st.date_input(f"Maturity Date {i+1}:", value=datetime.today().date() + relativedelta(years=10), key=f"maturity_date_{i}")
    callable = False
    error_message = ""
    if bond_type == "Corporate":
        callable = st.checkbox(f"Callable {i+1}", key=f"callable_{i}")
        if callable:
            # Default call date is set to one year before maturity date
            call_date = st.date_input(f"Call Date {i+1}:", value=maturity_date - relativedelta(years=1), key=f"call_date_{i}")
            call_price = st.number_input(f"Call Price {i+1}:", min_value=0.0, value=100.0, step=0.01, key=f"call_price_{i}")
            if call_date >= maturity_date:
                error_message = f"Error: Call date for Bond {i+1} must be earlier than maturity date."

    par_value = st.number_input(f"Par Value {i+1}:", min_value=0.0, value=100.0, step=0.01, key=f"par_value_{i}")
    quantity = st.number_input(f"Quantity {i+1}:", min_value=1, value=10, step=1, key=f"quantity_{i}")
    settlement_date = st.date_input(f"Settlement Date {i+1}:", value=datetime.today().date(), key=f"settlement_date_{i}")
    total_markup = st.number_input(f"Total Markup {i+1}:", min_value=0.0, value=0.0, step=0.01, key=f"markup_{i}")
    duration_type = st.selectbox(f"Duration Type {i+1}:", ["Macaulay", "Modified", "Key Rate"], key=f"duration_type_{i}")

    bond = {
        "bond_type": bond_type,
        "price": price,
        "annual_coupon_rate": annual_coupon_rate,
        "coupon_frequency": coupon_frequency,
        "maturity_date": maturity_date,
        "callable": callable,
        "call_date": call_date if callable else None,
        "call_price": call_price if callable else None,
        "par_value": par_value,
        "quantity": quantity,
        "settlement_date": settlement_date,
        "total_markup": total_markup,
        "duration_type": duration_type,
        "error_message": error_message
    }
    bonds.append(bond)

# Create columns for buttons
col1, col2, _ = st.columns([2, 1, 6])  # Adjusted column widths

# Calculate button
if col1.button("Calculate"):
    portfolio_duration = 0
    total_value = 0

    for i, bond in enumerate(bonds):
        freq_dict = {"Annual": 1, "Semi-Annual": 2, "Quarterly": 4, "Monthly/GSE": 12}
        freq = freq_dict[bond["coupon_frequency"]]
        n_periods = (bond["maturity_date"] - bond["settlement_date"]).days // (365 // freq)

        if n_periods <= 0:
            st.error(f"Error: Settlement date for Bond {i+1} must be before the maturity date.")
        else:
            coupon_payment = bond["annual_coupon_rate"] / 100 * bond["par_value"] / freq
            ytm = calculate_ytm(bond["price"], bond["par_value"], bond["annual_coupon_rate"], n_periods, freq)
            ytc = None
            convexity_callable = None
            duration_callable = None
            if bond["callable"]:
                ytc = calculate_ytc(bond["price"], bond["par_value"], bond["annual_coupon_rate"], bond["call_price"], bond["call_date"], bond["settlement_date"], freq)
                convexity_callable = calculate_convexity_callable(bond["price"], bond["par_value"], bond["annual_coupon_rate"], bond["call_price"], bond["call_date"], ytm, bond["settlement_date"], freq)
                duration_callable = calculate_macaulay_duration(bond["par_value"], bond["annual_coupon_rate"], ytc, n_periods, freq)
                if bond["duration_type"] == "Modified":
                    duration_callable = calculate_modified_duration(duration_callable, ytc, freq)
                elif bond["duration_type"] == "Key Rate":
                    duration_callable = calculate_key_rate_duration(bond["price"], bond["par_value"], bond["annual_coupon_rate"], ytc, n_periods, freq)

            macaulay_duration = calculate_macaulay_duration(bond["par_value"], bond["annual_coupon_rate"], ytm, n_periods, freq)
            if bond["duration_type"] == "Macaulay":
                duration = macaulay_duration
            elif bond["duration_type"] == "Modified":
                duration = calculate_modified_duration(macaulay_duration, ytm, freq)
            elif bond["duration_type"] == "Key Rate":
                duration = calculate_key_rate_duration(bond["price"], bond["par_value"], bond["annual_coupon_rate"], ytm, n_periods, freq)

            convexity = calculate_convexity(bond["price"], bond["par_value"], bond["annual_coupon_rate"], ytm, n_periods, freq)

            accrued_interest = (datetime.now().date() - bond["settlement_date"]).days / 365 * (bond["annual_coupon_rate"] / 100) * bond["par_value"]
            total_cost = bond["price"] * bond["quantity"] + bond["total_markup"]

            bond_value = bond["price"] * bond["quantity"]
            weighted_duration = duration * bond_value
            portfolio_duration += weighted_duration
            total_value += bond_value

            # Create a DataFrame for the output of each bond
            output_data = {
                "Metric": ["Coupon Payment", "Number of Periods", "Accrued Interest", "Total Cost", "Yield to Maturity (YTM)", "Duration", "Convexity"],
                "Value": [f"${coupon_payment:.2f}", n_periods, f"${accrued_interest:.2f}", f"${total_cost:.2f}", f"{ytm:.2f}%", f"{duration:.2f} years", f"{convexity:.2f} years"]
            }
            if bond["callable"]:
                output_data["Metric"].extend(["Yield to Call (YTC)", "Duration (Callable)", "Convexity (Callable)"])
                output_data["Value"].extend([f"{ytc:.2f}%", f"{duration_callable:.2f} years", f"{convexity_callable:.2f} years"])

            df_output = pd.DataFrame(output_data)

            # Convert DataFrame to HTML table with bold headers and white background for headers
            table_html = df_output.to_html(index=False, justify="left")
            table_html = table_html.replace('<table border="1" class="dataframe">', '<table style="width:50%; border-collapse: collapse;">')
            table_html = table_html.replace('<thead>', '<thead style="font-weight: bold; background-color: #ffffff;">')
            table_html = table_html.replace('<th>', '<th style="border: 1px solid black; padding: 8px;">')
            table_html = table_html.replace('<td>', '<td style="border: 1px solid black; padding: 8px;">')

            st.markdown(f"### Bond {i+1} Results", unsafe_allow_html=True)
            st.markdown(table_html, unsafe_allow_html=True)

    # Calculate and display portfolio duration
    if total_value > 0:
        portfolio_duration /= total_value
        st.write(f"## Portfolio Duration: {portfolio_duration:.2f} years")

    # Plotting the graph for the last bond
    prices = np.linspace(bonds[-1]["price"] - 10, bonds[-1]["price"] + 10, 50)
    ytm_values = [calculate_ytm(p, bonds[-1]["par_value"], bonds[-1]["annual_coupon_rate"], n_periods, freq) for p in prices]
    ytc_values = [calculate_ytc(p, bonds[-1]["par_value"], bonds[-1]["annual_coupon_rate"], bonds[-1]["call_price"], bonds[-1]["call_date"], bonds[-1]["settlement_date"], freq) for p in prices] if bonds[-1]["callable"] else None

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
if col2.button("Reset"):
    st.experimental_rerun()

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
