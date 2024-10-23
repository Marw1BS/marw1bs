import streamlit as st
import pandas as pd
from case1 import run_case_1
from case2 import run_case_2

# Page configuration: Set this as the first Streamlit command
st.set_page_config(
    page_title="Healthcare Data Analysis",
    page_icon=":hospital:",
    layout="wide"
)

# Caching data loading for efficiency
@st.cache_data
def load_data():
    data = pd.read_csv('medecin-accredites-has (2).csv')
    return data

data = load_data()

# Sidebar for navigation
st.sidebar.title("🔎 Investigation Cases")
case = st.sidebar.selectbox("Choose a case to investigate:", 
                            ["Overview", "Case 1: Specialty Distribution", 
                             "Case 2: Accreditation Trends"])

# Main content based on selected case
if case == "Overview":
    # Title and welcome message
    st.title("🏥 Healthcare Investigation Overview")
    st.write(
        """
        Welcome to the **Healthcare Data Analysis App**! This app allows you to explore 
        the distribution of accredited medical professionals across different regions and specialties in France.
        Our goal is to uncover patterns, trends, and potential disparities in healthcare access. 
        Below is a summary of the key areas of investigation.
        """
    )

    # Case 1 overview with icon and button
    st.subheader("🔍 Case 1: Specialty Distribution Across Departments")
    st.write(
        """
        In this case, we investigate the distribution of medical specialties across various departments. 
        By visualizing the concentration of specialties in different regions, we aim to identify:
        - **Geographical disparities** in healthcare access.
        - **Specialty clusters** where certain regions may be underserved or overserved.
        """)
    
    case_1_col1, case_1_col2 = st.columns([0.8, 0.2])
    with case_1_col1:
        st.write("Explore this case to uncover insights about how medical specialties are distributed.")
    with case_1_col2:
        if st.button("🔍 Explore Case 1"):
            run_case_1(data)
    
    # Divider for better structure
    st.markdown("---")

    # Case 2 overview with icon and button
    st.subheader("📈 Case 2: Accreditation Trends Over Time")
    st.write(
        """
        This case analyzes the accreditation trends of medical professionals over time. 
        We track how the number of accredited professionals in various specialties has evolved over the years to:
        - **Highlight growing fields** with an increasing number of professionals.
        - **Identify declining specialties** that may need more attention or support.
        """)
    
    case_2_col1, case_2_col2 = st.columns([0.8, 0.2])
    with case_2_col1:
        st.write("Explore this case to understand accreditation trends over time.")
    with case_2_col2:
        if st.button("📈 Explore Case 2"):
            run_case_2(data)

elif case == "Case 1: Specialty Distribution":
    # Directly run case 1
    run_case_1(data)

elif case == "Case 2: Accreditation Trends":
    # Directly run case 2
    run_case_2(data)

# Footer or sidebar notes
st.sidebar.markdown("---")
st.sidebar.write("Built with ❤️ using Streamlit")
