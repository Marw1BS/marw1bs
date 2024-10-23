import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# Load your dataset
@st.cache_data
def load_data():
    data = pd.read_csv('medecin-accredites-has (2).csv')
    return data

data = load_data()


def run_case_2(data):
   
    st.title("Case 2: Accreditation Trends Over Time ")
   
    st.subheader("How have the accreditation trends of medical professionals evolved over time across different specialties, and which fields are experiencing significant growth or decline? ")
    st.header("Key Questions for Analysis:")

    st.write("""
    **1. Which specialties have shown the most growth in terms of accredited professionals?**
    
    **2. Are there any specialties that are declining in terms of the number of accredited practitioners?**
    
    **3. Is there a shift in the demand for accreditation in certain specialties over time (e.g., newer medical fields or emerging technologies)?**
    
    **4. How do these trends compare across urban vs. rural departments or specific regions?**
    
    **5. What might be the underlying causes of the observed trends (e.g., aging populations, technological advancements, public health priorities)?**
    """)
 
    # Convert the 'Date accréditation' column to datetime format and extract the year
    data['Date accréditation'] = pd.to_datetime(data['Date accréditation'], format='%d/%m/%Y', errors='coerce')
    data['Year'] = data['Date accréditation'].dt.year
    
    # Group by year and specialty to count the number of accreditations per year for each specialty
    accreditation_trends = data.groupby(['Year', 'Spécialité']).size().unstack(fill_value=0)
    
    # Plotting the accreditation trends over time
    st.title("Accreditation Trends Over Time by Specialty")
    st.subheader("First thing let us see our data : What are we dealing with ? ")
    fig, ax = plt.subplots(figsize=(12, 6))
    accreditation_trends.plot(kind='line', ax=ax)
    plt.title('Accreditation Trends Over Time by Specialty')
    plt.xlabel('Year')
    plt.ylabel('Number of Accreditations')
    plt.legend(title='Specialty', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Display the plot in Streamlit
    st.pyplot(fig)
   
       # Calculate total accreditations per specialty across all years
    total_accreditations = accreditation_trends.sum()
    
    # Separate into lowest and highest specialties based on total accreditations
    lowest_accreditations = total_accreditations.nsmallest(5).index  # 5 lowest specialties
    highest_accreditations = total_accreditations.nlargest(5).index  # 5 highest specialties
    
    # Filter the data for the lowest and highest accreditations
    lowest_accreditation_trends = accreditation_trends[lowest_accreditations]
    highest_accreditation_trends = accreditation_trends[highest_accreditations]
    
    # Get the maximum value for the y-axis from both datasets to use the same scale
    y_max = max(lowest_accreditation_trends.max().max(), highest_accreditation_trends.max().max())
    # Plotting the highest accreditation trends
    st.title("Highest Accreditation Trends Over Time by Specialty")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    highest_accreditation_trends.plot(kind='line', ax=ax)
    plt.title('Highest Accreditation Trends Over Time by Specialty')
    plt.xlabel('Year')
    plt.ylabel('Number of Accreditations')
    plt.ylim(0, y_max)  # Set the same y-axis limit
    plt.legend(title='Specialty', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    st.pyplot(fig)
    # Plotting the lowest accreditation trends
    st.title("Lowest Accreditation Trends Over Time by Specialty")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    lowest_accreditation_trends.plot(kind='line', ax=ax)
    plt.title('Lowest Accreditation Trends Over Time by Specialty')
    plt.xlabel('Year')
    plt.ylabel('Number of Accreditations')
    plt.ylim(0, y_max)  # Set the same y-axis limit
    plt.legend(title='Specialty', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    st.pyplot(fig)
    
   
    st.subheader("Lets zoom in")

    fig, ax = plt.subplots(figsize=(12, 6))
    lowest_accreditation_trends.plot(kind='line', ax=ax)
    plt.title('Lowest Accreditation Trends Over Time by Specialty')
    plt.xlabel('Year')
    plt.ylabel('Number of Accreditations')
    plt.legend(title='Specialty', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    st.pyplot(fig)     
    
    
    st.write("What's important to notice here is that for all specialitise the schema is the same. In 2019 we have a strong augmentation of accreditation. ")
    st.subheader("Let us see which ones and why : ")
        
    
            
  
    
    # Group the data by Year and Specialty to get total accreditations per year for each specialty
    accreditation_growth = data.groupby(['Year', 'Spécialité']).size().unstack(fill_value=0)
  
         # Calculate the growth by subtracting the accreditations in 2020 from those in 2024
    growth = accreditation_growth.loc[2024] - accreditation_growth.loc[2020]
    
    # Sort specialties by the most growth
    most_growth_specialties = growth.sort_values(ascending=False)
    
    # Display the top 5 specialties with the most growth
    st.write("Specialties with the most growth from 2020 to 2024:")
    st.write(most_growth_specialties.head())
      
    
        
    st.subheader("1. Anesthésie-réanimation (Intensive Care, Reanimation, and Anesthesia)")
    
    st.write("""
    **-COVID-19 Pandemic**: One of the main reasons for the significant growth in this specialty is likely the increased demand for intensive care services and anesthesiologists due to the COVID-19 pandemic. Intensive care units became critical during the pandemic, which led to a surge in the demand for accredited professionals in this field.
    
    **-Critical Surgical Care**: Anesthesia is required for most surgeries, and there has likely been an ongoing demand for these professionals to support surgical care, especially as elective surgeries ramped up post-pandemic.
    """)
    
    st.subheader("2. Gastro-entérologie (Interventional Gastroenterology)")

    st.write("""
    **-Long COVID Symptoms**: Many COVID-19 survivors have reported persistent gastrointestinal symptoms, including abdominal pain, diarrhea, nausea, and loss of appetite, long after recovering from the initial infection. This has led to an increase in demand for gastroenterologists to diagnose and manage these symptoms.
    
    **-Gastrointestinal Manifestations of COVID-19**: Studies have shown that COVID-19 can directly affect the gastrointestinal tract, leading to conditions like gastroenteritis or colitis. Some patients also report inflammatory bowel conditions post-infection, further driving the need for specialists in this field.
    """)
     
    st.subheader(" Are there any specialties that are declining in terms of the number of accredited practitioners?")   
           # Calculate the change by subtracting the accreditations in 2020 from those in 2024
    growth = accreditation_growth.loc[2024] - accreditation_growth.loc[2020]
    
    # Filter out specialties where the number of accreditations has decreased (negative growth)
    declining_specialties = growth[growth < 0].sort_values()
    
    # Create a DataFrame to store the decline and the peak year
    declining_specialties_df = pd.DataFrame(declining_specialties, columns=['Decline'])
    
    # Find the year with the maximum accreditations for each declining specialty
    declining_specialties_df['Peak Year'] = [accreditation_growth[specialty].idxmax() for specialty in declining_specialties.index]
    
    # Display the specialties with the most decline and their peak year
    st.write("Specialties with declining accreditations from 2020 to 2024 and their peak year:")
    st.write(declining_specialties_df)
    
    st.write("The thing that catches the eye here is that gynécologie specialisation is declining significantly fast. Maybe we coould look for reasons to that.")
    st.write("Let's see if departements hava something to do with it.")
        
    # Convert 'Date accréditation' column to datetime and extract year
    data['Date accréditation'] = pd.to_datetime(data['Date accréditation'], format='%d/%m/%Y', errors='coerce')
    data['Year'] = data['Date accréditation'].dt.year
    
    # Filter the data to include only the specialty Gynécologie-obstétrique
    gynecology_data = data[data['Spécialité'].str.contains('Gynécologie-obstétrique', na=False)]
    
    # Group the gynecology data by 'Year' and 'Département' to count the number of accreditations per department each year
    gynecology_by_dept_year = gynecology_data.groupby(['Year', 'Département']).size().unstack(fill_value=0)
    gynecology_decline = gynecology_by_dept_year.loc[2024] - gynecology_by_dept_year.loc[2020]
    
    # Sort departments by the most significant decline
    gynecology_decline_sorted = gynecology_decline.sort_values(ascending=True)
    
    # Display the departments with the most decline in gynecology accreditations
    st.write("Departments with the most decline in Gynécologie-obstétrique accreditations from 2020 to 2024:")
    st.write(gynecology_decline_sorted)
  
    fig, ax = plt.subplots(figsize=(10, 6))
    gynecology_decline_sorted.plot(kind='bar', ax=ax)
    plt.title('Decline in Gynécologie-obstétrique Accreditations by Department (2020-2024)')
    plt.xlabel('Department')
    plt.ylabel('Decline in Number of Accreditations')
    st.pyplot(fig)
    
    st.write("As we can see Gynécologie-obstétrique accreditations does appear to be regionally distributed, and there doesn’t seem to be an extreme or disproportionate drop in any single region, except for a few outliers.")
    
    st.title("Overall Conclusion")
    
    st.write("""
    The analysis reveals a clear impact of the COVID-19 pandemic on the accreditation trends of medical professionals, with critical care and diagnostics fields experiencing the most significant growth. On the other hand, **gynécologie-obstétrique** stands out as a specialty facing a notable decline, which may be the result of changing population needs and healthcare priorities.
    
    Further analysis could explore whether these trends continue post-pandemic or if other external factors, such as policy changes or technological advancements, will further alter the landscape of medical professional accreditations.
    """)