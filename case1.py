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

# Case 1: Applying PCA to identify patterns in specialty distribution
def run_case_1(data):
    st.title("Case 1: Identifying Patterns in Specialty Distribution")
   
    st.subheader("Problematic: Are certain medical specialties concentrated in specific regions, while others are underserved?")
   
   # Encode specialties numerically
    st.write("We’ll assign a unique number to each specialty. This allows us to analyze the distribution of specialties more abstractly.")
    le = LabelEncoder()
    data['Specialty_Encoded'] = le.fit_transform(data['Spécialité'])
   
   # Display the mapping between original specialties and their encoded values
    specialty_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    st.write("Specialty encoding:", specialty_mapping)
   
    dept_specialty_encoded = data.groupby(['Département', 'Specialty_Encoded']).size().unstack(fill_value=0)
   
   # Standardize the data for PCA
    scaler = StandardScaler()
    dept_specialty_scaled = scaler.fit_transform(dept_specialty_encoded)
   
   # Apply PCA
    pca = PCA(n_components=2)  # Reduce to 2 components for visualization
    principal_components = pca.fit_transform(dept_specialty_scaled)
   
   # Create a DataFrame for the principal components
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'], index=dept_specialty_encoded.index)
   
   # Apply KMeans clustering (with fixed random_state for consistency)
    kmeans = KMeans(n_clusters=3, random_state=42)  # Set random_state for consistent clustering
    pca_df['Cluster'] = kmeans.fit_predict(principal_components)
   
   # Assign consistent colors to clusters
    cluster_colors = {0: 'blue', 1: 'gray', 2: 'red'}
    pca_df['Color'] = pca_df['Cluster'].map(cluster_colors)

   # Visualize the first two principal components with consistent cluster colors
    st.subheader("PCA: Visualizing Departments in Terms of Specialties")
    st.write("Now that we've encoded the specialties numerically, we can apply Principal Component Analysis (PCA) to reduce the dimensionality of the data and uncover patterns or trends that might not be immediately visible.")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='PC1', y='PC2', hue='Cluster', palette=cluster_colors, data=pca_df, ax=ax)
    plt.title("PCA of Departments with Consistent Cluster Colors")
    st.pyplot(fig)
     
    st.write("We have here highlighted different clusters using KMeans clustering.")
    st.write("As we can see we have 3 different clusters, let's try to understand this data distribution:")
   
    
   
    
   # Explained variance ratio
    st.subheader("Explained Variance by Each Principal Component")
    explained_variance = pca.explained_variance_ratio_
    st.write(f"PC1 explains {explained_variance[0]:.2f} of the variance")
    st.write(f"PC2 explains {explained_variance[1]:.2f} of the variance")
    st.write("The majority of departments fall into Cluster 2, which suggests that they have relatively similar distributions of specialties. This large cluster centered around the origin indicates little variation between these departments in terms of the first two principal components.")


   # Identify the outlier department in Cluster 2
    outlier_department = pca_df[pca_df['Cluster'] == 2]
    st.write("Outlier department in Cluster 2:")
    st.write(outlier_department)
   

   
   # Function to plot the specialty distribution in Cluster 2
    def plot_cluster_2_distribution(dept_specialty_encoded, pca_df):
       # Filter the departments in Cluster 2
        cluster_2_departments = pca_df[pca_df['Cluster'] == 2]
       
       # Get the specialty distribution for Cluster 2
        cluster_2_specialty_dist = dept_specialty_encoded.loc[cluster_2_departments.index].sum()
   
       # Calculate the average specialty distribution for all other clusters (Cluster 0 and 1)
        other_clusters_departments = pca_df[pca_df['Cluster'] != 2]
        avg_other_clusters_dist = dept_specialty_encoded.loc[other_clusters_departments.index].mean()
   
      
       # Plot comparison with average specialty distribution in other clusters
        st.subheader("Comparison of Cluster 2 with Average Distribution in Other Clusters")
        comparison_df = pd.DataFrame({
           'Cluster 2': cluster_2_specialty_dist.values,
           'Other Clusters (Average)': avg_other_clusters_dist.values
       }, index=cluster_2_specialty_dist.index)
   
        fig, ax = plt.subplots(figsize=(10, 6))
        comparison_df.plot(kind='bar', ax=ax)
        plt.title("Cluster 2 vs. Average Specialty Distribution in Other Clusters")
        plt.xlabel("Specialty (Encoded)")
        plt.ylabel("Count")
        st.pyplot(fig)
   
   # Call the function to plot Cluster 2 distribution
    plot_cluster_2_distribution(dept_specialty_encoded, pca_df)
     
    # Ensure correct filtering of Cluster 2
    cluster_2_departments = pca_df[pca_df['Cluster'] == 2].index
    
    # Check how many departments are actually in Cluster 2
    st.write(f"Number of departments in Cluster 2: {len(cluster_2_departments)}")
    
    # Display only the departments in Cluster 2
    if len(cluster_2_departments) > 0:
        st.write("Departments in Cluster 2:")
        st.write(cluster_2_departments.tolist())
    else:
        st.write("No departments found in Cluster 2.")
        
    st.header("Key Specialties in Cluster 2")
    
    # Specialty 1
    st.subheader("Chirurgie orthopédique et traumatologie (Specialty 4)")
    st.write("""
    Chirurgie orthopédique et traumatologie is the most dominant specialty in Cluster 2, with a count of around 700 professionals. This suggests that orthopedic and trauma surgery is a significant focus in these departments.
    
    This may indicate that Cluster 2 includes regions or departments where trauma care and orthopedic surgery are particularly important, likely due to higher incidents of trauma-related cases (e.g., accidents, sports injuries, etc.).
    """)
    
    # Specialty 2
    st.subheader("Anesthésie-réanimation; Activité de réanimation; Activité de soins intensifs (Specialty 0)")
    st.write("""
    The second most common specialty is Anesthésie-réanimation, with a count above 500. This high count suggests that intensive care and anesthesiology play a vital role in these departments.
    
    The presence of a large number of anesthesiologists may indicate that these departments also have a focus on surgery or intensive care units (ICUs).
    """)
    
    # Specialty 3
    st.subheader("Gastro-entérologie (activité interventionnelle) (Specialty 10)")
    st.write("""
    Gastro-entérologie is another significant specialty in Cluster 2, with a count near 400. This may point to a higher demand for gastroenterological interventions in these departments.
    """)
    
    # Specialty 4
    st.subheader("Chirurgie viscérale et digestive (Specialty 9)")
    st.write("""
    Chirurgie viscérale et digestive, with a count above 300, is a key specialty in Cluster 2. This surgical specialty focuses on the digestive system, which may indicate that these departments handle a significant number of digestive surgeries (such as gallbladder, liver, and abdominal surgeries).
    """)
    
    # Specialty 5
    st.subheader("Gynécologie-obstétrique (Specialty 11)")
    st.write("""
    Gynécologie-obstétrique also has a high representation in Cluster 2, indicating a strong focus on women's health and maternity services.
    """)
        
    st.title("Departments not in Cluster 2")

    # List of departments
    departments = [
        "06",
        "13",
        "29",
        "33",
        "34",
        "38",
        "44",
        "49",
        "54",
        "59",
        "62",
        "64",
        "67",
        "75",
        "78",
        "83",
        "91",
        "92",
        "93",
        "94"
    ]
    
    # Display the departments as a markdown list
    st.markdown("### Department Codes:")
    for dept in departments:
        st.markdown(f"- {dept}")
        
      
    st.subheader("Potential Urban Concentration")
    
  
    st.write("Some of the excluded departments include major urban centers like:")
    
    
    departments = [
        {
            "code": "13 (Bouches-du-Rhône)",
            "description": "Contains the city of Marseille."
        },
        {
            "code": "75 (Paris)",
            "description": "The capital city of France, which would have a very distinct medical infrastructure."
        },
        {
            "code": "92, 93, 94 (Hauts-de-Seine, Seine-Saint-Denis, and Val-de-Marne)",
            "description": "Suburbs of Paris with distinct healthcare profiles."
        }
    ]
    
    # Display each department and its description
    for dept in departments:
        st.markdown(f"**{dept['code']}**")
        st.write(dept["description"])
        st.write("")  # Add a blank line for spacing
    
    # Conclusion
    st.write(
        "These departments are likely to have higher concentrations of medical professionals "
        "in certain specialties that are not as common in more rural areas, which could explain "
        "why they are not grouped in Cluster 2."
    )
    
    
        # Filter out the departments in Cluster 1
    cluster_1_departments = pca_df[pca_df['Cluster'] == 1].index
    
    # Display the departments in Cluster 1
    st.write("Departments in Cluster 1:")
    st.write(cluster_1_departments.tolist())
    
    # Calculate the specialty distribution for Cluster 1
    cluster_1_specialty_dist = dept_specialty_encoded.loc[cluster_1_departments].sum()
    
    
    
    
    # Filter out the departments in Cluster 0
    cluster_0_departments = pca_df[pca_df['Cluster'] == 0].index
    
    # Calculate the specialty distribution for Cluster 0
    cluster_0_specialty_dist = dept_specialty_encoded.loc[cluster_0_departments].sum()
    cluster_2_specialty_dist = dept_specialty_encoded.loc[cluster_2_departments].sum()
   
    
    # Compare Cluster 0 with Cluster 1 and Cluster 2
    comparison_df = pd.DataFrame({
        'Cluster 0': cluster_0_specialty_dist.values,
        'Cluster 1': cluster_1_specialty_dist.values,
        'Cluster 2': cluster_2_specialty_dist.values
    }, index=cluster_0_specialty_dist.index)
    
    fig, ax = plt.subplots(figsize=(10, 6))

    comparison_df.plot(kind='bar', ax=ax, color=['blue', 'gray', 'red'])
    plt.title("Cluster 0 vs. Cluster 1 vs. Cluster 2: Specialty Distribution Comparison")
    plt.xlabel("Specialty (Encoded)")
    plt.ylabel("Count")
    st.pyplot(fig)
    st.write("")
    
            

    
    # Introduction
    st.write("""
    Cluster 1 represents urban medical centers with a strong presence of specialized services, including **radiology**, **neurosurgery**, and **gynecology**. These departments likely handle more specialized procedures and diagnostics due to the presence of major hospitals in cities like **Paris**, **Marseille**, and **Lyon**.
    """)
    
   
    st.write("""
    Orthopedic surgery is still prominent in urban centers, but Cluster 2 dominates this specialty, likely due to its larger number of departments.
    """)
   
    st.write("""
    The lower overall counts of practitioners in Cluster 1 can be explained by the fact that it only contains three departments, but its focus on advanced and specialized care is clear from the distribution.
    """)
    
    st.header("So what can we say ? ")
    st.subheader("Cluster 2 is Predominantly Regional with Trauma and Surgical Focus")
       
    st.write("""
    Cluster 2 serves a more rural and trauma-focused population, with an emphasis on orthopedic surgery and intensive care.
    """)
    
    st.subheader("Cluster 1 contains Specialized Urban Medical Centers")
    
    st.write("""
    Cluster 1 provides advanced and specialized care in urban centers, focusing on diagnostic imaging, neurosurgery, and complex procedures, and it competes in the number of practitioners with larger clusters despite containing only three departments.
    """)
    
    
    st.subheader("Cluster 0 contains Generalized and Widespread Services")
    
   
   
    st.write("""
    Cluster 0 reflects a more generalized healthcare system, serving a wide range of regions with diverse healthcare needs.
    """)
    

    
        