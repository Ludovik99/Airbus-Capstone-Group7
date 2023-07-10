import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error

data_folder = "Data"
csv_files = sorted([file for file in os.listdir(data_folder) if file.endswith(".csv")])

# Landing page
def landing_page():
    # Set background image
    st.markdown(
        f"""
         <style>
         .stApp {{
             background-image: url("https://mediarenditions.airbus.com/_FlB7HEJU3QR1lTZQJr4OGhR_cK6uT4UEKh3eh2CWHQ/resize?src=kpkp://airbus/38/542/542497-7sshfapqbs.jpg&w=1920&h=1920&t=fit");
             background-attachment: fixed;
             background-size: cover;
             color: white;
         }}
         </style>
         """,
         unsafe_allow_html=True
     )
    
    # Add content on the landing page
    st.markdown("<h1 style='color: white;'>Airbus: Anomaly Detection</h1>", unsafe_allow_html=True)
    st.write("Prepared by Lia Dollison, Ludovico Gandolfi, Adam Jamison, Iván López and Roman Zotkin")

# Input page
#@st.cache(allow_output_mutation=True, suppress_st_warning=True)
#@st.cache_data(experimental_allow_widgets=True) 

def input_page():
    tab1, tab2, tab3 = st.tabs(["Scree plot", "Reconstruction Error", "Anomalies"])
    with tab1:
        st.header("Principal Component Analysis")
        # Sidebar select boxes
    
        selected_dataset = st.sidebar.selectbox("Select dataset:", csv_files)
        if selected_dataset:
            file_path = os.path.join(data_folder, selected_dataset)
            data = pd.read_csv(file_path, delimiter=';')
        
            # Pre-processing the dataset
            large_dataset_columns = ['UTC_TIME', 'MSN', 'Flight', 'day', 'month', 'time', 'year', 'FUEL_USED_1', 'FUEL_USED_2', 'FUEL_USED_3', 'FUEL_USED_4']
            small_dataset_columns = ['UTC_TIME', 'MSN', 'Flight', 'FUEL_USED_1', 'FUEL_USED_2', 'FUEL_USED_3', 'FUEL_USED_4', 'FLIGHT_PHASE_COUNT']
            if data.shape[1] == 111:
                pca_data = data.drop(large_dataset_columns, axis=1)
                #pca_data.drop(pca_data[(pca_data["FLIGHT_PHASE_COUNT"] == 1) | (pca_data["FLIGHT_PHASE_COUNT"] == 2) | (pca_data["FLIGHT_PHASE_COUNT"] == 12)].index, inplace=True)
                column_means = pca_data.mean()
                pca_data = pca_data.fillna(column_means)
                pca_data.dropna(inplace=True)
            
            else:
                pca_data = data.drop(data[(data["FLIGHT_PHASE_COUNT"] == 1) | (data["FLIGHT_PHASE_COUNT"] == 2) | (data["FLIGHT_PHASE_COUNT"] == 12)].index)
                pca_data.drop(small_dataset_columns, axis=1, inplace=True)
                data_nodev = data.drop(data[(data["FLIGHT_PHASE_COUNT"] == 1) | (data["FLIGHT_PHASE_COUNT"] == 2) | (data["FLIGHT_PHASE_COUNT"] == 12)].index)
                data_nodev.drop(['FUEL_USED_1','FUEL_USED_2','FUEL_USED_3','FUEL_USED_4'], axis=1, inplace=True)

                
                columns_to_scale = pca_data.columns
                for column in columns_to_scale:
                    pca_data[column] = pca_data[column].diff()
                    data_nodev[f"Dx_{column}"] = data_nodev[column].diff()
                pca_data.dropna(inplace=True)
                data_nodev.dropna(inplace=True)
                scaler = RobustScaler()
                pca_data[columns_to_scale] = scaler.fit_transform(pca_data[columns_to_scale])
        
            # Perform PCA
            pca = PCA()
            pca.fit(pca_data)
        
            # Calculate the explained variance ratio
            explained_variance_ratio = pca.explained_variance_ratio_
            cumulative_variance = np.cumsum(explained_variance_ratio)
        
            # Create a Scree dataframe for visualisation
            scree_df = pd.DataFrame({'Number of Components': np.arange(1, len(explained_variance_ratio) + 1),
                         'Cumulative Explained Variance': cumulative_variance})
        
            # Plot the scree plot using Plotly Express
            fig1 = px.line(scree_df, x='Number of Components', y='Cumulative Explained Variance',
              title='Scree Plot', labels={'Cumulative Explained Variance': 'Cumulative Explained Variance'})
            fig1.update_layout(xaxis_title='Number of Components', yaxis_title='Cumulative Explained Variance', xaxis=dict(tickmode='linear',dtick=1))
            fig1.update_traces(mode="lines+markers", marker_size=10, line_width=3, error_y_color="gray", error_y_thickness=1, error_y_width=10)
            st.plotly_chart(fig1, use_container_width=True)
    
        # Let the user select the number of components manually
        
        components = [2,3,4,5,6,7,8,9]
        n_components_manual = st.sidebar.selectbox("Select number of components",components)
    
        # Perform PCA with Minka's MLE
        pca_mle = PCA(n_components='mle')      
        data_pca_mle = pca_mle.fit_transform(pca_data)
        data_reconstructed_mle = pca_mle.inverse_transform(data_pca_mle)
        reconstruction_error_mle = mean_squared_error(pca_data, data_reconstructed_mle)
        n_components_mle = pca_mle.n_components_
    
        # Perform PCA with a manually selected number of components
        pca_manual = PCA(n_components=n_components_manual)
        data_pca_manual = pca_manual.fit_transform(pca_data)
        data_reconstructed_manual = pca_manual.inverse_transform(data_pca_manual)
        reconstruction_error_manual = mean_squared_error(pca_data, data_reconstructed_manual)
    
        st.write("Reconstruction Error - Automatic, {} components: ".format(str(n_components_mle)), reconstruction_error_mle)
        st.write("Reconstruction Error - Manual, {} components: ".format(str(n_components_manual)), reconstruction_error_manual)
        if reconstruction_error_mle < reconstruction_error_manual:
            st.write("Suggest to select the same number of components as Automatic")
    
    with tab2:
        st.header("Principal Component Analysis")
        
        #Getting the number of components from the Minka's MLE method defined above
        n_components_mle = pca_mle.n_components_

        # Initialize lists to store the number of components and the corresponding reconstruction errors
        num_components_list = []
        reconstruction_error_list = []

        for n in range(1, n_components_mle + 1):
            # Perform PCA with 'n' components
            pca = PCA(n_components=n)
            data_pca = pca.fit_transform(pca_data)
            data_reconstructed = pca.inverse_transform(data_pca)

            # Calculate reconstruction error
            reconstruction_error = mean_squared_error(pca_data, data_reconstructed)

            # Append the number of components and reconstruction error to the lists
            num_components_list.append(n)
            reconstruction_error_list.append(reconstruction_error)
            
        # Create a dataframe with the data
        error_df = pd.DataFrame({'Number of Components': num_components_list, 'Reconstruction Error': reconstruction_error_list})

        # Create an interactive scatter plot using Plotly
        fig2 = px.scatter(error_df, x='Number of Components', y='Reconstruction Error', hover_data=['Reconstruction Error'])
        fig2.update_traces(mode='lines+markers')
        fig2.update_layout(
    title='Reconstruction Error vs Number of Components',
    xaxis_title='Number of Components',
    yaxis_title='Reconstruction Error'
)
        st.plotly_chart(fig2, use_container_width=True)
    
    with tab3:
        st.header("Anomalies")
        
        options_length = [10,20,30,40,50,60,70,80,90,100]
        default_value = 30
        sequence_length = st.sidebar.selectbox("Select anomaly length (consecutive seconds):", options_length, index=options_length.index(default_value))
        
        options_stdev = [0.5, 1.0, 1.5, 2.0]
        number_of_stdev = st.sidebar.selectbox("Select threshold in st. deviations for the reconstruction error:", options_stdev)
        
        # Reconstruct data from reduced dimensions with the number of PCA components chosen by user
        data_reconstructed2 = data_reconstructed_manual
        
        # Compute reconstruction error as mean square error between original and reconstructed data
        reconstruction_error2 = ((pca_data - data_reconstructed2) ** 2).mean(axis=1)
        
        # Identify samples with high reconstruction errors as potential anomalies
        threshold = reconstruction_error2.mean() + (number_of_stdev * reconstruction_error2.std())
        anomaly_indices = reconstruction_error2[reconstruction_error2 > threshold].index
        
        positive_values = []
        negative_values = []

        for index in anomaly_indices:
            value = pca_data.at[index, 'VALUE_FOB']
            if value > 0:
                positive_values.append(index)
            elif value < 0:
                negative_values.append(index)
        
        
        
        sequences = []
        current_sequence = []
        seql = []

        # Iterate over the numbers list
        for i in range(len(negative_values)-1):
            # Check if the next number is consecutive
            if negative_values[i+1] == negative_values[i] + 1 or 0 < negative_values[i+1] - negative_values[i] <= 5:
                # If it is, add the current number to the current sequence
                current_sequence.append(negative_values[i])
            else:
                # If it's not, add the current number and the current sequence to the sequences list
                current_sequence.append(negative_values[i])
                sequences.append(current_sequence)
                # Reset the current sequence
                current_sequence = []

        # Add the last number to the last sequence
        current_sequence.append(negative_values[-1])
        sequences.append(current_sequence)

        # Print sequences that are over the defined limit of seconds long
        for seq in sequences:
            if len(seq) > sequence_length:
                seql.append(seq)
                #st.write(seq)
        
        # Flatten the 'anomaly_indices' list
        anomaly_indices_flat = [index for sublist in seql for index in sublist]
        
        # Specify the column names 
        columns_needed = ['UTC_TIME', 'Flight', 'FLIGHT_PHASE_COUNT', 'VALUE_FUEL_QTY_FT1', 'VALUE_FUEL_QTY_FT2', 'VALUE_FUEL_QTY_FT3', 'VALUE_FUEL_QTY_FT4', 'VALUE_FUEL_QTY_LXT', 'VALUE_FUEL_QTY_RXT', 'VALUE_FUEL_QTY_CT']

        # Create the dataframe 'anomaly_data' using flattened 'anomaly_indices'
        anomaly_data = data.loc[anomaly_indices_flat, columns_needed]

        flights_df = anomaly_data.groupby('Flight')

        # Create a dictionary to store the separate DataFrames for each flight
        flight_dfs = {}
        
        # Iterate over each group (flight) and store the corresponding DataFrame in the dictionary
        for flight_number, flight_data in flights_df:
            flight_dfs[flight_number] = flight_data
        
        for flight_number, flight_df in flight_dfs.items():
            st.subheader(f"Flight Number: {flight_number}")
            i = 1

            # Filter the flight DataFrame for each sequence in 'seql'
            for sequence in seql:
                sequence_df = flight_df[flight_df.index.isin(sequence)]
                

                if not sequence_df.empty:
                    st.write(f"Sequence: {i}, number of anomalies: {len(sequence)}")
                    utc_times = sequence_df.loc[sequence, 'UTC_TIME']
                    first_time = utc_times.iloc[0]
                    last_time = utc_times.iloc[-1]
                    st.write(f"Start: {first_time}, end: {last_time}")
                    i += 1
                    
                    subset_dev = pca_data.loc[sequence]
                    subset_og = data_nodev.loc[sequence]

                    average_values = subset_dev.mean()

                    common_flight_phase = subset_og['FLIGHT_PHASE_COUNT'].mode()

                    # Calculate the average value for each feature within the subset
                    filtered_phase = data_nodev.loc[data_nodev['FLIGHT_PHASE_COUNT']==common_flight_phase[0]]
                    filtered_phase_index = filtered_phase.index
                    filtered_pca = pca_data.loc[filtered_phase_index]

                    original_means = filtered_pca.mean()

                    diff = abs(average_values - original_means)
                    diff = diff.drop("VALUE_FOB")

                    # Find the top eatures with the largest absolute differences
                    top_feature = diff.nlargest(1)
                    st.write(f"Top feature causing the anomaly: {top_feature.index[0]}")
                    st.dataframe(sequence_df)

                    st.write("---")  # Add a separator between each flight   
    
            
# Main app
def main():
    st.set_page_config(layout="wide", initial_sidebar_state="collapsed")

    # Page navigation
    pages = {
        "Landing Page": landing_page,
        "Input": input_page,
    }

    # Sidebar navigation
    current_page = st.sidebar.selectbox("Navigate", list(pages.keys()))

    # Render the selected page
    pages[current_page]()

# Run the app
if __name__ == "__main__":
    main()
