import streamlit as st
from PIL import Image
from services.pdf_handler import PDFReportGenerator
from datetime import datetime
import pandas as pd

def get_severity_badge(diagnosis):
    """Return severity level and color based on diagnosis"""
    severity_map = {
        'Normal': {'level': '✅ NORMAL', 'color': '#00D084'},  # Green
        'Pneumonia': {'level': '⚠️ PNEUMONIA', 'color': '#FFA500'},  # Orange
        'Tuberculosis': {'level': '🔴 TUBERCULOSIS', 'color': '#FF0000'}  # Red
    }
    return severity_map.get(diagnosis, {'level': diagnosis, 'color': '#808080'})

def show_dashboard(user_email, auth_service, data_service, model_service):
    # Sidebar for User Info & Logout
    st.sidebar.title("🩺 Control Panel")
    st.sidebar.info(f"Radiologist: **{user_email}**")
    
    # Navigation
    page = st.sidebar.radio(
        "Navigation",
        ["📋 Diagnosis", "📊 Analytics"],
        key="nav_radio"
    )
    
    if st.sidebar.button("Logout"):
        auth_service.logout()
        st.rerun()
    
    # Import analytics here to avoid circular imports
    from ui.analytics import show_analytics
    
    # Route to selected page
    if page == "📋 Diagnosis":
        show_diagnosis_page(user_email, auth_service, data_service, model_service)
    else:
        show_analytics(data_service, user_email)

def show_diagnosis_page(user_email, auth_service, data_service, model_service):
    st.title("Clinical Diagnostic Dashboard")

    # SECTION: New Analysis
    st.header("New Analysis")
    
    # Tabs for single and batch upload
    tab1, tab2 = st.tabs(["📄 Single Diagnosis", "📦 Batch Upload"])
    
    # TAB 1: Single Diagnosis
    with tab1:
        with st.expander("Process New X-ray", expanded=True):
            patient_name = st.text_input("Patient Name", placeholder="e.g., P-1024", key="single_patient")
            uploaded_file = st.file_uploader("Upload Chest X-ray", type=['jpg', 'jpeg', 'png'], key="single_upload")

            if uploaded_file and patient_name:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Scan", use_container_width=True)
                
                if st.button("Run AI Diagnosis"):
                    # Store in session state for persistence across reruns
                    with st.spinner("Analyzing anatomical features..."):
                        # Call the Model Service
                        result = model_service.predict(image)
                        st.session_state['last_result'] = result
                        st.session_state['last_image'] = image
                        st.session_state['last_patient_name'] = patient_name
                        
                        # Save the result to Supabase
                        data_service.save_diagnosis(user_email, result, patient_name)
                    
                    # Display results with confidence
                    diagnosis = result['diagnosis']
                    confidence = result['confidence'] * 100
                    severity = get_severity_badge(diagnosis)
                    
                    st.success(f"Analysis Complete!")
                    
                    # Show diagnosis with color badge and confidence
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Diagnosis", severity['level'])
                    with col2:
                        st.metric("Confidence", f"{confidence:.1f}%")
                    with col3:
                        # Show severity indicator
                        st.markdown(f"<div style='background-color:{severity['color']};padding:20px;border-radius:10px;text-align:center;color:white;font-weight:bold;'>SEVERITY</div>", unsafe_allow_html=True)
                    
                    # Show all probabilities
                    st.subheader("Prediction Probabilities:")
                    for class_name, prob in result['probabilities'].items():
                        st.progress(prob, text=f"{class_name}: {prob*100:.1f}%")
                    
                    # Generate and display heatmap (with caching)
                    st.divider()
                    st.subheader("🔥 AI Interpretation Heatmap")
                    st.info("Red areas indicate regions that most influenced the AI diagnosis. Darker blue indicates lower influence.")
                    
                    with st.spinner("Generating interpretation map (this may take 5-10 seconds)..."):
                        try:
                            # Pass the result to avoid redundant prediction
                            heatmap_image = model_service.generate_heatmap(image, result)
                            st.session_state['last_heatmap'] = heatmap_image
                            st.image(heatmap_image, caption="Grad-CAM Attention Map - Shows which regions influenced the diagnosis", use_container_width=True)
                        except Exception as e:
                            st.warning(f"Could not generate heatmap: {str(e)}")
                    
                    # Generate and offer PDF download
                    st.divider()
                    pdf_generator = PDFReportGenerator()
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    pdf_data = pdf_generator.generate_diagnosis_report(
                        patient_name=patient_name,
                        diagnosis=diagnosis,
                        confidence=confidence,
                        probabilities=result['probabilities'],
                        radiologist_email=user_email,
                        timestamp=timestamp
                    )
                    
                    st.download_button(
                        label="📥 Download Report (PDF)",
                        data=pdf_data,
                        file_name=f"Diagnosis_{patient_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf"
                    )
    
    # TAB 2: Batch Upload
    with tab2:
        st.subheader("Process Multiple X-rays")
        st.info("Upload multiple X-ray images with their patient IDs to process them in batch.")
        
        # Batch upload section
        with st.form("batch_form"):
            # CSV input for patient IDs
            st.write("**Step 1:** Enter Patient IDs (one per line)")
            batch_patient_ids = st.text_area(
                "Patient IDs",
                placeholder="P-1001\nP-1002\nP-1003",
                height=100,
                key="batch_ids"
            )
            
            # Multiple file upload
            st.write("**Step 2:** Upload corresponding X-ray images")
            uploaded_files = st.file_uploader(
                "Upload multiple X-rays",
                type=['jpg', 'jpeg', 'png'],
                accept_multiple_files=True,
                key="batch_upload"
            )
            
            submit_batch = st.form_submit_button("🚀 Process Batch", use_container_width=True)
        
        if submit_batch:
            patient_ids = [id.strip() for id in batch_patient_ids.strip().split('\n') if id.strip()]
            
            if not patient_ids or not uploaded_files:
                st.error("Please provide both patient IDs and X-ray images")
            elif len(patient_ids) != len(uploaded_files):
                st.error(f"Mismatch: {len(patient_ids)} patient IDs provided but {len(uploaded_files)} files uploaded")
            else:
                st.subheader("Processing Results")
                results = []
                
                # Process each file
                progress_bar = st.progress(0)
                status_placeholder = st.empty()
                results_container = st.container()
                
                for idx, (patient_id, uploaded_file) in enumerate(zip(patient_ids, uploaded_files)):
                    status_placeholder.write(f"Processing {idx + 1}/{len(uploaded_files)}: {patient_id}")
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                    
                    try:
                        # Open and predict
                        image = Image.open(uploaded_file)
                        result = model_service.predict(image)
                        
                        # Save to database
                        data_service.save_diagnosis(user_email, result, patient_id)
                        
                        results.append({
                            'Patient ID': patient_id,
                            'Diagnosis': result['diagnosis'],
                            'Confidence': f"{result['confidence']*100:.1f}%",
                            'Status': '✅ Success'
                        })
                    except Exception as e:
                        results.append({
                            'Patient ID': patient_id,
                            'Diagnosis': 'Error',
                            'Confidence': 'N/A',
                            'Status': f'❌ {str(e)[:30]}'
                        })
                
                status_placeholder.empty()
                progress_bar.empty()
                
                # Display results table
                results_df = pd.DataFrame(results)
                st.dataframe(results_df, use_container_width=True, hide_index=True)
                
                # Summary statistics
                successful = len([r for r in results if r['Status'] == '✅ Success'])
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Processed", len(results))
                with col2:
                    st.metric("Successful", successful)
                with col3:
                    st.metric("Failed", len(results) - successful)
                
                st.success(f"Batch processing complete! {successful}/{len(results)} files processed successfully.")
                st.rerun()

    # SECTION: History Table
    st.divider()
    st.header("Past Diagnostic History")
    history = data_service.get_history(user_email)
    
    if history.data:
        # Add severity badge to history data
        df = pd.DataFrame(history.data)
        df['Severity'] = df['prediction'].apply(lambda x: get_severity_badge(x)['level'])
        
        # Format timestamps to readable format
        if 'created_at' in df.columns:
            df['created_at'] = pd.to_datetime(df['created_at']).dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Search and Filter Section
        st.subheader("Search & Filter")
        search_col, diagnosis_col = st.columns(2)
        
        with search_col:
            search_patient = st.text_input("🔍 Search Patient ID", placeholder="e.g., P-1025")
        
        with diagnosis_col:
            diagnoses = ['All'] + df['prediction'].unique().tolist()
            selected_diagnosis = st.selectbox("Filter by Diagnosis", diagnoses)
        
        # Apply filters
        filtered_df = df.copy()
        
        if search_patient:
            filtered_df = filtered_df[filtered_df['patient_name'].str.contains(search_patient, case=False, na=False)]
        
        if selected_diagnosis != 'All':
            filtered_df = filtered_df[filtered_df['prediction'] == selected_diagnosis]
        
        # Display filtered history with download buttons
        st.subheader(f"Results: {len(filtered_df)} record(s)")
        
        # Create table header
        col1, col2, col3, col4, col5, col6 = st.columns([1.5, 1.5, 2, 1.2, 1.5, 1.5])
        with col1:
            st.write("**Patient ID**")
        with col2:
            st.write("**Diagnosis**")
        with col3:
            st.write("**Date & Time**")
        with col4:
            st.write("**Confidence**")
        with col5:
            st.write("**Severity**")
        with col6:
            st.write("**Action**")
        
        st.divider()
        
        # Display each record with download button in same row
        for idx, (_, record) in enumerate(filtered_df.iterrows()):
            col1, col2, col3, col4, col5, col6 = st.columns([1.5, 1.5, 2, 1.2, 1.5, 1.5])
            
            with col1:
                st.write(record['patient_name'])
            with col2:
                st.write(record['prediction'])
            with col3:
                st.write(record['created_at'])
            with col4:
                st.write(f"{record.get('confidence_score', 'N/A')}%")
            with col5:
                st.write(record['Severity'])
            with col6:
                if st.button("📥 Download", key=f"download_{idx}", use_container_width=True):
                    pdf_generator = PDFReportGenerator()
                    probabilities = {
                        'Normal': 0.33,
                        'Pneumonia': 0.33,
                        'Tuberculosis': 0.34
                    }
                    pdf_data = pdf_generator.generate_diagnosis_report(
                        patient_name=record.get('patient_name', 'Unknown'),
                        diagnosis=record.get('prediction', 'Unknown'),
                        confidence=record.get('confidence_score', 0),
                        probabilities=probabilities,
                        radiologist_email=record.get('radiologist_email', user_email),
                        timestamp=record.get('created_at', 'Unknown')
                    )
                    st.download_button(
                        label="Click to Download PDF",
                        data=pdf_data,
                        file_name=f"Report_{record.get('patient_name')}_{record.get('created_at', '').replace(' ', '_').replace(':', '')}.pdf",
                        mime="application/pdf",
                        key=f"btn_{idx}",
                        use_container_width=True
                    )
    else:
        st.info("No records found in your portal.")